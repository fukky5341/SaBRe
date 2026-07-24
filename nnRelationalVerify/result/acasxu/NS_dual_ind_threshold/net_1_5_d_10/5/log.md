## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 554.967677004936


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-148.1843414, 471.4650574, -148.1843414, 471.4650574, -619.6494141, 619.6494141)
1: (-208.4494476, 474.2722168, -208.4494476, 474.2722168, -682.7216797, 682.7216797)
2: (-176.0516663, 524.4273071, -176.0516663, 524.4273071, -700.4790039, 700.4790039)
3: (-185.7463837, 673.9080200, -185.7463837, 673.9080200, -859.6543579, 859.6543579)
4: (-158.1413727, 615.8510742, -158.1413727, 615.8510742, -773.9923706, 773.9923706)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.73 + 2.33 = 3.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -554.9898766, upper bound: 554.9898766

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9815300, upper bound: 554.9814249
time: 0.90 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784369, upper bound: 554.9784369
time: 0.98 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.94 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.94
Output dim: 0, lower bound: -554.9815300, upper bound: 554.9814249
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.94
Output dim: 0, lower bound: -554.9784369, upper bound: 554.9784369

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -135.4273682, 427.6317749, -145.9505463, 464.0416870, -599.4690552, 573.5823364
1: -189.9990234, 430.8988953, -205.2934418, 466.8550415, -656.8540649, 636.1920776
2: -160.4833527, 476.7970886, -173.3911743, 516.2078857, -676.6912231, 650.1881104
3: -169.3506470, 612.0695801, -182.9147797, 663.2875366, -832.6381226, 794.9842529
4: -144.2209015, 559.9951782, -155.7505646, 606.1411133, -750.3619995, 715.7457275

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
time: 1.20 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9813566
time: 1.26 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -141.9587555, 452.3752747, -144.1129913, 458.9883728, -600.9471436, 596.4882812
1: -199.8379517, 454.9771729, -202.8209686, 461.6579285, -661.4958496, 657.7981567
2: -168.7648926, 503.0252075, -171.2863922, 510.4345398, -679.1992188, 674.3115845
3: -178.0149994, 646.6646729, -180.6889801, 656.0998535, -834.1148682, 827.3536377
4: -151.5778503, 590.7100830, -153.8487396, 599.4162598, -750.9941406, 744.5587158

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784369, upper bound: 554.9777726
time: 0.93 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9784067, upper bound: 554.9784067
time: 1.07 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.74 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9813566
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 0, lower bound: -554.9784369, upper bound: 554.9777726
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 0, lower bound: -554.9784067, upper bound: 554.9784067

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -131.6882477, 415.2092590, -126.8219757, 402.9844360, -534.6726685, 542.0311890
1: -184.7559814, 418.5219116, -178.4099426, 405.5857849, -590.3417969, 596.9316406
2: -156.1006165, 463.0516663, -150.8152466, 448.2987366, -604.3993530, 613.8668213
3: -164.6693573, 594.2855835, -158.9646454, 576.7651367, -741.4344482, 753.2502441
4: -140.3057556, 543.8449707, -135.5935974, 526.7003784, -667.0061035, 679.4385376

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
time: 0.88 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -132.6952209, 418.5322876, -141.5488281, 449.3009949, -581.9962158, 560.0809937
1: -186.0959778, 421.9254761, -199.0150146, 452.2501831, -638.3461304, 620.9404907
2: -157.2136993, 466.9020996, -168.1295929, 500.1103821, -657.3240356, 635.0316772
3: -165.8747559, 599.1263428, -177.3263550, 642.3260498, -808.2007446, 776.4525146
4: -141.2933502, 548.3496094, -151.0378265, 587.2087402, -728.5020752, 699.3873901

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9760235, upper bound: 554.9766658
time: 1.26 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9751264, upper bound: 554.9766658
time: 0.89 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -138.8268585, 442.0000916, -125.2509460, 398.7146301, -537.5414429, 567.2510376
1: -195.4588470, 444.5378418, -176.2799225, 401.1724243, -596.6312866, 620.8177490
2: -165.0877228, 491.4695435, -149.0080872, 443.4668579, -608.5545654, 640.4775391
3: -174.0976410, 631.6799316, -157.0465393, 570.6887207, -744.7861938, 788.7263184
4: -148.2701874, 577.1235962, -133.9428101, 521.0123291, -669.2824097, 711.0663452

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9562249, upper bound: 554.9627269
time: 1.19 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9780442, upper bound: 554.9760236
time: 1.08 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -138.9973602, 442.4841614, -139.7932892, 444.5494690, -583.5468140, 582.2774658
1: -195.6101074, 445.1821289, -196.6535187, 447.3696594, -642.9797363, 641.8356323
2: -165.2203979, 492.2207031, -166.1165466, 494.6666870, -659.8869019, 658.3372803
3: -174.2521210, 632.6038818, -175.2006683, 635.5876465, -809.8397827, 807.8044434
4: -148.4040833, 577.9964600, -149.2187347, 580.8721313, -729.2762451, 727.2151489

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9719799, upper bound: 554.9736409
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9736408, upper bound: 554.9736409
time: 0.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.18 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -554.9760235, upper bound: 554.9766658
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -554.9751264, upper bound: 554.9766658
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.18
Output dim: 0, lower bound: -554.9562249, upper bound: 554.9627269
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -554.9780442, upper bound: 554.9760236
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -554.9719799, upper bound: 554.9736409
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.18
Output dim: 0, lower bound: -554.9736408, upper bound: 554.9736409

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -123.8089371, 391.1022034, -125.1236649, 397.3967590, -521.2056885, 516.2258911
1: -173.6267548, 394.2460938, -176.0187836, 400.0007324, -573.6275024, 570.2648926
2: -146.6638794, 436.1742249, -148.8076782, 442.0823669, -588.7461548, 584.9818726
3: -154.7870026, 559.9298096, -156.8412018, 568.7158203, -723.5027466, 716.7709961
4: -131.8817444, 512.2440186, -133.8073273, 519.3491821, -651.2309570, 646.0513306

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
time: 1.23 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -129.3839264, 407.2226868, -125.2514267, 397.5273743, -526.9113159, 532.4741211
1: -181.4086456, 410.7022705, -176.1459656, 400.2363892, -581.6448975, 586.8482056
2: -153.2976685, 454.3988037, -148.9157410, 442.3871460, -595.6848145, 603.3145752
3: -161.7082977, 583.0927124, -156.9596252, 569.0974731, -730.8057251, 740.0523682
4: -137.8280945, 533.6986084, -133.9087830, 519.7307739, -657.5587158, 667.6072998

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
time: 1.29 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -130.2214661, 410.5768433, -140.5324249, 446.0257874, -576.2472534, 551.1091309
1: -182.6616821, 413.9107056, -197.6004639, 448.9571228, -631.6187744, 611.5110474
2: -154.3190155, 458.0404663, -166.9410400, 496.4663696, -650.7853394, 624.9815063
3: -162.8068085, 587.6780396, -176.0655060, 637.6236572, -800.4303589, 763.7435303
4: -138.6925507, 537.9248657, -149.9702606, 582.9315796, -721.6240845, 687.8950806

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9760235, upper bound: 554.9766658
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9760235, upper bound: 554.9766658
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -160.9272919, 509.3600769, -131.4853058, 415.6713867, -576.5986938, 640.8453979
1: -226.2525787, 513.2770996, -184.6799316, 419.0795593, -645.3320312, 697.9570312
2: -190.7022705, 568.0831299, -155.9360046, 463.6845703, -654.3867798, 724.0190430
3: -201.9649811, 729.6130981, -164.6857910, 595.0548706, -797.0198364, 894.2988892
4: -171.7263641, 668.8966675, -140.2144928, 544.6845093, -716.4107666, 809.1111450

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9751264, upper bound: 554.9766658
time: 1.24 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9751264, upper bound: 554.9766658
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -135.8643646, 432.2290039, -123.6811142, 393.5485840, -529.4129639, 555.9100342
1: -191.2366180, 434.7472229, -174.0296478, 395.9923401, -587.2289429, 608.7767944
2: -161.5336609, 480.6412659, -147.1132050, 437.7755737, -599.3092041, 627.7544556
3: -170.3521271, 617.7262573, -155.0534821, 563.3153076, -733.6674194, 772.7797241
4: -145.0975952, 564.4382935, -132.2497101, 514.3186035, -659.4161987, 696.6879883

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9780442, upper bound: 554.9760236
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9780442, upper bound: 554.9760236
time: 1.25 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -136.9058838, 435.5904541, -138.8144684, 441.3802185, -578.2861328, 574.4049072
1: -192.6961517, 438.2692871, -195.2920837, 444.1801453, -636.8762207, 633.5614014
2: -162.7675934, 484.5889282, -164.9724731, 491.1402893, -653.9078979, 649.5613403
3: -171.6521912, 622.6895752, -173.9872742, 631.0311890, -802.6833496, 796.6768188
4: -146.2008209, 569.0245972, -148.1903229, 576.7307129, -722.9314575, 717.2149048

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9719799, upper bound: 554.9736409
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9719799, upper bound: 554.9736409
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -164.7813110, 524.8296509, -129.8533936, 411.4375610, -576.2188721, 654.6830444
1: -232.3491669, 528.1937866, -182.5253296, 414.7172241, -647.0663452, 710.7191162
2: -195.8564301, 584.4012451, -154.0866394, 458.8809814, -654.7373047, 738.4878540
3: -207.2706757, 751.0905151, -162.7477570, 589.0689087, -796.3395386, 913.8382568
4: -176.2318573, 687.7275391, -138.5522156, 539.1452026, -715.3770752, 826.2797852

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8966775, upper bound: 554.8866839
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9736250, upper bound: 554.9736250
time: 0.84 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.52 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.52
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.52
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.52
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.52
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.52
Output dim: 0, lower bound: -554.9760235, upper bound: 554.9766658
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.52
Output dim: 0, lower bound: -554.9760235, upper bound: 554.9766658
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.52
Output dim: 0, lower bound: -554.9751264, upper bound: 554.9766658
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.52
Output dim: 0, lower bound: -554.9751264, upper bound: 554.9766658
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.52
Output dim: 0, lower bound: -554.9780442, upper bound: 554.9760236
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.52
Output dim: 0, lower bound: -554.9780442, upper bound: 554.9760236
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.52
Output dim: 0, lower bound: -554.9719799, upper bound: 554.9736409
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.52
Output dim: 0, lower bound: -554.9719799, upper bound: 554.9736409
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.52
Output dim: 0, lower bound: -554.8966775, upper bound: 554.8866839
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.52
Output dim: 0, lower bound: -554.9736250, upper bound: 554.9736250

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -123.8089371, 391.1022034, -113.3766861, 356.6911926, -480.5000610, 504.4788513
1: -173.6267548, 394.2460938, -158.9641418, 360.0633240, -533.6900024, 553.2102051
2: -146.6638794, 436.1742249, -134.4400482, 398.2441101, -544.9079590, 570.6142578
3: -154.7870026, 559.9298096, -141.7183228, 511.6480713, -666.4350586, 701.6481323
4: -131.8817444, 512.2440186, -121.0022049, 468.0400696, -599.9218140, 633.2462158

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
time: 1.54 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -123.8089371, 391.1022034, -121.4794464, 386.7561340, -510.5650330, 512.5816040
1: -173.6267548, 394.2460938, -171.0062408, 389.1526184, -562.7792358, 565.2523193
2: -146.6638794, 436.1742249, -144.5613861, 430.1128845, -576.7766724, 580.7355347
3: -154.7870026, 559.9298096, -152.3398895, 553.5312500, -708.3182373, 712.2697144
4: -131.8817444, 512.2440186, -129.9564667, 505.2542114, -637.1359863, 642.2005005

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
time: 2.99 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -129.3839264, 407.2226868, -113.9852982, 358.6299133, -488.0138550, 521.2079468
1: -181.4086456, 410.7022705, -159.7526703, 362.0180054, -543.4265747, 570.4549561
2: -153.2976685, 454.3988037, -135.1020660, 400.4610901, -553.7587891, 589.5008545
3: -161.7082977, 583.0927124, -142.4242249, 514.5977783, -676.3060913, 725.5169067
4: -137.8280945, 533.6986084, -121.5977554, 470.7221375, -608.5502319, 655.2963867

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -129.3839264, 407.2226868, -121.7310181, 387.2667847, -516.6506348, 528.9537354
1: -181.4086456, 410.7022705, -171.3088989, 389.7390442, -571.1475830, 582.0111694
2: -153.2976685, 454.3988037, -144.8196411, 430.8157349, -584.1134033, 599.2184448
3: -161.7082977, 583.0927124, -152.6085663, 554.4318848, -716.1401978, 735.7012329
4: -137.8280945, 533.6986084, -130.1881714, 506.1542664, -643.9821167, 663.8867798

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -130.2214661, 410.5768433, -130.5581818, 411.5263977, -541.7478638, 541.1349487
1: -182.6616821, 413.9107056, -183.0713501, 414.9476624, -597.6091919, 596.9819336
2: -154.3190155, 458.0404663, -154.6722260, 459.1911926, -613.5101929, 612.7127075
3: -162.8068085, 587.6780396, -163.1810455, 589.1135864, -751.9202881, 750.8590088
4: -138.6925507, 537.9248657, -139.0134277, 539.2700195, -677.9624634, 676.9382935

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711127, upper bound: 554.9690122
time: 1.46 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9760235, upper bound: 554.9766658
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -130.2214661, 410.5768433, -136.6411591, 434.7170715, -564.9385376, 547.2178345
1: -182.6616821, 413.9107056, -192.2805939, 437.4548035, -620.1163940, 606.1912842
2: -154.3190155, 458.0404663, -162.4260406, 483.6763611, -637.9952393, 620.4664917
3: -162.8068085, 587.6780396, -171.2880249, 621.5264893, -784.3331909, 758.9660645
4: -138.6925507, 537.9248657, -145.8975220, 567.9535522, -706.6461182, 683.8223877

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9711127, upper bound: 554.9690122
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9760235, upper bound: 554.9766658
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -160.9272919, 509.3600769, -122.3508987, 384.2171631, -545.1444092, 631.7109985
1: -226.2525787, 513.2770996, -171.3531799, 388.0908813, -614.3434448, 684.6302490
2: -190.7022705, 568.0831299, -144.6886597, 429.6298218, -620.3320923, 712.7717285
3: -201.9649811, 729.6130981, -152.8703308, 551.1052246, -753.0701904, 882.4833984
4: -171.7263641, 668.8966675, -130.1881409, 505.0003967, -676.7267456, 799.0847778

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9666730, upper bound: 554.9684306
time: 1.34 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9751264, upper bound: 554.9766658
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -160.9272919, 509.3600769, -127.6652985, 404.7774353, -565.7047119, 637.0253906
1: -226.2525787, 513.2770996, -179.5050354, 407.9857178, -634.2382812, 692.7821045
2: -190.7022705, 568.0831299, -151.5302887, 451.4077148, -642.1099854, 719.6134033
3: -201.9649811, 729.6130981, -160.0409546, 579.5805664, -781.5455322, 889.6540527
4: -171.7263641, 668.8966675, -136.2546082, 530.3859253, -702.1122437, 805.1511841

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9666730, upper bound: 554.9684306
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9751264, upper bound: 554.9766658
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -135.8643646, 432.2290039, -113.8246460, 358.3888855, -494.2532349, 546.0536499
1: -191.2366180, 434.7472229, -159.5540314, 361.6755371, -552.9121094, 594.3011475
2: -161.5336609, 480.6412659, -134.9241791, 400.0952454, -561.6289062, 615.5654297
3: -170.3521271, 617.7262573, -142.2442017, 514.1212769, -684.4733887, 759.9704590
4: -145.0975952, 564.4382935, -121.4232635, 470.2883606, -615.3859863, 685.8615723

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9273496, upper bound: 554.9328510
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9780442, upper bound: 554.9760236
time: 1.31 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -135.8643646, 432.2290039, -121.6867599, 387.4214172, -523.2857056, 553.9157715
1: -191.2366180, 434.7472229, -171.2631226, 389.7895508, -581.0261230, 606.0101318
2: -161.5336609, 480.6412659, -144.7751312, 430.9050598, -592.4387207, 625.4163818
3: -170.3521271, 617.7262573, -152.5698242, 554.5527954, -724.9049072, 770.2960815
4: -145.0975952, 564.4382935, -130.1375732, 506.2353821, -651.3329468, 694.5758667

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9273496, upper bound: 554.9328510
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9780442, upper bound: 554.9760236
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -136.9058838, 435.5904541, -130.5581818, 411.5263977, -548.4322510, 566.1486206
1: -192.6961517, 438.2692871, -183.0713501, 414.9476624, -607.6437378, 621.3404541
2: -162.7675934, 484.5889282, -154.6722260, 459.1911926, -621.9588013, 639.2611694
3: -171.6521912, 622.6895752, -163.1810455, 589.1135864, -760.7657471, 785.8705444
4: -146.2008209, 569.0245972, -139.0134277, 539.2700195, -685.4706421, 708.0380249

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9047749, upper bound: 554.9126362
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9719799, upper bound: 554.9736409
time: 1.39 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -136.9058838, 435.5904541, -136.6411591, 434.7170715, -571.6229248, 572.2314453
1: -192.6961517, 438.2692871, -192.2805939, 437.4548035, -630.1509399, 630.5498047
2: -162.7675934, 484.5889282, -162.4260406, 483.6763611, -646.4437866, 647.0149536
3: -171.6521912, 622.6895752, -171.2880249, 621.5264893, -793.1786499, 793.9775391
4: -146.2008209, 569.0245972, -145.8975220, 567.9535522, -714.1543579, 714.9221191

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9047749, upper bound: 554.9126362
time: 1.24 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9719799, upper bound: 554.9736409
time: 1.24 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -163.0251465, 519.0306396, -127.0742188, 402.3394470, -565.3645630, 646.1047974
1: -229.8345947, 522.3915405, -178.5610657, 405.6591187, -635.4936523, 700.9526367
2: -193.7358246, 577.9889526, -150.7407684, 448.8837891, -642.6196289, 728.7296143
3: -205.0413513, 742.8426514, -159.2339783, 576.1705322, -781.2119141, 902.0765991
4: -174.3342590, 680.2318726, -135.5692902, 527.4261475, -701.7603760, 815.8011475

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9010663, upper bound: 554.9097326
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9736250, upper bound: 554.9736250
time: 1.10 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.14 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9812708, upper bound: 554.9801511
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9711127, upper bound: 554.9690122
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9760235, upper bound: 554.9766658
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9711127, upper bound: 554.9690122
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9760235, upper bound: 554.9766658
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9666730, upper bound: 554.9684306
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9751264, upper bound: 554.9766658
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9666730, upper bound: 554.9684306
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9751264, upper bound: 554.9766658
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9273496, upper bound: 554.9328510
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9780442, upper bound: 554.9760236
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9273496, upper bound: 554.9328510
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9780442, upper bound: 554.9760236
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9047749, upper bound: 554.9126362
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9719799, upper bound: 554.9736409
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9047749, upper bound: 554.9126362
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9719799, upper bound: 554.9736409
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9010663, upper bound: 554.9097326
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.14
Output dim: 0, lower bound: -554.9736250, upper bound: 554.9736250

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -107.2945023, 338.7147522, -113.3766861, 356.6911926, -463.9856567, 452.0914001
1: -150.3399353, 341.8565674, -158.9641418, 360.0633240, -510.4032288, 500.8206787
2: -127.1170197, 378.1401367, -134.4400482, 398.2441101, -525.3611450, 512.5802002
3: -134.0518646, 486.1622314, -141.7183228, 511.6480713, -645.6998901, 627.8805542
4: -114.4648819, 444.4047546, -121.0022049, 468.0400696, -582.5049438, 565.4069824

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9830296, upper bound: 554.9843761
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9830167, upper bound: 554.9834238
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -123.8264008, 391.1094055, -113.3766861, 356.6911926, -480.5175781, 504.4860535
1: -173.4983215, 394.4565735, -158.9641418, 360.0633240, -533.5615845, 553.4207153
2: -146.6014099, 436.4610291, -134.4400482, 398.2441101, -544.8455200, 570.9009399
3: -154.7384491, 560.1813354, -141.7183228, 511.6480713, -666.3865356, 701.8996582
4: -131.8179932, 512.5811157, -121.0022049, 468.0400696, -599.8580322, 633.5833130

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9830296, upper bound: 554.9843761
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9829951, upper bound: 554.9834238
time: 1.12 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -107.2945023, 338.7147522, -121.4794464, 386.7561340, -494.0506287, 460.1942139
1: -150.3399353, 341.8565674, -171.0062408, 389.1526184, -539.4924316, 512.8627319
2: -127.1170197, 378.1401367, -144.5613861, 430.1128845, -557.2299194, 522.7014160
3: -134.0518646, 486.1622314, -152.3398895, 553.5312500, -687.5831299, 638.5020752
4: -114.4648819, 444.4047546, -129.9564667, 505.2542114, -619.7191162, 574.3612061

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9398676, upper bound: 554.9304234
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -123.8264008, 391.1094055, -121.4794464, 386.7561340, -510.5825195, 512.5888062
1: -173.4983215, 394.4565735, -171.0062408, 389.1526184, -562.6508789, 565.4628296
2: -146.6014099, 436.4610291, -144.5613861, 430.1128845, -576.7142944, 581.0222168
3: -154.7384491, 560.1813354, -152.3398895, 553.5312500, -708.2697144, 712.5211792
4: -131.8179932, 512.5811157, -129.9564667, 505.2542114, -637.0722046, 642.5375977

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9398676, upper bound: 554.9304233
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -113.1604538, 355.7814636, -113.9852982, 358.6299133, -471.7903748, 469.7667542
1: -158.5563812, 359.2316284, -159.7526703, 362.0180054, -520.5744019, 518.9842529
2: -134.0978394, 397.3772888, -135.1020660, 400.4610901, -534.5589600, 532.4793701
3: -141.3645935, 510.6240845, -142.4242249, 514.5977783, -655.9624023, 653.0482788
4: -120.7083435, 467.1063843, -121.5977554, 470.7221375, -591.4304810, 588.7041626

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9830296, upper bound: 554.9839974
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9830045, upper bound: 554.9823967
time: 1.17 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -129.3510437, 407.1157227, -113.9852982, 358.6299133, -487.9809570, 521.1010132
1: -181.2432861, 410.7232971, -159.7526703, 362.0180054, -543.2612305, 570.4759521
2: -153.1453857, 454.5035400, -135.1020660, 400.4610901, -553.6064453, 589.6055908
3: -161.5780792, 583.0626831, -142.4242249, 514.5977783, -676.1757812, 725.4868774
4: -137.6862946, 533.7856445, -121.5977554, 470.7221375, -608.4084473, 655.3833618

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9830296, upper bound: 554.9839974
time: 1.78 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9830045, upper bound: 554.9823967
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -113.1604538, 355.7814636, -121.7310181, 387.2667847, -500.4272461, 477.5124817
1: -158.5563812, 359.2316284, -171.3088989, 389.7390442, -548.2954102, 530.5404663
2: -134.0978394, 397.3772888, -144.8196411, 430.8157349, -564.9135742, 542.1968994
3: -141.3645935, 510.6240845, -152.6085663, 554.4318848, -695.7965088, 663.2326050
4: -120.7083435, 467.1063843, -130.1881714, 506.1542664, -626.8624268, 597.2945557

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9396109, upper bound: 554.9057600
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9764831, upper bound: 554.9755468
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -129.3510437, 407.1157227, -121.7310181, 387.2667847, -516.6177979, 528.8467407
1: -181.2432861, 410.7232971, -171.3088989, 389.7390442, -570.9822388, 582.0322266
2: -153.1453857, 454.5035400, -144.8196411, 430.8157349, -583.9611206, 599.3231812
3: -161.5780792, 583.0626831, -152.6085663, 554.4318848, -716.0099487, 735.6712036
4: -137.6862946, 533.7856445, -130.1881714, 506.1542664, -643.8402710, 663.9738159

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9396109, upper bound: 554.9057600
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9764831, upper bound: 554.9755468
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -122.5189514, 387.1344604, -128.5477600, 404.7292480, -527.2481689, 515.6822510
1: -171.7540436, 390.3070984, -180.2500916, 408.2422791, -579.9962158, 570.5571289
2: -145.0989380, 431.8921204, -152.3078461, 451.7229614, -596.8218384, 584.1998291
3: -153.1572266, 554.2796021, -160.6741791, 579.4128418, -732.5700073, 714.9537964
4: -130.4618378, 507.1836548, -136.9094543, 530.4569092, -660.9187622, 644.0930786

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708400, upper bound: 554.9721169
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -127.9922638, 402.8768616, -129.1515808, 406.6602478, -534.6525269, 532.0284424
1: -179.4128571, 406.3682251, -181.0203400, 410.1860657, -589.5989380, 587.3885498
2: -151.5976715, 449.6938782, -152.9542084, 453.9168701, -605.5144653, 602.6480713
3: -159.9352112, 576.8893433, -161.3681030, 582.2987671, -742.2340088, 738.2574463
4: -136.2904053, 528.1300659, -137.4976349, 533.0811768, -669.3715820, 665.6276245

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9716903, upper bound: 554.9735084
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
time: 1.15 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -122.5189514, 387.1344604, -134.7592773, 428.4426270, -550.9615479, 521.8937378
1: -171.7540436, 390.3070984, -189.6244812, 431.2402954, -602.9942627, 579.9315796
2: -145.0989380, 431.8921204, -160.1987305, 476.7707825, -621.8696899, 592.0908203
3: -153.1572266, 554.2796021, -168.9320984, 612.5681763, -765.7253418, 723.2114868
4: -130.4618378, 507.1836548, -143.9124298, 559.7844849, -690.2463379, 651.0960083

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9640517, upper bound: 554.9623243
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9657771, upper bound: 554.9626069
time: 1.37 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -127.9922638, 402.8768616, -135.1296692, 429.4629211, -557.4552002, 538.0065308
1: -179.4128571, 406.3682251, -190.0903015, 432.2945557, -611.7073364, 596.4584961
2: -151.5976715, 449.6938782, -160.5883636, 477.9683228, -629.5659790, 610.2822266
3: -159.9352112, 576.8893433, -169.3495026, 614.1288452, -774.0640259, 746.2388306
4: -136.2904053, 528.1300659, -144.2715759, 561.2570801, -697.5474854, 672.4016113

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9688858, upper bound: 554.9691362
time: 1.14 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9679441, upper bound: 554.9655416
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -154.9423828, 491.9876709, -120.1860733, 376.8414612, -531.7837524, 612.1737671
1: -217.8140106, 495.5997314, -168.2992096, 380.8317871, -598.6457520, 663.8987427
2: -183.5143738, 548.5043335, -142.1320190, 421.5544434, -605.0688477, 690.6362915
3: -194.4727173, 704.8303223, -150.1540833, 540.6198730, -735.0925293, 854.9843750
4: -165.3093719, 645.8911133, -127.9101562, 495.4717102, -660.7810669, 773.8012695

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9587292, upper bound: 554.9649307
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9571519, upper bound: 554.9598750
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -158.5133820, 500.9492493, -121.0297394, 379.6234436, -538.1367798, 621.9789429
1: -222.7555695, 505.0705872, -169.4295044, 383.5905151, -606.3460693, 674.5001221
2: -187.7836761, 559.0161133, -143.0758667, 424.6470947, -612.4307861, 702.0919800
3: -198.8746338, 717.8222046, -151.1700439, 544.6914673, -743.5659790, 868.9922485
4: -169.1457825, 658.1546021, -128.7679749, 499.1670532, -668.3128662, 786.9224854

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9651421, upper bound: 554.9702092
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9621286, upper bound: 554.9621286
time: 1.18 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -154.9423828, 491.9876709, -125.6846390, 398.1746216, -553.1170044, 617.6723022
1: -217.8140106, 495.5997314, -176.7011261, 401.4560547, -619.2700806, 672.3007202
2: -183.5143738, 548.5043335, -149.1832428, 444.1006165, -627.6149902, 697.6874390
3: -194.4727173, 704.8303223, -157.5509338, 570.1860962, -764.6587524, 862.3811035
4: -165.3093719, 645.8911133, -134.1588898, 521.7680664, -687.0774536, 780.0499878

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9532078, upper bound: 554.9563021
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9537332, upper bound: 554.9556670
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -158.5133820, 500.9492493, -126.2409439, 399.7536926, -558.2670898, 627.1901855
1: -222.7555695, 505.0705872, -177.4444885, 403.0908508, -625.8464355, 682.5150757
2: -187.7836761, 559.0161133, -149.8011475, 446.0278625, -633.8115234, 708.8170776
3: -198.8746338, 717.8222046, -158.2167664, 572.5492554, -771.4238281, 876.0389404
4: -169.1457825, 658.1546021, -134.7262421, 524.0810547, -693.2268066, 792.8808594

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9629810, upper bound: 554.9686039
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9620155, upper bound: 554.9621325
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -133.3499451, 423.4847717, -112.4537735, 353.6067505, -486.9566956, 535.9385376
1: -187.6094360, 426.1604004, -157.5681152, 357.0036621, -544.6130981, 583.7285156
2: -158.4878387, 471.1259766, -133.2574921, 394.9233704, -553.4111938, 604.3834229
3: -167.1376495, 605.4196167, -140.4841614, 507.4504395, -674.5880737, 745.9036865
4: -142.3977203, 553.2965698, -119.9475937, 464.2285767, -606.6262817, 673.2441406

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9763253, upper bound: 554.9765642
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9762179, upper bound: 554.9763036
time: 1.11 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -133.3499451, 423.4847717, -120.1231079, 381.9604187, -515.3103638, 543.6079102
1: -187.6094360, 426.1604004, -169.0050354, 384.4270630, -572.0364990, 595.1654053
2: -158.4878387, 471.1259766, -142.8797150, 424.9842224, -583.4720459, 614.0055542
3: -167.1376495, 605.4196167, -150.5679321, 546.8603516, -713.9979858, 755.9875488
4: -142.3977203, 553.2965698, -128.4556732, 499.2896729, -641.6872559, 681.7522583

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9338909, upper bound: 554.9008933
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9731889, upper bound: 554.9721387
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -134.3871918, 426.9288330, -129.1515808, 406.6602478, -541.0473633, 556.0804443
1: -189.0603333, 429.7427063, -181.0203400, 410.1860657, -599.2463379, 610.7630615
2: -159.7166595, 475.1416931, -152.9542084, 453.9168701, -613.6335449, 628.0958862
3: -168.4313049, 610.4849854, -161.3681030, 582.2987671, -750.7301025, 771.8530884
4: -143.4966583, 557.9626465, -137.4976349, 533.0811768, -676.5778198, 695.4602661

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9677831, upper bound: 554.9702490
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9652223, upper bound: 554.9636534
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -134.3871918, 426.9288330, -135.1296692, 429.4629211, -563.8500977, 562.0584717
1: -189.0603333, 429.7427063, -190.0903015, 432.2945557, -621.3547974, 619.8329468
2: -159.7166595, 475.1416931, -160.5883636, 477.9683228, -637.6849976, 635.7300415
3: -168.4313049, 610.4849854, -169.3495026, 614.1288452, -782.5601196, 779.8344727
4: -143.4966583, 557.9626465, -144.2715759, 561.2570801, -704.7537231, 702.2342529

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9670657, upper bound: 554.9692189
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9652223, upper bound: 554.9636534
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -160.5294037, 510.3281860, -125.5864258, 397.1070557, -557.6363525, 635.9146118
1: -226.2262268, 513.8760376, -176.4090118, 400.5584106, -626.7846680, 690.2850342
2: -190.7153778, 568.5244751, -148.9344177, 443.2764893, -633.9916992, 717.4588623
3: -201.8455200, 730.6190186, -157.3296356, 568.8559570, -770.7014771, 887.9485474
4: -171.6612701, 669.0941772, -133.9732666, 520.8510742, -692.5122070, 803.0674438

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9624180, upper bound: 554.9691702
time: 1.16 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9614611, upper bound: 554.9614611
time: 1.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.33 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9830296, upper bound: 554.9843761
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9830167, upper bound: 554.9834238
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9830296, upper bound: 554.9843761
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9829951, upper bound: 554.9834238
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9398676, upper bound: 554.9304234
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
NS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9398676, upper bound: 554.9304233
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9707074, upper bound: 554.9700515
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9830296, upper bound: 554.9839974
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9830045, upper bound: 554.9823967
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9830296, upper bound: 554.9839974
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9830045, upper bound: 554.9823967
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9396109, upper bound: 554.9057600
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9764831, upper bound: 554.9755468
NS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9396109, upper bound: 554.9057600
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9764831, upper bound: 554.9755468
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9708400, upper bound: 554.9721169
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9716903, upper bound: 554.9735084
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9640517, upper bound: 554.9623243
NS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9657771, upper bound: 554.9626069
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9688858, upper bound: 554.9691362
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9679441, upper bound: 554.9655416
NS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9587292, upper bound: 554.9649307
NS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9571519, upper bound: 554.9598750
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9651421, upper bound: 554.9702092
NS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9621286, upper bound: 554.9621286
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9532078, upper bound: 554.9563021
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9537332, upper bound: 554.9556670
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9629810, upper bound: 554.9686039
NS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9620155, upper bound: 554.9621325
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9763253, upper bound: 554.9765642
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9762179, upper bound: 554.9763036
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9338909, upper bound: 554.9008933
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9731889, upper bound: 554.9721387
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9677831, upper bound: 554.9702490
NS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9652223, upper bound: 554.9636534
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9670657, upper bound: 554.9692189
NS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9652223, upper bound: 554.9636534
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9624180, upper bound: 554.9691702
NS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.33
Output dim: 0, lower bound: -554.9614611, upper bound: 554.9614611

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -104.4086075, 329.5281067, -108.3259354, 340.8139648, -445.2225647, 437.8540039
1: -146.1786041, 332.6824036, -151.6252136, 344.2046204, -490.3832397, 484.3075256
2: -123.5864792, 368.0419312, -128.1639252, 380.8513184, -504.4377747, 496.2058105
3: -130.3619995, 473.2057800, -135.2170715, 489.4111328, -619.7730103, 608.4228516
4: -111.3262939, 432.6192627, -115.4383469, 447.7984009, -559.1246948, 548.0574951

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9824671
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9807122, upper bound: 554.9823714
time: 1.13 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -106.8021164, 337.2336731, -110.7321091, 348.6766357, -455.4787598, 447.9657898
1: -149.6208801, 340.3574219, -155.1325226, 352.0030823, -501.6239624, 495.4899292
2: -126.5142288, 376.4923706, -131.2045288, 389.3873291, -515.9014893, 507.6968994
3: -133.4238586, 484.0639648, -138.3477631, 500.3580933, -633.7819824, 622.4117432
4: -113.9306259, 442.4792786, -118.1326904, 457.6735229, -571.6040649, 560.6119385

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9832861, upper bound: 554.9844968
time: 1.24 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9832861, upper bound: 554.9844968
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -121.0258942, 382.1751099, -108.3259354, 340.8139648, -461.8398438, 490.5010376
1: -169.4587708, 385.5886536, -151.6252136, 344.2046204, -513.6633911, 537.2138062
2: -143.1635742, 426.6737976, -128.1639252, 380.8513184, -524.0146484, 554.8375854
3: -151.1623230, 547.6746216, -135.2170715, 489.4111328, -640.5734863, 682.8917236
4: -128.7742004, 501.1573486, -115.4383469, 447.7984009, -576.5726318, 616.5955200

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803871
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9248401, upper bound: 554.9289479
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -123.3313370, 389.6283264, -110.7321091, 348.6766357, -472.0079651, 500.3604431
1: -172.7788849, 392.9657593, -155.1325226, 352.0030823, -524.7819214, 548.0982666
2: -145.9981995, 434.8196411, -131.2045288, 389.3873291, -535.3854370, 566.0239868
3: -154.1102600, 558.0957642, -138.3477631, 500.3580933, -654.4683838, 696.4434814
4: -131.2829437, 510.6628723, -118.1326904, 457.6735229, -588.9562988, 628.7955322

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9830167, upper bound: 554.9834238
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9830167, upper bound: 554.9834238
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -106.3036041, 335.5457764, -119.6192627, 380.7054138, -487.0089722, 455.1650391
1: -148.9076538, 338.6786804, -168.3222656, 383.0946350, -532.0023193, 507.0008545
2: -125.9161453, 374.6339111, -142.3124695, 423.4490967, -549.3651733, 516.9464111
3: -132.7842255, 481.6592712, -149.9646454, 544.9266968, -677.7109375, 631.6239014
4: -113.3964233, 440.2872009, -127.9536591, 497.4026489, -610.7990723, 568.2407227

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9644562, upper bound: 554.9556680
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9588936, upper bound: 554.9550380
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -122.9333191, 388.2527161, -119.6192627, 380.7054138, -503.6387329, 507.8719482
1: -172.2098236, 391.6052856, -168.3222656, 383.0946350, -555.3044434, 559.9275513
2: -145.5211487, 433.3106995, -142.3124695, 423.4490967, -568.9701538, 575.6231689
3: -153.5996094, 556.1386719, -149.9646454, 544.9266968, -698.5263062, 706.1033325
4: -130.8562927, 508.8780823, -127.9536591, 497.4026489, -628.2589111, 636.8316650

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9568541, upper bound: 554.9491528
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9581437, upper bound: 554.9542852
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -110.4272919, 347.1417847, -109.1276627, 343.4307556, -453.8580322, 456.2694397
1: -154.6246490, 350.5993958, -152.6891937, 346.8094482, -501.4340515, 503.2885437
2: -130.7504883, 387.8750305, -129.0598907, 383.8005981, -514.5510864, 516.9348755
3: -137.8753510, 498.4550781, -136.1706543, 493.2536316, -631.1289673, 634.6257324
4: -117.7354355, 456.0023499, -116.2423248, 451.3079224, -569.0433350, 572.2446899

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9808060
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9807122, upper bound: 554.9814253
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -112.6539154, 354.2465515, -111.3616714, 350.7014160, -463.3553467, 465.6082153
1: -157.8234711, 357.6878662, -155.9575653, 354.0258789, -511.8493347, 513.6454468
2: -133.4787750, 395.6806641, -131.8963776, 391.6761780, -525.1549072, 527.5770264
3: -140.7195587, 508.4624634, -139.0837860, 503.4078064, -644.1273193, 647.5462036
4: -120.1591187, 465.1217346, -118.7537308, 460.4471436, -580.6062012, 583.8753052

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808553, upper bound: 554.9800605
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9807415, upper bound: 554.9807415
time: 1.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -126.6605453, 398.5959473, -109.1276627, 343.4307556, -470.0913086, 507.7236023
1: -177.3393250, 402.2569580, -152.6891937, 346.8094482, -524.1486816, 554.9461670
2: -149.8453522, 445.1570435, -129.0598907, 383.8005981, -533.6459351, 574.2169189
3: -158.1479950, 571.0987549, -136.1706543, 493.2536316, -651.4016113, 707.2694092
4: -134.7570190, 522.8743896, -116.2423248, 451.3079224, -586.0648193, 639.1166992

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803773
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9798328, upper bound: 554.9789992
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -128.8388214, 405.5616150, -111.3616714, 350.7014160, -479.5402222, 516.9232178
1: -180.5035706, 409.1608276, -155.9575653, 354.0258789, -534.5294189, 565.1182861
2: -152.5203094, 452.7844849, -131.8963776, 391.6761780, -544.1964111, 584.6808472
3: -160.9274139, 580.8601685, -139.0837860, 503.4078064, -664.3352051, 719.9439697
4: -137.1315460, 531.7753906, -118.7537308, 460.4471436, -597.5785522, 650.5289917

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9796299
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9796040, upper bound: 554.9783765
time: 1.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -112.6539154, 354.2465515, -119.0779648, 379.1736145, -491.8275146, 473.3245239
1: -157.8234711, 357.6878662, -167.4508057, 381.5766602, -539.4001465, 525.1386719
2: -133.4787750, 395.6806641, -141.5800171, 421.9069214, -555.3856812, 537.2606201
3: -140.7195587, 508.4624634, -149.2287598, 542.9818115, -683.7013550, 657.6911011
4: -120.1591187, 465.1217346, -127.3220367, 495.7239990, -615.8831177, 592.4437866

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9759589, upper bound: 554.9751112
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9759589, upper bound: 554.9763974
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -128.8388214, 405.5616150, -119.0779648, 379.1736145, -508.0124207, 524.6395264
1: -180.5035706, 409.1608276, -167.4508057, 381.5766602, -562.0802002, 576.6115723
2: -152.5203094, 452.7844849, -141.5800171, 421.9069214, -574.4272461, 594.3644409
3: -160.9274139, 580.8601685, -149.2287598, 542.9818115, -703.9092407, 730.0889282
4: -137.1315460, 531.7753906, -127.3220367, 495.7239990, -632.8555298, 659.0973511

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9736956, upper bound: 554.9732078
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9740898, upper bound: 554.9738436
time: 1.26 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -119.7067642, 378.1649780, -123.7954330, 389.8925476, -509.5993042, 501.9604187
1: -167.6988068, 381.4119873, -173.2761383, 393.4765015, -561.1752930, 554.6881104
2: -141.6554260, 422.0812683, -146.3733063, 435.4965515, -577.1519775, 568.4545288
3: -149.5681305, 541.7323608, -154.5551147, 558.6590576, -708.2271118, 696.2874146
4: -127.4089966, 495.7291565, -131.6522369, 511.5636597, -638.9726562, 627.3814087

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
time: 1.38 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
time: 1.27 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -122.0185089, 385.6240845, -125.8709412, 396.6095581, -518.6279907, 511.4950256
1: -171.0256348, 388.7881775, -176.3805389, 400.0802612, -571.1058960, 565.1686401
2: -144.4877930, 430.2211304, -149.0376434, 442.7485657, -587.2363281, 579.2586670
3: -152.5213318, 552.1527710, -157.2720184, 567.9288330, -720.4501953, 709.4246826
4: -129.9188538, 505.2306519, -134.0078583, 519.9619751, -649.8808594, 639.2383423

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
time: 1.38 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -125.3303528, 394.4789124, -124.4810410, 392.0953674, -517.4257202, 518.9598999
1: -175.5488586, 398.0114441, -174.1590271, 395.6826782, -571.2313843, 572.1703491
2: -148.3279266, 440.4768372, -147.1170654, 437.9806519, -586.3084717, 587.5938721
3: -156.5398712, 565.0341187, -155.3543549, 561.9267578, -718.4666138, 720.3883057
4: -133.3886414, 517.3546143, -132.3297272, 514.5254517, -647.9138794, 649.6843262

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
time: 1.40 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -127.4782867, 401.3152161, -126.5045471, 398.6362000, -526.1145020, 527.8197632
1: -178.6703033, 404.7981567, -177.1974182, 402.1199646, -580.7902222, 581.9956055
2: -150.9702911, 447.9669495, -149.7234039, 445.0426941, -596.0128784, 597.6901855
3: -159.2820282, 574.6738892, -158.0061035, 570.9616089, -730.2436523, 732.6799927
4: -135.7336578, 526.1090088, -134.6305389, 522.7007446, -658.4343872, 660.7395020

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
time: 1.52 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
time: 1.13 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -125.3303528, 394.4789124, -130.0903625, 413.6673279, -538.9975586, 524.5692139
1: -175.5488586, 398.0114441, -182.7200775, 416.4730530, -592.0217896, 580.7313232
2: -148.3279266, 440.4768372, -154.3327942, 460.6996155, -609.0274658, 594.8096313
3: -156.5398712, 565.0341187, -162.8623352, 591.7930908, -748.3329468, 727.8964233
4: -133.3886414, 517.3546143, -138.7350464, 540.9747314, -674.3632812, 656.0895996

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9641789, upper bound: 554.9620923
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9639029, upper bound: 554.9618922
time: 1.36 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -127.4782867, 401.3152161, -132.4458008, 421.2570496, -548.7352905, 533.7609863
1: -178.6703033, 404.7981567, -186.2030640, 424.0274353, -602.6976929, 591.0012207
2: -150.9702911, 447.9669495, -157.3111115, 468.9357910, -619.9060059, 605.2780762
3: -159.2820282, 574.6738892, -165.9320068, 602.5164795, -761.7985229, 740.6058960
4: -135.7336578, 526.1090088, -141.3695831, 550.6623535, -686.3959351, 667.4784546

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9638761, upper bound: 554.9613311
time: 1.37 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9635812, upper bound: 554.9611341
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -156.2862091, 494.0547791, -116.8465729, 366.9560852, -523.2423096, 610.9013672
1: -219.5714417, 498.1759949, -163.2964325, 370.8496094, -590.4210205, 661.4724121
2: -185.0682373, 551.3927612, -137.8662415, 410.6406860, -595.7089233, 689.2590332
3: -196.0511475, 708.0711670, -145.8018341, 526.9473267, -722.9984741, 853.8729858
4: -166.7432098, 649.2206421, -124.1511078, 482.9225769, -649.6657715, 773.3716431

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8836878, upper bound: 554.8469141
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8717875, upper bound: 554.8429950
time: 1.21 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -156.2862091, 494.0547791, -121.7388763, 385.6563721, -541.9425659, 615.7936401
1: -219.5714417, 498.1759949, -170.8512268, 389.0423889, -608.6138306, 669.0272217
2: -185.0682373, 551.3927612, -144.2147064, 430.6347351, -615.7029419, 695.6073608
3: -196.0511475, 708.0711670, -152.4146576, 552.8292236, -748.8803711, 860.4857178
4: -166.7432098, 649.2206421, -129.7707367, 506.1480408, -672.8912354, 778.9912720

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8756395, upper bound: 554.8367659
time: 1.02 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8623936, upper bound: 554.8323789
time: 1.28 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -130.6728210, 415.0418396, -107.6234055, 338.5358276, -469.2086182, 522.6651611
1: -183.7640076, 417.6983948, -150.5482330, 341.8979797, -525.6619873, 568.2466431
2: -155.2106018, 461.8368530, -127.2554092, 378.3724365, -533.5830078, 589.0922241
3: -163.7250061, 593.4776611, -134.2705383, 486.2786865, -650.0036621, 727.7481689
4: -139.4848785, 542.3856201, -114.6283417, 444.9475708, -584.4324341, 657.0139160

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9738747, upper bound: 554.9752823
time: 1.18 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9738747, upper bound: 554.9765642
time: 1.89 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -132.8328857, 421.8960876, -109.8226089, 345.6860962, -478.5189819, 531.7186890
1: -186.8561401, 424.5634766, -153.7610474, 348.9942322, -535.8503418, 578.3244629
2: -157.8572388, 469.3826599, -130.0419922, 386.1200562, -543.9772339, 599.4245605
3: -166.4794312, 603.1749268, -137.1337738, 496.2381592, -662.7175903, 740.3086548
4: -141.8393707, 551.2485352, -117.0943680, 453.9305115, -595.7698975, 668.3428955

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9752395, upper bound: 554.9759736
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9752395, upper bound: 554.9763036
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -132.8328857, 421.8960876, -117.4492950, 373.7903442, -506.6232300, 539.3453979
1: -186.8561401, 424.5634766, -165.1167908, 376.2081604, -563.0643311, 589.6801758
2: -157.8572388, 469.3826599, -139.6141357, 415.9974060, -573.8545532, 608.9968262
3: -166.4794312, 603.1749268, -147.1610107, 535.3037720, -701.7832031, 750.3359375
4: -141.8393707, 551.2485352, -125.5658035, 488.8012390, -630.6406250, 676.8143311

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723429, upper bound: 554.9721387
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9723429, upper bound: 554.9721387
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -131.6624756, 418.3391418, -124.4810410, 392.0953674, -523.7578125, 542.8201294
1: -185.1131897, 421.1303101, -174.1590271, 395.6826782, -580.7958374, 595.2892456
2: -156.3782501, 465.6748352, -147.1170654, 437.9806519, -594.3588867, 612.7918701
3: -164.9568481, 598.3212891, -155.3543549, 561.9267578, -726.8834839, 753.6754761
4: -140.5334015, 546.8330688, -132.3297272, 514.5254517, -655.0588379, 679.1627808

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9644820, upper bound: 554.9632292
time: 1.28 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9644820, upper bound: 554.9636534
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -131.6624756, 418.3391418, -130.0903625, 413.6673279, -545.3297729, 548.4295044
1: -185.1131897, 421.1303101, -182.7200775, 416.4730530, -601.5861816, 603.8501587
2: -156.3782501, 465.6748352, -154.3327942, 460.6996155, -617.0778809, 620.0076294
3: -164.9568481, 598.3212891, -162.8623352, 591.7930908, -756.7498169, 761.1835938
4: -140.5334015, 546.8330688, -138.7350464, 540.9747314, -681.5081177, 685.5681152

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9582255, upper bound: 554.9585525
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9602762, upper bound: 554.9597628
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -158.1524506, 502.9331360, -121.0321503, 382.9255371, -541.0780029, 623.9652100
1: -222.8274689, 506.4367981, -169.7449951, 386.4121399, -609.2396240, 676.1817017
2: -187.8215485, 560.3198853, -143.2832336, 427.7496948, -615.5712280, 703.6030884
3: -198.8290405, 720.1148682, -151.4639587, 549.0118408, -747.8408813, 871.5786743
4: -169.0952148, 659.4795532, -128.9590607, 502.7503662, -671.8455811, 788.4385986

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8691254, upper bound: 554.8351511
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8605655, upper bound: 554.8321579
time: 1.24 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.16 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9824671
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9807122, upper bound: 554.9823714
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9832861, upper bound: 554.9844968
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9832861, upper bound: 554.9844968
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803871
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9248401, upper bound: 554.9289479
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9830167, upper bound: 554.9834238
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9830167, upper bound: 554.9834238
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9644562, upper bound: 554.9556680
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9588936, upper bound: 554.9550380
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9568541, upper bound: 554.9491528
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9581437, upper bound: 554.9542852
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9808060
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9807122, upper bound: 554.9814253
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9808553, upper bound: 554.9800605
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9807415, upper bound: 554.9807415
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803773
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9798328, upper bound: 554.9789992
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9796299
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9796040, upper bound: 554.9783765
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9759589, upper bound: 554.9751112
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9759589, upper bound: 554.9763974
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9736956, upper bound: 554.9732078
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9740898, upper bound: 554.9738436
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9641789, upper bound: 554.9620923
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9639029, upper bound: 554.9618922
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9638761, upper bound: 554.9613311
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9635812, upper bound: 554.9611341
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.8836878, upper bound: 554.8469141
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.8717875, upper bound: 554.8429950
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.8756395, upper bound: 554.8367659
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.8623936, upper bound: 554.8323789
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9738747, upper bound: 554.9752823
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9738747, upper bound: 554.9765642
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9752395, upper bound: 554.9759736
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9752395, upper bound: 554.9763036
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9723429, upper bound: 554.9721387
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9723429, upper bound: 554.9721387
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9644820, upper bound: 554.9632292
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9644820, upper bound: 554.9636534
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9582255, upper bound: 554.9585525
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.9602762, upper bound: 554.9597628
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.8691254, upper bound: 554.8351511
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.16
Output dim: 0, lower bound: -554.8605655, upper bound: 554.8321579

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -98.3299866, 309.2431335, -103.7427368, 325.8336182, -424.1636047, 412.9858704
1: -137.3795319, 312.6152344, -145.0235291, 329.3263245, -466.7058716, 457.6387634
2: -116.1775284, 345.9602966, -122.6124725, 364.4682007, -480.6456909, 468.5727539
3: -122.6197586, 444.8223267, -129.4006500, 468.4344788, -591.0542603, 574.2229614
4: -104.7861938, 406.9027405, -110.5200729, 428.7301331, -533.5163574, 517.4227905

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9824671
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9824671
time: 1.37 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -101.1333008, 318.6616211, -106.2378998, 333.9205933, -435.0538940, 424.8994751
1: -141.4566040, 321.8780212, -148.6168060, 337.3444519, -478.8009644, 470.4947815
2: -119.5876923, 356.1258240, -125.6034164, 373.2866211, -492.8743286, 481.7292480
3: -126.1745453, 457.8981934, -132.5520935, 479.6986694, -605.8732300, 590.4501953
4: -107.7777023, 418.7268372, -113.1747284, 438.9867554, -546.7644653, 531.9015503

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9793432, upper bound: 554.9814627
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9799121, upper bound: 554.9815473
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -101.7687454, 321.0721741, -110.7321091, 348.6766357, -450.4453735, 431.8042908
1: -142.2561188, 324.2406616, -155.1325226, 352.0030823, -494.2592163, 479.3731079
2: -120.2643738, 358.8330383, -131.2045288, 389.3873291, -509.6516418, 490.0375671
3: -126.9005737, 461.4047852, -138.3477631, 500.3580933, -627.2586670, 599.7525635
4: -108.3677826, 421.9619446, -118.1326904, 457.6735229, -566.0411987, 540.0946045

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9800605, upper bound: 554.9818870
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9807006, upper bound: 554.9819415
time: 1.48 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -104.7906265, 331.2607422, -110.7321091, 348.6766357, -453.4672546, 441.9928589
1: -146.6872864, 334.3048706, -155.1325226, 352.0030823, -498.6903687, 489.4373474
2: -124.0560684, 369.8325500, -131.2045288, 389.3873291, -513.4433594, 501.0370789
3: -130.8629608, 475.6061707, -138.3477631, 500.3580933, -631.2210693, 613.9539185
4: -111.7536011, 434.6999207, -118.1326904, 457.6735229, -569.4270630, 552.8325806

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9800605, upper bound: 554.9818870
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9807006, upper bound: 554.9819400
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -115.4432297, 363.5004578, -103.7427368, 325.8336182, -441.2768250, 467.2431946
1: -161.3256989, 367.1196289, -145.0235291, 329.3263245, -490.6520081, 512.1431885
2: -136.3480377, 406.3174133, -122.6124725, 364.4682007, -500.8162231, 528.9298706
3: -144.0263977, 521.6188354, -129.4006500, 468.4344788, -612.4606934, 651.0194702
4: -122.7440643, 477.4877625, -110.5200729, 428.7301331, -551.4741821, 588.0078125

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803871
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803871
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -118.7197800, 374.9416809, -110.7321091, 348.6766357, -467.3964233, 485.6737976
1: -166.0222015, 378.3865051, -155.1325226, 352.0030823, -518.0252686, 533.5189819
2: -140.2467957, 418.8092346, -131.2045288, 389.3873291, -529.6340332, 550.0136719
3: -148.1370850, 537.6093140, -138.3477631, 500.3580933, -648.4951782, 675.9570923
4: -126.1754761, 492.0418396, -118.1326904, 457.6735229, -583.8488770, 610.1745605

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8348724, upper bound: 554.8764682
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8357560, upper bound: 554.8760825
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -121.2934799, 383.5518494, -110.7321091, 348.6766357, -469.9701233, 494.2839661
1: -169.8303375, 386.8514404, -155.1325226, 352.0030823, -521.8331909, 541.9839478
2: -143.5142212, 428.0850525, -131.2045288, 389.3873291, -532.9015503, 559.2894287
3: -151.5243835, 549.5518188, -138.3477631, 500.3580933, -651.8824463, 687.8995972
4: -129.0835724, 502.7966309, -118.1326904, 457.6735229, -586.7568970, 620.9293213

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.8348724, upper bound: 554.9779322
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.8357560, upper bound: 554.9771348
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -104.7348099, 328.1687317, -104.4780502, 328.2094421, -432.9442444, 432.6467896
1: -146.4063873, 331.8330078, -145.9867401, 331.6909790, -478.0973511, 477.8197632
2: -123.8388138, 367.2252502, -123.4240417, 367.1634521, -491.0021667, 490.6492920
3: -130.6424255, 471.8414612, -130.2625122, 471.9193726, -602.5617676, 602.1040039
4: -111.6278076, 431.9108276, -111.2500687, 431.9387207, -543.5665283, 543.1608887

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9808060
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9808060
time: 1.22 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -107.0088196, 335.8064575, -106.9919510, 336.3670349, -443.3758545, 442.7984009
1: -149.6930084, 339.3215637, -149.6130371, 339.7845764, -489.4775391, 488.9345093
2: -126.5539703, 375.4403381, -126.4410858, 376.0506287, -502.6045837, 501.8814087
3: -133.5061035, 482.4893188, -133.4465942, 483.3049622, -616.8110352, 615.9358521
4: -114.0258713, 441.5177917, -113.9284058, 442.2828369, -556.3087158, 555.4461060

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9793432, upper bound: 554.9807900
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9799121, upper bound: 554.9809697
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -106.9028320, 334.9200439, -106.6988754, 335.2537842, -442.1566162, 441.6189270
1: -149.5377655, 338.6195068, -149.2487335, 338.7245483, -488.2622986, 487.8681946
2: -126.4873657, 374.6968689, -126.2458954, 374.8329773, -501.3203125, 500.9427185
3: -133.4022217, 481.4251099, -133.1520691, 481.7516174, -615.1537476, 614.5771484
4: -113.9835739, 440.6597900, -113.7519836, 440.8028870, -554.7864380, 554.4117432

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9809205, upper bound: 554.9800605
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9809205, upper bound: 554.9800605
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -109.2695389, 343.0928955, -109.2309113, 343.6681519, -452.9376831, 452.3237915
1: -152.9467468, 346.5700378, -152.8745575, 347.0171204, -499.9638672, 499.4445801
2: -129.3332520, 383.4140015, -129.2837067, 383.9430542, -513.2763062, 512.6976929
3: -136.4020996, 492.7350769, -136.3652344, 493.5013733, -629.9033813, 629.1003418
4: -116.4928894, 450.8341064, -116.4444580, 451.4496765, -567.9425659, 567.2785645

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9759870, upper bound: 554.9730483
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9807415, upper bound: 554.9807415
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -121.2144470, 380.3314514, -104.4780502, 328.2094421, -449.4238892, 484.8094788
1: -169.4951477, 384.2859192, -145.9867401, 331.6909790, -501.1861267, 530.2726440
2: -143.2245331, 425.3398438, -123.4240417, 367.1634521, -510.3879089, 548.7637939
3: -151.2274933, 545.6925049, -130.2625122, 471.9193726, -623.1468506, 675.9550171
4: -128.9034424, 499.8409424, -111.2500687, 431.9387207, -560.8421631, 611.0910034

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803773
time: 1.43 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803773
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -123.2593460, 387.3041382, -106.9919510, 336.3670349, -459.6263428, 494.2960815
1: -172.4235077, 391.1097412, -149.6130371, 339.7845764, -512.2080688, 540.7227783
2: -145.6826172, 432.8518982, -126.4410858, 376.0506287, -521.7332153, 559.2929688
3: -153.8143768, 555.2932129, -133.4465942, 483.3049622, -637.1193237, 688.7397461
4: -131.0803680, 508.5370178, -113.9284058, 442.2828369, -573.3632202, 622.4653320

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9785486, upper bound: 554.9773635
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9789941, upper bound: 554.9776111
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -123.2693558, 386.7746582, -106.6988754, 335.2537842, -458.5231323, 493.4735413
1: -172.4797363, 390.7420959, -149.2487335, 338.7245483, -511.2042847, 539.9906006
2: -145.7486115, 432.4692383, -126.2458954, 374.8329773, -520.5816040, 558.7150269
3: -153.8495636, 554.8112793, -133.1520691, 481.7516174, -635.6010132, 687.9633789
4: -131.1451416, 508.1822205, -113.7519836, 440.8028870, -571.9479370, 621.9340820

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9544437, upper bound: 554.9619391
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9538158, upper bound: 554.9544434
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -125.5078049, 394.5447998, -109.2309113, 343.6681519, -469.1759338, 503.7756958
1: -175.6966248, 398.2737732, -152.8745575, 347.0171204, -522.7137451, 551.1483154
2: -148.4478760, 440.7606812, -129.2837067, 383.9430542, -532.3908691, 570.0442505
3: -156.6925201, 565.4210815, -136.3652344, 493.5013733, -650.1938477, 701.7863159
4: -133.5351257, 517.7722778, -116.4444580, 451.4496765, -584.9848022, 634.2167358

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9524506, upper bound: 554.9526861
time: 1.44 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9796040, upper bound: 554.9783765
time: 1.32 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -108.3533554, 340.7630615, -119.0779648, 379.1736145, -487.5269470, 459.8410339
1: -151.5644073, 344.1869507, -167.4508057, 381.5766602, -533.1410522, 511.6377563
2: -128.1149445, 380.9014282, -141.5800171, 421.9069214, -550.0218506, 522.4813843
3: -135.1742859, 489.5056152, -149.2287598, 542.9818115, -678.1561279, 638.7343750
4: -115.4054642, 447.8998718, -127.3220367, 495.7239990, -611.1294556, 575.2218628

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.8655782, upper bound: 554.9696532
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9399418, upper bound: 554.9749829
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -110.5352020, 347.8608093, -119.0779648, 379.1736145, -489.7088013, 466.9387817
1: -154.7587738, 351.2330017, -167.4508057, 381.5766602, -536.3353882, 518.6837158
2: -130.8897247, 388.5863037, -141.5800171, 421.9069214, -552.7966309, 530.1661987
3: -138.0217896, 499.4254150, -149.2287598, 542.9818115, -681.0036011, 648.6541138
4: -117.8621674, 456.8241272, -127.3220367, 495.7239990, -613.5861816, 584.1461182

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.8655782, upper bound: 554.9750505
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9399418, upper bound: 554.9759225
time: 1.38 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -123.2693558, 386.7746582, -114.3646393, 363.5522156, -486.8215332, 501.1392822
1: -172.4797363, 390.7420959, -160.6582031, 366.1672974, -538.6470337, 551.4002686
2: -145.7486115, 432.4692383, -135.8584747, 404.9287109, -550.6773071, 568.3276978
3: -153.8495636, 554.8112793, -143.2285919, 521.0695190, -674.9190674, 698.0398560
4: -131.1451416, 508.1822205, -122.2575912, 475.9293823, -607.0745239, 630.4396362

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9483036, upper bound: 554.9497668
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9457593, upper bound: 554.9467464
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -125.5078049, 394.5447998, -116.7544632, 371.3944397, -496.9022217, 511.2992554
1: -175.6966248, 398.2737732, -164.0906525, 373.8855591, -549.5820312, 562.3643799
2: -148.4478760, 440.7606812, -138.7384338, 413.4578247, -561.9056396, 579.4989014
3: -156.6925201, 565.4210815, -146.2642670, 532.0112915, -688.7037354, 711.6853638
4: -133.5351257, 517.7722778, -124.8051682, 485.9022217, -619.4373779, 642.5774536

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9533013, upper bound: 554.9545258
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9526829, upper bound: 554.9492446
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -117.5084839, 371.1361694, -123.7954330, 389.8925476, -507.4010010, 494.9316101
1: -164.4124603, 374.4757080, -173.2761383, 393.4765015, -557.8889160, 547.7517090
2: -138.8667755, 414.5050659, -146.3733063, 435.4965515, -574.3633423, 560.8782959
3: -146.6768494, 531.9971313, -154.5551147, 558.6590576, -705.3359375, 686.5522461
4: -124.9249344, 486.9445190, -131.6522369, 511.5636597, -636.4885864, 618.5966797

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708400, upper bound: 554.9721169
time: 1.03 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708400, upper bound: 554.9721169
time: 1.29 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -119.9901428, 379.5783997, -123.7954330, 389.8925476, -509.8826904, 503.3738403
1: -168.0755157, 382.7011108, -173.2761383, 393.4765015, -561.5520020, 555.9772339
2: -142.0143890, 423.5182190, -146.3733063, 435.4965515, -577.5109253, 569.8915405
3: -149.9470978, 543.6427002, -154.5551147, 558.6590576, -708.6061401, 698.1977539
4: -127.7269058, 497.3981323, -131.6522369, 511.5636597, -639.2905884, 629.0503540

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708400, upper bound: 554.9721169
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9708400, upper bound: 554.9721169
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -117.5084839, 371.1361694, -125.8709412, 396.6095581, -514.1179199, 497.0071106
1: -164.4124603, 374.4757080, -176.3805389, 400.0802612, -564.4927368, 550.8561401
2: -138.8667755, 414.5050659, -149.0376434, 442.7485657, -581.6153564, 563.5425415
3: -146.6768494, 531.9971313, -157.2720184, 567.9288330, -714.6057129, 689.2691650
4: -124.9249344, 486.9445190, -134.0078583, 519.9619751, -644.8868408, 620.9522705

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
time: 1.32 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
time: 1.23 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -119.9901428, 379.5783997, -125.8709412, 396.6095581, -516.5997314, 505.4493408
1: -168.0755157, 382.7011108, -176.3805389, 400.0802612, -568.1557617, 559.0816650
2: -142.0143890, 423.5182190, -149.0376434, 442.7485657, -584.7629395, 572.5558472
3: -149.9470978, 543.6427002, -157.2720184, 567.9288330, -717.8759155, 700.9146729
4: -127.7269058, 497.3981323, -134.0078583, 519.9619751, -647.6889038, 631.4060059

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
time: 1.23 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -123.3336258, 388.3833923, -124.4810410, 392.0953674, -515.4290161, 512.8644409
1: -172.5829468, 391.9216309, -174.1590271, 395.6826782, -568.2655029, 566.0806274
2: -145.7717285, 433.8309631, -147.1170654, 437.9806519, -583.7521973, 580.9479370
3: -153.9373627, 556.5512695, -155.3543549, 561.9267578, -715.8641357, 711.9054565
4: -131.1349640, 509.6212769, -132.3297272, 514.5254517, -645.6602783, 641.9509888

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9716903, upper bound: 554.9735084
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9716903, upper bound: 554.9735084
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -125.3263550, 394.7814026, -124.4810410, 392.0953674, -517.4216919, 519.2624512
1: -175.5616302, 398.2318420, -174.1590271, 395.6826782, -571.2440186, 572.3908691
2: -148.3434296, 440.7427979, -147.1170654, 437.9806519, -586.3238525, 587.8598022
3: -156.5477295, 565.4174194, -155.3543549, 561.9267578, -718.4744873, 720.7716064
4: -133.4026337, 517.6563110, -132.3297272, 514.5254517, -647.9280396, 649.9860229

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9716903, upper bound: 554.9735084
time: 1.47 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9716903, upper bound: 554.9735084
time: 1.29 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -123.3336258, 388.3833923, -126.5045471, 398.6362000, -521.9698486, 514.8878784
1: -172.5829468, 391.9216309, -177.1974182, 402.1199646, -574.7028809, 569.1189575
2: -145.7717285, 433.8309631, -149.7234039, 445.0426941, -590.8143311, 583.5541992
3: -153.9373627, 556.5512695, -158.0061035, 570.9616089, -724.8989258, 714.5573120
4: -131.1349640, 509.6212769, -134.6305389, 522.7007446, -653.8356934, 644.2517090

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -125.3263550, 394.7814026, -126.5045471, 398.6362000, -523.9625244, 521.2859497
1: -175.5616302, 398.2318420, -177.1974182, 402.1199646, -577.6814575, 575.4292603
2: -148.3434296, 440.7427979, -149.7234039, 445.0426941, -593.3860474, 590.4660034
3: -156.5477295, 565.4174194, -158.0061035, 570.9616089, -727.5092773, 723.4234619
4: -133.4026337, 517.6563110, -134.6305389, 522.7007446, -656.1033936, 652.2867432

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
time: 1.14 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
time: 1.10 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -115.3809128, 366.6003113, -107.6234055, 338.5358276, -453.9167175, 474.2237244
1: -162.1850586, 369.1214294, -150.5482330, 341.8979797, -504.0830078, 519.6696167
2: -137.1014404, 408.1495361, -127.2554092, 378.3724365, -515.4738159, 535.4049072
3: -144.5214386, 525.0899048, -134.2705383, 486.2786865, -630.8001099, 659.3604126
4: -123.3201828, 479.5672607, -114.6283417, 444.9475708, -568.2677612, 594.1956177

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9720055, upper bound: 554.9721425
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9738371, upper bound: 554.9752823
time: 1.16 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -129.6118011, 411.4613037, -107.6234055, 338.5358276, -468.1476135, 519.0845947
1: -182.1010437, 414.3415527, -150.5482330, 341.8979797, -523.9989624, 564.8897705
2: -153.8475800, 458.2642822, -127.2554092, 378.3724365, -532.2200317, 585.5195923
3: -162.2984924, 588.6221924, -134.2705383, 486.2786865, -648.5771484, 722.8927002
4: -138.2847137, 538.0656738, -114.6283417, 444.9475708, -583.2322998, 652.6940308

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.8808187, upper bound: 554.8414235
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.8808187, upper bound: 554.9764607
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -117.6301117, 373.7481995, -109.8226089, 345.6860962, -463.3162231, 483.5708008
1: -165.4068756, 376.2588196, -153.7610474, 348.9942322, -514.4011230, 530.0198364
2: -139.8576813, 416.0183105, -130.0419922, 386.1200562, -525.9776611, 546.0602417
3: -147.3913879, 535.2241211, -137.1337738, 496.2381592, -643.6295166, 672.3579102
4: -125.7698212, 488.7886658, -117.0943680, 453.9305115, -579.7003174, 605.8829346

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9478912, upper bound: 554.9511858
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9752395, upper bound: 554.9759736
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -131.7418823, 418.1990051, -109.8226089, 345.6860962, -477.4279480, 528.0216064
1: -185.1879120, 421.0933838, -153.7610474, 348.9942322, -534.1821289, 574.8543701
2: -156.4616241, 465.6882935, -130.0419922, 386.1200562, -542.5816040, 595.7302246
3: -165.0166321, 598.1781616, -137.1337738, 496.2381592, -661.2547607, 735.3118896
4: -140.6064301, 546.7978516, -117.0943680, 453.9305115, -594.5368652, 663.8921509

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9478912, upper bound: 554.9511858
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9752395, upper bound: 554.9763036
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -117.6301117, 373.7481995, -117.4492950, 373.7903442, -491.4204712, 491.1974792
1: -165.4068756, 376.2588196, -165.1167908, 376.2081604, -541.6150513, 541.3756104
2: -139.8576813, 416.0183105, -139.6141357, 415.9974060, -555.8551025, 555.6324463
3: -147.3913879, 535.2241211, -147.1610107, 535.3037720, -682.6951904, 682.3851318
4: -125.7698212, 488.7886658, -125.5658035, 488.8012390, -614.5710449, 614.3544312

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9303393, upper bound: 554.9407892
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9251964, upper bound: 554.9251231
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -131.7418823, 418.1990051, -117.4492950, 373.7903442, -505.5322266, 535.6482544
1: -185.1879120, 421.0933838, -165.1167908, 376.2081604, -561.3960571, 586.2100830
2: -156.4616241, 465.6882935, -139.6141357, 415.9974060, -572.4589844, 605.3024292
3: -165.0166321, 598.1781616, -147.1610107, 535.3037720, -700.3204346, 745.3391724
4: -140.6064301, 546.7978516, -125.5658035, 488.8012390, -629.4076538, 672.3635864

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9303393, upper bound: 554.9546545
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -554.9251964, upper bound: 554.9251231
time: 1.17 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.77 seconds
NS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9824671
NS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9824671
NS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9793432, upper bound: 554.9814627
NS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9799121, upper bound: 554.9815473
NS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9800605, upper bound: 554.9818870
NS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9807006, upper bound: 554.9819415
NS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9800605, upper bound: 554.9818870
NS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9807006, upper bound: 554.9819400
NS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803871
NS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803871
NS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.8348724, upper bound: 554.8764682
NS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.8357560, upper bound: 554.8760825
NS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.8348724, upper bound: 554.9779322
NS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.8357560, upper bound: 554.9771348
NS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9808060
NS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9808060
NS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9793432, upper bound: 554.9807900
NS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9799121, upper bound: 554.9809697
NS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9809205, upper bound: 554.9800605
NS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9809205, upper bound: 554.9800605
NS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9759870, upper bound: 554.9730483
NS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9807415, upper bound: 554.9807415
NS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803773
NS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803773
NS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9785486, upper bound: 554.9773635
NS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9789941, upper bound: 554.9776111
NS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9544437, upper bound: 554.9619391
NS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9538158, upper bound: 554.9544434
NS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9524506, upper bound: 554.9526861
NS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9796040, upper bound: 554.9783765
NS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.8655782, upper bound: 554.9696532
NS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9399418, upper bound: 554.9749829
NS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.8655782, upper bound: 554.9750505
NS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9399418, upper bound: 554.9759225
NS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9483036, upper bound: 554.9497668
NS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9457593, upper bound: 554.9467464
NS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9533013, upper bound: 554.9545258
NS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9526829, upper bound: 554.9492446
NS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9708400, upper bound: 554.9721169
NS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9708400, upper bound: 554.9721169
NS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9708400, upper bound: 554.9721169
NS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9708400, upper bound: 554.9721169
NS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
NS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
NS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
NS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9683026, upper bound: 554.9653979
NS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9716903, upper bound: 554.9735084
NS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9716903, upper bound: 554.9735084
NS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9716903, upper bound: 554.9735084
NS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9716903, upper bound: 554.9735084
NS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
NS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
NS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
NS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9683344, upper bound: 554.9655711
NS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9720055, upper bound: 554.9721425
NS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9738371, upper bound: 554.9752823
NS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.8808187, upper bound: 554.8414235
NS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.8808187, upper bound: 554.9764607
NS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9478912, upper bound: 554.9511858
NS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9752395, upper bound: 554.9759736
NS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9478912, upper bound: 554.9511858
NS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9752395, upper bound: 554.9763036
NS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9303393, upper bound: 554.9407892
NS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9251964, upper bound: 554.9251231
NS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9303393, upper bound: 554.9546545
NS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.77
Output dim: 0, lower bound: -554.9251964, upper bound: 554.9251231

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -98.3299866, 309.2431335, -97.1389542, 305.8904724, -404.2204590, 406.3820496
1: -137.3795319, 312.6152344, -135.5834503, 309.1815796, -446.5610962, 448.1986694
2: -116.1775284, 345.9602966, -114.6330261, 342.2584534, -458.4359436, 460.5932922
3: -122.6197586, 444.8223267, -121.0200577, 440.1478271, -562.7675781, 565.8424072
4: -104.7861938, 406.9027405, -103.3947906, 402.6805420, -507.4667358, 510.2975464

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9824671
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9824671
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -98.3299866, 309.2431335, -103.6970367, 325.5162354, -423.8462219, 412.9401855
1: -137.3795319, 312.6152344, -144.8534546, 329.0447693, -466.4243164, 457.4686890
2: -116.1775284, 345.9602966, -122.4707870, 364.2399902, -480.4174805, 468.4310913
3: -122.6197586, 444.8223267, -129.2581787, 468.1342163, -590.7539673, 574.0805054
4: -104.7861938, 406.9027405, -110.4063797, 428.5015564, -533.2876587, 517.3091431

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9824671
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9824671
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -98.2495346, 309.2796936, -100.8777390, 316.4779053, -414.7274475, 410.1574402
1: -137.3159180, 312.4825134, -140.9087219, 319.8545532, -457.1704712, 453.3912048
2: -116.0999146, 345.7975159, -119.1189270, 354.0654297, -470.1653442, 464.9164429
3: -122.4923325, 444.5791931, -125.6994781, 454.9519958, -577.4442749, 570.2786865
4: -104.6707230, 406.5886230, -107.3937607, 416.4188843, -521.0895996, 513.9823608

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9793432, upper bound: 554.9814627
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9793432, upper bound: 554.9814627
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -99.8711090, 314.5484619, -107.3419647, 339.0615540, -438.9326477, 421.8904114
1: -139.6881256, 317.7532043, -150.3267365, 342.5500183, -482.2381592, 468.0799561
2: -118.0875244, 351.5680847, -127.1960983, 378.9282227, -497.0157471, 478.7641602
3: -124.5943527, 452.0466614, -134.1276245, 488.0730591, -612.6674194, 586.1743164
4: -106.4330902, 413.3847351, -114.7084808, 446.0370483, -552.4701538, 528.0932007

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9799121, upper bound: 554.9815473
time: 1.31 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9793432, upper bound: 554.9815473
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -97.1389542, 305.8904724, -104.8152390, 328.7303162, -425.8692322, 410.7056885
1: -135.5834503, 309.1815796, -146.5957794, 332.3063965, -467.8898315, 455.7773438
2: -114.6330261, 342.2584534, -124.0083542, 367.7120361, -482.3450317, 466.2668152
3: -121.0200577, 440.1478271, -130.8078308, 472.4818420, -593.5018921, 570.9556274
4: -103.3947906, 402.6805420, -111.7735977, 432.4317627, -535.8265381, 514.4540405

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9797169, upper bound: 554.9821542
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808060, upper bound: 554.9822429
time: 1.15 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -99.7234726, 314.3067932, -107.3358459, 337.4536743, -437.1771240, 421.6426392
1: -139.3111877, 317.5144043, -150.2258148, 340.8237610, -480.1349182, 467.7401733
2: -117.7668228, 351.4234924, -127.0471344, 377.0484619, -494.8152771, 478.4706116
3: -124.2901764, 451.8710632, -134.0171204, 484.5490723, -608.8392334, 585.8879395
4: -106.1513596, 413.3246460, -114.4563065, 443.3150635, -549.4664307, 527.7809448

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9804045, upper bound: 554.9821622
time: 1.22 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9814253, upper bound: 554.9821646
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -99.9723892, 315.3446350, -104.8152390, 328.7303162, -428.7026672, 420.1598206
1: -139.7364807, 318.5329895, -146.5957794, 332.3063965, -472.0428772, 465.1287231
2: -118.2134552, 352.4734192, -124.0083542, 367.7120361, -485.9254456, 476.4817810
3: -124.7394333, 453.3199463, -130.8078308, 472.4818420, -597.2212524, 584.1277466
4: -106.5846558, 414.4823608, -111.7735977, 432.4317627, -539.0164185, 526.2559204

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9792325, upper bound: 554.9815764
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9800605, upper bound: 554.9818870
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -102.8013916, 324.7541504, -107.3358459, 337.4536743, -440.2550659, 432.0899963
1: -143.8240356, 327.8259277, -150.2258148, 340.8237610, -484.6477661, 478.0516968
2: -121.6228790, 362.6682129, -127.0471344, 377.0484619, -498.6713257, 489.7153015
3: -128.3245239, 466.4479980, -134.0171204, 484.5490723, -612.8735962, 600.4650879
4: -109.6015320, 426.3526917, -114.4563065, 443.3150635, -552.9166260, 540.8089600

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9799630, upper bound: 554.9819400
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9807006, upper bound: 554.9819172
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -115.4432297, 363.5004578, -97.1389542, 305.8904724, -421.3336792, 460.6393738
1: -161.3256989, 367.1196289, -135.5834503, 309.1815796, -470.5072327, 502.7030640
2: -136.3480377, 406.3174133, -114.6330261, 342.2584534, -478.6064758, 520.9504395
3: -144.0263977, 521.6188354, -121.0200577, 440.1478271, -584.1740723, 642.6389160
4: -122.7440643, 477.4877625, -103.3947906, 402.6805420, -525.4245605, 580.8825684

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803871
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803871
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -115.4432297, 363.5004578, -103.6970367, 325.5162354, -440.9594116, 467.1974792
1: -161.3256989, 367.1196289, -144.8534546, 329.0447693, -490.3704224, 511.9730835
2: -136.3480377, 406.3174133, -122.4707870, 364.2399902, -500.5880127, 528.7882080
3: -144.0263977, 521.6188354, -129.2581787, 468.1342163, -612.1605225, 650.8770142
4: -122.7440643, 477.4877625, -110.4063797, 428.5015564, -551.2456055, 587.8940430

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803871
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808511, upper bound: 554.9803871
time: 1.24 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -116.4599228, 367.5085449, -104.8152390, 328.7303162, -445.1902466, 472.3237610
1: -162.8598938, 370.9739990, -146.5957794, 332.3063965, -495.1662903, 517.5697632
2: -137.6465912, 410.5739746, -124.0083542, 367.7120361, -505.3586426, 534.5823364
3: -145.3731232, 527.1368408, -130.8078308, 472.4818420, -617.8549805, 657.9447021
4: -123.8880157, 482.4217529, -111.7735977, 432.4317627, -556.3197632, 594.1951294

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9748910, upper bound: 554.9771286
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9613468, upper bound: 554.9735697
time: 1.22 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -119.2588348, 376.9016724, -107.3358459, 337.4536743, -456.7125244, 484.2374878
1: -166.9020081, 380.2698669, -150.2258148, 340.8237610, -507.7257690, 530.4956665
2: -141.0318756, 420.8034973, -127.0471344, 377.0484619, -518.0803223, 547.8505859
3: -148.9388275, 540.2503052, -134.0171204, 484.5490723, -633.4879150, 674.2674561
4: -126.8879929, 494.3246155, -114.4563065, 443.3150635, -570.2030640, 608.7808228

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9646022, upper bound: 554.9686913
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9759570, upper bound: 554.9771348
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -104.7348099, 328.1687317, -97.1389542, 305.8904724, -410.6252747, 425.3076477
1: -146.4063873, 331.8330078, -135.5834503, 309.1815796, -455.5879517, 467.4164429
2: -123.8388138, 367.2252502, -114.6330261, 342.2584534, -466.0971985, 481.8582458
3: -130.6424255, 471.8414612, -121.0200577, 440.1478271, -570.7902832, 592.8615112
4: -111.6278076, 431.9108276, -103.3947906, 402.6805420, -514.3083496, 535.3056030

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9808060
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9808060
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -104.7348099, 328.1687317, -103.6970367, 325.5162354, -430.2510376, 431.8657837
1: -146.4063873, 331.8330078, -144.8534546, 329.0447693, -475.4511414, 476.6864624
2: -123.8388138, 367.2252502, -122.4707870, 364.2399902, -488.0787659, 489.6960144
3: -130.6424255, 471.8414612, -129.2581787, 468.1342163, -598.7766113, 601.0996094
4: -111.6278076, 431.9108276, -110.4063797, 428.5015564, -540.1293945, 542.3170776

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9808060
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808577, upper bound: 554.9808060
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -104.1153488, 326.4046326, -101.6258926, 318.9038696, -423.0191650, 428.0304871
1: -145.5387878, 329.8930664, -141.8971558, 322.2751160, -467.8138123, 471.7902222
2: -123.0573654, 365.0768738, -119.9465790, 356.8123779, -479.8697510, 485.0234070
3: -129.8116150, 469.1174622, -126.5855942, 458.5134277, -588.3250732, 595.7030640
4: -110.9080734, 429.3227844, -108.1369476, 419.6895142, -530.5975952, 537.4595947

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9793432, upper bound: 554.9807697
time: 1.13 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9793432, upper bound: 554.9807900
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -105.6997452, 331.4900513, -108.2233276, 341.8808899, -447.5806274, 439.7133789
1: -147.8638916, 334.9985657, -151.5287781, 345.3777466, -493.2416077, 486.5273438
2: -125.0009918, 370.6652527, -128.2083282, 382.1197205, -507.1207275, 498.8735962
3: -131.8701782, 476.3554382, -135.1952057, 492.1797180, -624.0498047, 611.5505371
4: -112.6319885, 435.9363098, -115.6126251, 449.8273926, -562.4593506, 551.5489502

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9799121, upper bound: 554.9809572
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9799121, upper bound: 554.9809697
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -106.9028320, 334.9200439, -99.9723892, 315.3446350, -422.2474670, 434.8923950
1: -149.5377655, 338.6195068, -139.7364807, 318.5329895, -468.0707092, 478.3559875
2: -126.4873657, 374.6968689, -118.2134552, 352.4734192, -478.9607849, 492.9102783
3: -133.4022217, 481.4251099, -124.7394333, 453.3199463, -586.7221680, 606.1645508
4: -113.9835739, 440.6597900, -106.5846558, 414.4823608, -528.4659424, 547.2443848

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808553, upper bound: 554.9800605
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -554.9808553, upper bound: 554.9800605
time: 1.12 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.06 + 419.14 = 422.20 seconds
