## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.5653432899999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0365533, 0.6424438, -0.0365533, 0.6424438, -0.6789970, 0.6789970)
1: (-0.0992057, 0.8423603, -0.0992057, 0.8423603, -0.9415661, 0.9415661)
2: (-0.1934414, 0.7124153, -0.1934414, 0.7124153, -0.9058567, 0.9058567)
3: (-0.2859350, 0.8194001, -0.2859350, 0.8194001, -1.1053351, 1.1053351)
4: (-0.3152393, 0.9457331, -0.3152393, 0.9457331, -1.2609724, 1.2609724)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.46 + 0.98 = 2.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5950982, upper bound: 0.5950982

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5950982, upper bound: 0.5629308
time: 0.30 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5885052
time: 0.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.73 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.5950982, upper bound: 0.5629308
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5885052

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0674889, 0.4535027, -0.0365533, 0.6424438, -0.5749549, 0.4900560
1: 0.0201817, 0.5976362, -0.0992057, 0.8423603, -0.8221787, 0.6968419
2: -0.0460193, 0.5242411, -0.1934414, 0.7124153, -0.7584347, 0.7176825
3: -0.1192396, 0.5392005, -0.2859350, 0.8194001, -0.9386396, 0.8251356
4: -0.1227688, 0.6923177, -0.3152393, 0.9457331, -1.0685018, 1.0075570

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5629308, upper bound: 0.5629308
time: 0.29 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5629308, upper bound: 0.5629308
time: 0.34 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0106674, 0.5881721, -0.0365533, 0.6424438, -0.6531112, 0.6247254
1: -0.0680232, 0.7645921, -0.0992057, 0.8423603, -0.9103835, 0.8637978
2: -0.1492940, 0.6635130, -0.1934414, 0.7124153, -0.8617094, 0.8569544
3: -0.2353030, 0.7381678, -0.2859350, 0.8194001, -1.0547031, 1.0241028
4: -0.2566378, 0.8659467, -0.3152393, 0.9457331, -1.2023709, 1.1811860

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5629308, upper bound: 0.5885052
time: 0.29 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5629308, upper bound: 0.5885052
time: 0.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.07 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.07
Output dim: 0, lower bound: -0.5629308, upper bound: 0.5629308
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.07
Output dim: 0, lower bound: -0.5629308, upper bound: 0.5629308
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -0.5629308, upper bound: 0.5885052
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.07
Output dim: 0, lower bound: -0.5629308, upper bound: 0.5885052

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0106674, 0.5881721, 0.0674889, 0.4535027, -0.4641702, 0.5206832
1: -0.0680232, 0.7645921, 0.0201817, 0.5976362, -0.6656594, 0.7444105
2: -0.1492940, 0.6635130, -0.0460193, 0.5242411, -0.6735351, 0.7095323
3: -0.2353030, 0.7381678, -0.1192396, 0.5392005, -0.7745036, 0.8574073
4: -0.2566378, 0.8659467, -0.1227688, 0.6923177, -0.9489555, 0.9887154

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5691912
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5615867, upper bound: 0.5871832
time: 0.38 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0106674, 0.5881721, -0.0106674, 0.5881721, -0.5988396, 0.5988396
1: -0.0680232, 0.7645921, -0.0680232, 0.7645921, -0.8326153, 0.8326153
2: -0.1492940, 0.6635130, -0.1492940, 0.6635130, -0.8128070, 0.8128070
3: -0.2353030, 0.7381678, -0.2353030, 0.7381678, -0.9734708, 0.9734708
4: -0.2566378, 0.8659467, -0.2566378, 0.8659467, -1.1225845, 1.1225845

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5691912
time: 0.39 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5615867, upper bound: 0.5871832
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.14 seconds
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5691912
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -0.5615867, upper bound: 0.5871832
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5691912
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.14
Output dim: 0, lower bound: -0.5615867, upper bound: 0.5871832

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0069308, 0.7141405, 0.0674889, 0.4535027, -0.4604335, 0.6466516
1: -0.0711197, 0.9631418, 0.0201817, 0.5976362, -0.6687558, 0.9429601
2: -0.1773483, 0.7663486, -0.0460193, 0.5242411, -0.7015893, 0.8123679
3: -0.2441539, 0.9109904, -0.1192396, 0.5392005, -0.7833544, 1.0302299
4: -0.2841340, 1.0538890, -0.1227688, 0.6923177, -0.9764518, 1.1766578

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5820916
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5821036
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0083361, 0.5719668, 0.0674889, 0.4535027, -0.4451666, 0.5044779
1: -0.0506135, 0.7433118, 0.0201817, 0.5976362, -0.6482497, 0.7231302
2: -0.1297052, 0.6455543, -0.0460193, 0.5242411, -0.6539463, 0.6915736
3: -0.2118729, 0.7126579, -0.1192396, 0.5392005, -0.7510734, 0.8318975
4: -0.2322460, 0.8428500, -0.1227688, 0.6923177, -0.9245638, 0.9656187

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5610410, upper bound: 0.5915924
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5610410, upper bound: 0.5921052
time: 0.30 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0069308, 0.7141405, -0.0106674, 0.5881721, -0.5951030, 0.7248080
1: -0.0711197, 0.9631418, -0.0680232, 0.7645921, -0.8357118, 1.0311650
2: -0.1773483, 0.7663486, -0.1492940, 0.6635130, -0.8408613, 0.9156426
3: -0.2441539, 0.9109904, -0.2353030, 0.7381678, -0.9823216, 1.1462934
4: -0.2841340, 1.0538890, -0.2566378, 0.8659467, -1.1500807, 1.3105268

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5458074, upper bound: 0.5532493
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5458074, upper bound: 0.5691912
time: 0.36 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0083361, 0.5719668, -0.0106674, 0.5881721, -0.5798361, 0.5826342
1: -0.0506135, 0.7433118, -0.0680232, 0.7645921, -0.8152056, 0.8113350
2: -0.1297052, 0.6455543, -0.1492940, 0.6635130, -0.7932182, 0.7948483
3: -0.2118729, 0.7126579, -0.2353030, 0.7381678, -0.9500406, 0.9479610
4: -0.2322460, 0.8428500, -0.2566378, 0.8659467, -1.0981927, 1.0994878

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5615684, upper bound: 0.5649990
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5615684, upper bound: 0.5871831
time: 0.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.13 seconds
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5820916
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5821036
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 0, lower bound: -0.5610410, upper bound: 0.5915924
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 0, lower bound: -0.5610410, upper bound: 0.5921052
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.13
Output dim: 0, lower bound: -0.5458074, upper bound: 0.5532493
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 0, lower bound: -0.5458074, upper bound: 0.5691912
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.13
Output dim: 0, lower bound: -0.5615684, upper bound: 0.5649990
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.13
Output dim: 0, lower bound: -0.5615684, upper bound: 0.5871831

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0069308, 0.7141405, 0.0667267, 0.5276489, -0.5345798, 0.6474138
1: -0.0711197, 0.9631418, 0.0209925, 0.7202063, -0.7913260, 0.9421493
2: -0.1773483, 0.7663486, -0.0676523, 0.5800259, -0.7573742, 0.8340009
3: -0.2441539, 0.9109904, -0.1276886, 0.6454976, -0.8896515, 1.0386790
4: -0.2841340, 1.0538890, -0.1461909, 0.8046894, -1.0888234, 1.2000799

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4692780, upper bound: 0.5564279
time: 0.29 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4679420, upper bound: 0.5819449
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0069308, 0.7141405, 0.0789177, 0.4454548, -0.4523857, 0.6352228
1: -0.0711197, 0.9631418, 0.0311321, 0.5851125, -0.6562322, 0.9320097
2: -0.1773483, 0.7663486, -0.0367235, 0.5172547, -0.6946030, 0.8030721
3: -0.2441539, 0.9109904, -0.1111584, 0.5267107, -0.7708645, 1.0221487
4: -0.2841340, 1.0538890, -0.1149590, 0.6824017, -0.9665357, 1.1688480

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4692780, upper bound: 0.5564279
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4679420, upper bound: 0.5819547
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0083361, 0.5719668, 0.0667267, 0.5276489, -0.5193129, 0.5052401
1: -0.0506135, 0.7433118, 0.0209925, 0.7202063, -0.7708198, 0.7223193
2: -0.1297052, 0.6455543, -0.0676523, 0.5800259, -0.7097311, 0.7132066
3: -0.2118729, 0.7126579, -0.1276886, 0.6454976, -0.8573705, 0.8403466
4: -0.2322460, 0.8428500, -0.1461909, 0.8046894, -1.0369354, 0.9890409

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4877828, upper bound: 0.5867103
time: 0.36 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5608827, upper bound: 0.5915616
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0083361, 0.5719668, 0.0789177, 0.4454548, -0.4371188, 0.4930490
1: -0.0506135, 0.7433118, 0.0311321, 0.5851125, -0.6357260, 0.7121797
2: -0.1297052, 0.6455543, -0.0367235, 0.5172547, -0.6469599, 0.6822778
3: -0.2118729, 0.7126579, -0.1111584, 0.5267107, -0.7385836, 0.8238163
4: -0.2322460, 0.8428500, -0.1149590, 0.6824017, -0.9146477, 0.9578090

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4877830, upper bound: 0.5917127
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5608828, upper bound: 0.5920728
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0069308, 0.7141405, 0.0083361, 0.5719668, -0.5788976, 0.7058045
1: -0.0711197, 0.9631418, -0.0506135, 0.7433118, -0.8144315, 1.0137553
2: -0.1773483, 0.7663486, -0.1297052, 0.6455543, -0.8229026, 0.8960538
3: -0.2441539, 0.9109904, -0.2118729, 0.7126579, -0.9568118, 1.1228633
4: -0.2841340, 1.0538890, -0.2322460, 0.8428500, -1.1269840, 1.2861351

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0083361, 0.5719668, 0.0083361, 0.5719668, -0.5636307, 0.5636307
1: -0.0506135, 0.7433118, -0.0506135, 0.7433118, -0.7939253, 0.7939253
2: -0.1297052, 0.6455543, -0.1297052, 0.6455543, -0.7752595, 0.7752595
3: -0.2118729, 0.7126579, -0.2118729, 0.7126579, -0.9245308, 0.9245308
4: -0.2322460, 0.8428500, -0.2322460, 0.8428500, -1.0750960, 1.0750960

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.44 + 36.63 = 39.07 seconds
