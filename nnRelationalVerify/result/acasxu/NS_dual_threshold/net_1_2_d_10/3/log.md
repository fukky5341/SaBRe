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
execution time: IAR + RelationalAnalysis = 1.59 + 0.98 = 2.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5950982, upper bound: 0.5950982

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5950982, upper bound: 0.5629308
time: 0.31 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5885052
time: 0.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.77 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.5950982, upper bound: 0.5629308
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.5885052, upper bound: 0.5885052

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0674889, 0.4535027, -0.0365533, 0.6424438, -0.5749549, 0.4900560
1: 0.0201817, 0.5976362, -0.0992057, 0.8423603, -0.8221787, 0.6968419
2: -0.0460193, 0.5242411, -0.1934414, 0.7124153, -0.7584347, 0.7176825
3: -0.1192396, 0.5392005, -0.2859350, 0.8194001, -0.9386396, 0.8251356
4: -0.1227688, 0.6923177, -0.3152393, 0.9457331, -1.0685018, 1.0075570

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5936709, upper bound: 0.5610410
time: 0.34 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5921052, upper bound: 0.5615868
time: 0.34 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0106674, 0.5881721, -0.0365533, 0.6424438, -0.6531112, 0.6247254
1: -0.0680232, 0.7645921, -0.0992057, 0.8423603, -0.9103835, 0.8637978
2: -0.1492940, 0.6635130, -0.1934414, 0.7124153, -0.8617094, 0.8569544
3: -0.2353030, 0.7381678, -0.2859350, 0.8194001, -1.0547031, 1.0241028
4: -0.2566378, 0.8659467, -0.3152393, 0.9457331, -1.2023709, 1.1811860

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5629308, upper bound: 0.5885052
time: 0.30 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5629308, upper bound: 0.5885052
time: 0.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.19 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -0.5936709, upper bound: 0.5610410
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -0.5921052, upper bound: 0.5615868
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -0.5629308, upper bound: 0.5885052
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -0.5629308, upper bound: 0.5885052

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: 0.0667267, 0.5276489, -0.0365533, 0.6424438, -0.5757171, 0.5642022
1: 0.0209925, 0.7202063, -0.0992057, 0.8423603, -0.8213678, 0.8194121
2: -0.0676523, 0.5800259, -0.1934414, 0.7124153, -0.7800677, 0.7734673
3: -0.1276886, 0.6454976, -0.2859350, 0.8194001, -0.9470887, 0.9314327
4: -0.1461909, 0.8046894, -0.3152393, 0.9457331, -1.0919240, 1.1199287

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5867103, upper bound: 0.4877830
time: 0.34 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5936291, upper bound: 0.5608828
time: 0.34 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: 0.0789177, 0.4454548, -0.0365533, 0.6424438, -0.5635260, 0.4820081
1: 0.0311321, 0.5851125, -0.0992057, 0.8423603, -0.8112282, 0.6843182
2: -0.0367235, 0.5172547, -0.1934414, 0.7124153, -0.7491388, 0.7106961
3: -0.1111584, 0.5267107, -0.2859350, 0.8194001, -0.9305584, 0.8126457
4: -0.1149590, 0.6824017, -0.3152393, 0.9457331, -1.0606921, 0.9976410

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5917126, upper bound: 0.5281438
time: 0.33 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5920728, upper bound: 0.5614230
time: 0.36 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0106674, 0.5881721, 0.0674889, 0.4535027, -0.4641702, 0.5206832
1: -0.0680232, 0.7645921, 0.0201817, 0.5976362, -0.6656594, 0.7444105
2: -0.1492940, 0.6635130, -0.0460193, 0.5242411, -0.6735351, 0.7095323
3: -0.2353030, 0.7381678, -0.1192396, 0.5392005, -0.7745036, 0.8574073
4: -0.2566378, 0.8659467, -0.1227688, 0.6923177, -0.9489555, 0.9887154

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5610410, upper bound: 0.5649990
time: 0.30 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5615868, upper bound: 0.5871831
time: 0.30 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0106674, 0.5881721, -0.0106674, 0.5881721, -0.5988396, 0.5988396
1: -0.0680232, 0.7645921, -0.0680232, 0.7645921, -0.8326153, 0.8326153
2: -0.1492940, 0.6635130, -0.1492940, 0.6635130, -0.8128070, 0.8128070
3: -0.2353030, 0.7381678, -0.2353030, 0.7381678, -0.9734708, 0.9734708
4: -0.2566378, 0.8659467, -0.2566378, 0.8659467, -1.1225845, 1.1225845

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5691912
time: 0.40 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5615867, upper bound: 0.5871832
time: 0.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.31 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -0.5867103, upper bound: 0.4877830
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -0.5936291, upper bound: 0.5608828
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -0.5917126, upper bound: 0.5281438
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -0.5920728, upper bound: 0.5614230
NS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 2.31
Output dim: 0, lower bound: -0.5610410, upper bound: 0.5649990
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -0.5615868, upper bound: 0.5871831
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5691912
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.31
Output dim: 0, lower bound: -0.5615867, upper bound: 0.5871832

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0667267, 0.5276489, -0.0585096, 0.6708779, -0.6041512, 0.5861585
1: 0.0209925, 0.7202063, -0.1226516, 0.9202629, -0.8992704, 0.8428579
2: -0.0676523, 0.5800259, -0.2526414, 0.7139475, -0.7815999, 0.8326674
3: -0.1276886, 0.6454976, -0.3343972, 0.8980739, -1.0257626, 0.9798948
4: -0.1461909, 0.8046894, -0.3914803, 1.0063322, -1.1525230, 1.1961697

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 19

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0667267, 0.5276489, -0.0351349, 0.6375349, -0.5708082, 0.5627838
1: 0.0209925, 0.7202063, -0.0974373, 0.8363609, -0.8153684, 0.8176436
2: -0.0676523, 0.5800259, -0.1911077, 0.7069756, -0.7746279, 0.7711337
3: -0.1276886, 0.6454976, -0.2829589, 0.8134989, -0.9411876, 0.9284565
4: -0.1461909, 0.8046894, -0.3120059, 0.9378967, -1.0840876, 1.1166953

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5761132, upper bound: 0.5608828
time: 0.33 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5761132, upper bound: 0.5608828
time: 0.30 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0789177, 0.4454548, -0.0585096, 0.6708779, -0.5919602, 0.5039644
1: 0.0311321, 0.5851125, -0.1226516, 0.9202629, -0.8891308, 0.7077641
2: -0.0367235, 0.5172547, -0.2526414, 0.7139475, -0.7506710, 0.7698961
3: -0.1111584, 0.5267107, -0.3343972, 0.8980739, -1.0092323, 0.8611079
4: -0.1149590, 0.6824017, -0.3914803, 1.0063322, -1.1212912, 1.0738820

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5892326, upper bound: 0.5065091
time: 0.33 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5853600, upper bound: 0.5205425
time: 0.32 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0789177, 0.4454548, -0.0351349, 0.6375349, -0.5586172, 0.4805897
1: 0.0311321, 0.5851125, -0.0974373, 0.8363609, -0.8052288, 0.6825498
2: -0.0367235, 0.5172547, -0.1911077, 0.7069756, -0.7436991, 0.7083625
3: -0.1111584, 0.5267107, -0.2829589, 0.8134989, -0.9246573, 0.8096696
4: -0.1149590, 0.6824017, -0.3120059, 0.9378967, -1.0528557, 0.9944075

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5820026, upper bound: 0.5449055
time: 0.36 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5820026, upper bound: 0.5451149
time: 0.35 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0106674, 0.5881721, 0.0789177, 0.4454548, -0.4561223, 0.5092544
1: -0.0680232, 0.7645921, 0.0311321, 0.5851125, -0.6531357, 0.7334600
2: -0.1492940, 0.6635130, -0.0367235, 0.5172547, -0.6665487, 0.7002365
3: -0.2353030, 0.7381678, -0.1111584, 0.5267107, -0.7620137, 0.8493261
4: -0.2566378, 0.8659467, -0.1149590, 0.6824017, -0.9390395, 0.9809057

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5821036
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5921052
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0069308, 0.7141405, -0.0106674, 0.5881721, -0.5951030, 0.7248080
1: -0.0711197, 0.9631418, -0.0680232, 0.7645921, -0.8357118, 1.0311650
2: -0.1773483, 0.7663486, -0.1492940, 0.6635130, -0.8408613, 0.9156426
3: -0.2441539, 0.9109904, -0.2353030, 0.7381678, -0.9823216, 1.1462934
4: -0.2841340, 1.0538890, -0.2566378, 0.8659467, -1.1500807, 1.3105268

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5458074, upper bound: 0.5532493
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5458074, upper bound: 0.5691912
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0083361, 0.5719668, -0.0106674, 0.5881721, -0.5798361, 0.5826342
1: -0.0506135, 0.7433118, -0.0680232, 0.7645921, -0.8152056, 0.8113350
2: -0.1297052, 0.6455543, -0.1492940, 0.6635130, -0.7932182, 0.7948483
3: -0.2118729, 0.7126579, -0.2353030, 0.7381678, -0.9500406, 0.9479610
4: -0.2322460, 0.8428500, -0.2566378, 0.8659467, -1.0981927, 1.0994878

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5615684, upper bound: 0.5649990
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5615684, upper bound: 0.5871831
time: 0.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.28 seconds
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.5761132, upper bound: 0.5608828
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.5761132, upper bound: 0.5608828
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.5892326, upper bound: 0.5065091
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.5853600, upper bound: 0.5205425
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.5820026, upper bound: 0.5449055
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.5820026, upper bound: 0.5451149
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5821036
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.5450628, upper bound: 0.5921052
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.5458074, upper bound: 0.5532493
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.5458074, upper bound: 0.5691912
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.5615684, upper bound: 0.5649990
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.28
Output dim: 0, lower bound: -0.5615684, upper bound: 0.5871831

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: 0.0667267, 0.5276489, 0.0679543, 0.4516431, -0.3849164, 0.4596946
1: 0.0209925, 0.7202063, 0.0208629, 0.5945244, -0.5735319, 0.6993434
2: -0.0676523, 0.5800259, -0.0448508, 0.5229036, -0.5905559, 0.6248767
3: -0.1276886, 0.6454976, -0.1183286, 0.5365019, -0.6641905, 0.7638262
4: -0.1461909, 0.8046894, -0.1217226, 0.6897607, -0.8359516, 0.9264120

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5761132, upper bound: 0.5460763
time: 0.32 seconds

## Relational analysis of NS_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5761132, upper bound: 0.5608828
time: 0.34 seconds

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: 0.0667267, 0.5276489, -0.0089394, 0.5828843, -0.5161576, 0.5365883
1: 0.0209925, 0.7202063, -0.0659792, 0.7574237, -0.7364312, 0.7861856
2: -0.0676523, 0.5800259, -0.1464720, 0.6577303, -0.7253826, 0.7264979
3: -0.1276886, 0.6454976, -0.2316325, 0.7310752, -0.8587639, 0.8771301
4: -0.1461909, 0.8046894, -0.2524751, 0.8578343, -1.0040252, 1.0571645

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5761132, upper bound: 0.5460763
time: 0.34 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5761132, upper bound: 0.5608828
time: 0.32 seconds

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0789177, 0.4454548, -0.0395788, 0.6407486, -0.5618309, 0.4850336
1: 0.0311321, 0.5851125, -0.1016229, 0.8804287, -0.8492966, 0.6867355
2: -0.0367235, 0.5172547, -0.2235353, 0.6828082, -0.7195317, 0.7407900
3: -0.1111584, 0.5267107, -0.3008456, 0.8524711, -0.9636294, 0.8275563
4: -0.1149590, 0.6824017, -0.3538617, 0.9610786, -1.0760376, 1.0362633

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5804445, upper bound: 0.5065091
time: 0.38 seconds

## Relational analysis of NS_A1_A2_B1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5804445, upper bound: 0.5065091
time: 0.33 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0789177, 0.4454548, -0.0587145, 0.6864064, -0.6074886, 0.5041693
1: 0.0311321, 0.5851125, -0.1260673, 0.9511679, -0.9200358, 0.7111798
2: -0.0367235, 0.5172547, -0.2601123, 0.7187865, -0.7555100, 0.7773670
3: -0.1111584, 0.5267107, -0.3402169, 0.9311209, -1.0422792, 0.8669276
4: -0.1149590, 0.6824017, -0.4028527, 1.0292013, -1.1441603, 1.0852543

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5804445, upper bound: 0.5205425
time: 0.33 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5804445, upper bound: 0.5205425
time: 0.33 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0789177, 0.4454548, -0.0277573, 0.7424477, -0.6635300, 0.4732121
1: 0.0311321, 0.5851125, -0.0962460, 1.0040389, -0.9729068, 0.6813585
2: -0.0367235, 0.5172547, -0.2100114, 0.7920854, -0.8288089, 0.7272661
3: -0.1111584, 0.5267107, -0.2813025, 0.9581012, -1.0692596, 0.8080132
4: -0.1149590, 0.6824017, -0.3271902, 1.0964527, -1.2114117, 1.0095918

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5739167, upper bound: 0.5449055
time: 0.43 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5739167, upper bound: 0.5449055
time: 0.37 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0789177, 0.4454548, -0.0131834, 0.6111942, -0.5322765, 0.4586382
1: 0.0311321, 0.5851125, -0.0753851, 0.8058444, -0.7747123, 0.6604976
2: -0.0367235, 0.5172547, -0.1671922, 0.6776322, -0.7143557, 0.6844469
3: -0.1111584, 0.5267107, -0.2535679, 0.7779697, -0.8891280, 0.7802786
4: -0.1149590, 0.6824017, -0.2808745, 0.9000369, -1.0149959, 0.9632761

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5739167, upper bound: 0.5451149
time: 0.32 seconds

## Relational analysis of NS_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5739167, upper bound: 0.5451149
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0069308, 0.7141405, 0.0789177, 0.4454548, -0.4523857, 0.6352228
1: -0.0711197, 0.9631418, 0.0311321, 0.5851125, -0.6562322, 0.9320097
2: -0.1773483, 0.7663486, -0.0367235, 0.5172547, -0.6946030, 0.8030721
3: -0.2441539, 0.9109904, -0.1111584, 0.5267107, -0.7708645, 1.0221487
4: -0.2841340, 1.0538890, -0.1149590, 0.6824017, -0.9665357, 1.1688480

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4692780, upper bound: 0.5564279
time: 0.37 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4679420, upper bound: 0.5819547
time: 0.30 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0083361, 0.5719668, 0.0789177, 0.4454548, -0.4371188, 0.4930490
1: -0.0506135, 0.7433118, 0.0311321, 0.5851125, -0.6357260, 0.7121797
2: -0.1297052, 0.6455543, -0.0367235, 0.5172547, -0.6469599, 0.6822778
3: -0.2118729, 0.7126579, -0.1111584, 0.5267107, -0.7385836, 0.8238163
4: -0.2322460, 0.8428500, -0.1149590, 0.6824017, -0.9146477, 0.9578090

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4692780, upper bound: 0.5656873
time: 0.31 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0069308, 0.7141405, 0.0083361, 0.5719668, -0.5788976, 0.7058045
1: -0.0711197, 0.9631418, -0.0506135, 0.7433118, -0.8144315, 1.0137553
2: -0.1773483, 0.7663486, -0.1297052, 0.6455543, -0.8229026, 0.8960538
3: -0.2441539, 0.9109904, -0.2118729, 0.7126579, -0.9568118, 1.1228633
4: -0.2841340, 1.0538890, -0.2322460, 0.8428500, -1.1269840, 1.2861351

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 25

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0083361, 0.5719668, 0.0083361, 0.5719668, -0.5636307, 0.5636307
1: -0.0506135, 0.7433118, -0.0506135, 0.7433118, -0.7939253, 0.7939253
2: -0.1297052, 0.6455543, -0.1297052, 0.6455543, -0.7752595, 0.7752595
3: -0.2118729, 0.7126579, -0.2118729, 0.7126579, -0.9245308, 0.9245308
4: -0.2322460, 0.8428500, -0.2322460, 0.8428500, -1.0750960, 1.0750960

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.57 + 63.69 = 66.25 seconds
