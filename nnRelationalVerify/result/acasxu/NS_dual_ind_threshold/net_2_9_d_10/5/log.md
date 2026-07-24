## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 5693.26040512119


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3552.1267090, 2750.7805176, -3552.1267090, 2750.7805176, -6302.9072266, 6302.9072266)
1: (-294.4743042, 208.4085846, -294.4743042, 208.4085846, -502.8828430, 502.8828430)
2: (-202.6398773, 349.5717773, -202.6398773, 349.5717773, -552.2115479, 552.2115479)
3: (-246.1018677, 512.2490845, -246.1018677, 512.2490845, -758.3508301, 758.3508301)
4: (-197.5094757, 359.1516113, -197.5094757, 359.1516113, -556.6610718, 556.6610718)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.09 + 2.21 = 5.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -5693.8297881, upper bound: 5693.8297881

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297881, upper bound: 5693.8297881
time: 0.68 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297881, upper bound: 5693.8297881
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.65 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 0, lower bound: -5693.8297881, upper bound: 5693.8297881
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 0, lower bound: -5693.8297881, upper bound: 5693.8297881

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3258.4960938, 2554.5190430, -3391.8928223, 2644.3701172, -5902.8652344, 5946.4116211
1: -272.6817322, 191.2189178, -282.6142273, 199.0559387, -471.7376709, 473.8331299
2: -185.8512726, 323.1545105, -193.4045715, 335.2016602, -521.0527954, 516.5590210
3: -226.0303497, 472.7031250, -235.0782471, 490.6989746, -716.7293091, 707.7813721
4: -181.5751495, 331.8414612, -188.7209167, 344.2978516, -525.8729858, 520.5623779

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297778, upper bound: 5693.8297791
time: 0.63 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297791, upper bound: 5693.8297790
time: 0.65 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3709.0078125, 2913.5207520, -3474.2268066, 2697.4882812, -6406.4960938, 6387.7475586
1: -311.0048828, 218.6035919, -288.5811157, 203.8171387, -514.8220215, 507.1846924
2: -215.8076630, 369.9015198, -198.0830231, 342.5096130, -558.3171387, 567.9845581
3: -262.0737610, 539.0894165, -240.6807404, 501.6675720, -763.7413330, 779.7701416
4: -210.4501495, 379.2772217, -193.1802673, 351.8709412, -562.3211060, 572.4573975

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297778, upper bound: 5693.8297790
time: 0.66 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297788, upper bound: 5693.8297790
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.37 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.37
Output dim: 0, lower bound: -5693.8297778, upper bound: 5693.8297791
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.37
Output dim: 0, lower bound: -5693.8297791, upper bound: 5693.8297790
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.37
Output dim: 0, lower bound: -5693.8297778, upper bound: 5693.8297790
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.37
Output dim: 0, lower bound: -5693.8297788, upper bound: 5693.8297790

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3221.1994629, 2526.5178223, -3276.8964844, 2558.2548828, -5779.4541016, 5803.4135742
1: -269.7080688, 188.9573059, -273.4557495, 192.0692749, -461.7773438, 462.4130249
2: -183.5847015, 319.4862976, -186.4002991, 323.9200134, -507.5046082, 505.8865967
3: -223.3195343, 467.3773804, -226.7043457, 474.2879333, -697.6074829, 694.0817261
4: -179.4344025, 328.0921326, -182.0988617, 332.7666931, -512.2011108, 510.1909485

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297442, upper bound: 5693.8297784
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297778, upper bound: 5693.8297788
time: 0.68 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3205.4819336, 2516.1064453, -3358.6342773, 2622.7722168, -5828.2539062, 5874.7407227
1: -268.5484924, 188.0226440, -280.2668152, 196.8130951, -465.3615723, 468.2893982
2: -182.7089844, 318.1445007, -191.2009583, 332.2777100, -514.9866943, 509.3453979
3: -222.2961578, 465.2529297, -232.5644684, 486.2658997, -708.5620728, 697.8173828
4: -178.5865631, 326.6849670, -186.8347778, 341.3394165, -519.9259644, 513.5196533

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297771, upper bound: 5693.8297789
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297769, upper bound: 5693.8297769
time: 0.65 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3666.3562012, 2880.7727051, -3356.7448730, 2608.4519043, -6274.8081055, 6237.5166016
1: -307.5176392, 215.9824982, -279.1216736, 196.6562347, -504.1738892, 495.1041565
2: -213.1251831, 365.6466370, -190.8621674, 330.8623962, -543.9874878, 556.5087280
3: -258.8401184, 532.9215698, -232.0287018, 484.8043213, -743.6442871, 764.9501343
4: -207.8750916, 374.9312439, -186.3397827, 340.0096436, -547.8846436, 561.2709961

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297737, upper bound: 5693.8297721
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297737, upper bound: 5693.8297744
time: 0.69 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3647.8620605, 2870.0712891, -3450.5671387, 2681.2917480, -6329.1538086, 6320.6376953
1: -306.2716064, 214.9206085, -286.8549194, 202.1117859, -508.3833008, 501.7755127
2: -212.2128754, 364.2180481, -196.3916626, 340.3193970, -552.5322266, 560.6096802
3: -257.7741699, 530.5578003, -238.7492828, 498.4507751, -756.2249756, 769.3070679
4: -207.0057068, 373.4195862, -191.7703552, 349.7203674, -556.7260742, 565.1898193

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297750, upper bound: 5693.8297721
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297750, upper bound: 5693.8297744
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.17 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 0, lower bound: -5693.8297442, upper bound: 5693.8297784
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 0, lower bound: -5693.8297778, upper bound: 5693.8297788
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 0, lower bound: -5693.8297771, upper bound: 5693.8297789
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 0, lower bound: -5693.8297769, upper bound: 5693.8297769
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 0, lower bound: -5693.8297737, upper bound: 5693.8297721
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 0, lower bound: -5693.8297737, upper bound: 5693.8297744
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 0, lower bound: -5693.8297750, upper bound: 5693.8297721
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.17
Output dim: 0, lower bound: -5693.8297750, upper bound: 5693.8297744

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3136.8208008, 2448.0227051, -3195.3420410, 2493.2778320, -5630.0986328, 5643.3647461
1: -261.5779114, 184.9218445, -266.5528870, 187.4663239, -449.0442200, 451.4747314
2: -179.3447876, 310.2882996, -181.9514923, 315.8022766, -495.1470642, 492.2397766
3: -217.7619324, 454.3684387, -221.1861420, 462.6006470, -680.3625488, 675.5545654
4: -175.0041046, 318.5545349, -177.7067413, 324.3446655, -499.3487244, 496.2612610

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297133, upper bound: 5693.8297743
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297377, upper bound: 5693.8297746
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3196.0888672, 2506.4096680, -3276.8964844, 2558.2548828, -5754.3437500, 5783.3056641
1: -267.5941467, 187.4685364, -273.4557495, 192.0692749, -459.6634216, 460.9242859
2: -182.1162720, 316.9140320, -186.4002991, 323.9200134, -506.0361633, 503.3143311
3: -221.5419617, 463.6735535, -226.7043457, 474.2879333, -695.8298950, 690.3777466
4: -177.9962006, 325.4358215, -182.0988617, 332.7666931, -510.7628784, 507.5345764

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297692, upper bound: 5693.8297752
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297733, upper bound: 5693.8297752
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3084.5390625, 2428.2434082, -3280.1984863, 2565.7871094, -5650.3251953, 5708.4418945
1: -259.0468750, 180.7317200, -274.0917358, 192.0765991, -451.1234741, 454.8233948
2: -175.5318909, 306.7290039, -186.5467529, 324.8645020, -500.3963928, 493.2757263
3: -213.7675781, 448.4678345, -227.0397186, 475.3461304, -689.1135254, 675.5075684
4: -171.8383179, 314.9244385, -182.4511566, 333.6987610, -505.5370483, 497.3755798

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297720, upper bound: 5693.8297750
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297750, upper bound: 5693.8297764
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3331.8295898, 2618.6132812, -3297.6157227, 2576.5690918, -5908.3984375, 5916.2290039
1: -279.3110046, 195.9529877, -275.2831421, 193.1350403, -472.4460449, 471.2361450
2: -190.1842499, 331.4261780, -187.4118652, 326.3096008, -516.4938354, 518.8380127
3: -231.2799072, 484.8575134, -228.0477295, 477.5785828, -708.8583984, 712.9052124
4: -186.0662079, 340.3319092, -183.2454529, 335.2496948, -521.3159180, 523.5773315

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297725, upper bound: 5693.8297732
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297750, upper bound: 5693.8297736
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3587.0068359, 2823.0834961, -3320.9565430, 2582.8583984, -6169.8652344, 6144.0385742
1: -301.2445374, 211.1838684, -276.3536377, 194.5093689, -495.7539062, 487.5375061
2: -208.3085632, 358.0579529, -188.7444458, 327.4535522, -535.7620850, 546.8022461
3: -253.0674133, 521.8152466, -229.5106049, 479.8237305, -732.8911133, 751.3258057
4: -203.2913971, 367.1704712, -184.3457336, 336.5025024, -539.7938843, 551.5161743

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297622
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297713, upper bound: 5693.8297689
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3616.4267578, 2849.3635254, -3303.7026367, 2570.9221191, -6187.3486328, 6153.0664062
1: -303.9840393, 213.1044464, -275.0464478, 193.5155792, -497.4996338, 488.1508789
2: -209.8126068, 361.2061768, -187.7647247, 325.9067688, -535.7193604, 548.9708862
3: -254.8677521, 526.4301758, -228.3285522, 477.5374756, -732.4052124, 754.7587280
4: -204.7521973, 370.3325806, -183.4088135, 334.8867798, -539.6388550, 553.7413940

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297501, upper bound: 5693.8297674
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297721, upper bound: 5693.8297722
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3569.8676758, 2813.2487793, -3416.8479004, 2656.9545898, -6226.8212891, 6230.0957031
1: -300.1016541, 210.1969452, -284.2279358, 200.0773010, -500.1789551, 494.4248657
2: -207.4517517, 356.7412415, -194.3924561, 337.0769653, -544.5285034, 551.1336060
3: -252.0720062, 519.6295776, -236.3754883, 493.7316589, -745.8035278, 756.0050659
4: -202.4729462, 365.7718506, -189.8885345, 346.3944702, -548.8673706, 555.6603394

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297627, upper bound: 5693.8297667
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297727, upper bound: 5693.8297688
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3599.4047852, 2839.6999512, -3393.9907227, 2641.5195312, -6240.9238281, 6233.6894531
1: -302.8581543, 212.1150360, -282.5311890, 198.7716064, -501.6297607, 494.6462402
2: -208.9619598, 359.9015198, -193.1180267, 335.0582275, -544.0201416, 553.0194702
3: -253.8788452, 524.2540283, -234.8378448, 490.7095032, -744.5883789, 759.0918579
4: -203.9355164, 368.9501343, -188.6682587, 344.2765808, -548.2120972, 557.6184082

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297628, upper bound: 5693.8297689
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297726, upper bound: 5693.8297722
time: 0.71 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.57 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297133, upper bound: 5693.8297743
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297377, upper bound: 5693.8297746
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297692, upper bound: 5693.8297752
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297733, upper bound: 5693.8297752
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297720, upper bound: 5693.8297750
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297750, upper bound: 5693.8297764
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297725, upper bound: 5693.8297732
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297750, upper bound: 5693.8297736
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297622
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297713, upper bound: 5693.8297689
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297501, upper bound: 5693.8297674
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297721, upper bound: 5693.8297722
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297627, upper bound: 5693.8297667
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297727, upper bound: 5693.8297688
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297628, upper bound: 5693.8297689
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 0, lower bound: -5693.8297726, upper bound: 5693.8297722

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3106.9965820, 2426.6262207, -3125.1342773, 2442.7097168, -5549.7060547, 5551.7583008
1: -259.2481689, 183.1363220, -261.1015320, 183.2709656, -442.5191345, 444.2378235
2: -177.5505066, 307.4504089, -177.8521423, 309.1022034, -486.6527100, 485.3025513
3: -215.6120453, 450.2083130, -216.3113251, 452.8360901, -668.4479980, 666.5196533
4: -173.3029938, 315.6470337, -173.8690643, 317.4796143, -490.7825928, 489.5161133

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297133, upper bound: 5693.8297726
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3074.9343262, 2404.6682129, -3142.5876465, 2463.4389648, -5538.3730469, 5547.2558594
1: -256.8527832, 181.2507629, -263.1699219, 184.4818573, -441.3345947, 444.4206848
2: -175.7387543, 304.5736694, -178.8305054, 311.4608765, -487.1996155, 483.4041748
3: -213.4576111, 445.9179077, -217.5617371, 455.9268799, -669.3845215, 663.4796143
4: -171.6063538, 312.6821899, -174.8682861, 319.7934570, -491.3998108, 487.5504761

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297306, upper bound: 5693.8297731
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3163.9055176, 2482.9794922, -3207.4638672, 2507.9367676, -5671.8422852, 5690.4428711
1: -265.0802002, 185.5360107, -268.0381470, 187.9055634, -452.9857483, 453.5741577
2: -180.2150421, 313.8207092, -182.3226624, 317.2563782, -497.4714355, 496.1433716
3: -219.2850800, 459.1940613, -221.8622131, 464.6018066, -683.8867798, 681.0562744
4: -176.2197113, 322.2642517, -178.2803650, 325.9203796, -502.1400757, 500.5446167

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297670, upper bound: 5693.8297706
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297670, upper bound: 5693.8297748
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3144.5676270, 2469.5830078, -3226.4780273, 2528.9558105, -5673.5234375, 5696.0610352
1: -263.6154480, 184.4357605, -270.1625671, 189.1931763, -452.8085327, 454.5983276
2: -179.1380463, 312.0837097, -183.3457489, 319.7029724, -498.8410034, 495.4294128
3: -217.9765320, 456.6087952, -223.1637421, 467.8512573, -685.8277588, 679.7725220
4: -175.1854706, 320.4645386, -179.3373260, 328.3417664, -503.5272217, 499.8018799

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297732, upper bound: 5693.8297705
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297732, upper bound: 5693.8297752
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3012.3493652, 2372.5095215, -3171.2382812, 2481.2685547, -5493.6171875, 5543.7480469
1: -253.1251984, 176.3430023, -265.1055603, 185.4508057, -438.5759888, 441.4485474
2: -170.9636688, 299.5263672, -179.6564178, 313.9827576, -484.9464111, 479.1827393
3: -208.3424377, 438.1256104, -218.8523407, 459.6930542, -668.0355225, 656.9779663
4: -167.5212250, 307.5708313, -175.9422302, 322.6051331, -490.1263428, 483.5130615

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297510, upper bound: 5693.8297582
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297510, upper bound: 5693.8297751
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3004.4177246, 2372.3515625, -3264.5488281, 2565.2673340, -5569.6835938, 5636.9003906
1: -252.9020386, 175.9702148, -273.8424988, 191.1464539, -444.0484924, 449.8127136
2: -170.7626038, 299.3022461, -185.3799744, 324.1945801, -494.9571838, 484.6821899
3: -208.1069183, 437.5160217, -225.8269348, 474.0284119, -682.1353149, 663.3429565
4: -167.3368225, 307.2514954, -181.3990784, 332.8953552, -500.2321777, 488.6505127

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297680, upper bound: 5693.8297706
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297680, upper bound: 5693.8297770
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3246.9418945, 2554.8500977, -3192.4287109, 2494.5925293, -5741.5341797, 5747.2763672
1: -272.4971619, 190.8450165, -266.5606995, 186.7411804, -459.2383118, 457.4056702
2: -184.9988556, 323.1899109, -180.7002106, 315.7951965, -500.7940369, 503.8901367
3: -225.0634460, 472.9760437, -220.0727692, 462.4605713, -687.5239258, 693.0488281
4: -181.1568756, 331.9042969, -176.9029236, 324.5188599, -505.6757202, 508.8072205

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297519, upper bound: 5693.8297505
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297519, upper bound: 5693.8297732
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3270.0747070, 2574.7624512, -3292.7404785, 2583.1418457, -5853.2163086, 5867.5029297
1: -274.5148315, 192.2624969, -275.7785645, 192.8536835, -467.3684387, 468.0410767
2: -186.4056091, 325.6319580, -186.8198242, 326.6194153, -513.0250244, 512.4517822
3: -226.7892914, 476.3387756, -227.5098267, 477.7369995, -704.5263062, 703.8486328
4: -182.4790497, 334.3721924, -182.7255859, 335.4310303, -517.9100342, 517.0977783

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297678, upper bound: 5693.8297503
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297678, upper bound: 5693.8297735
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3502.9033203, 2759.5605469, -3200.8122559, 2492.0700684, -5994.9721680, 5960.3710938
1: -294.4180298, 206.1127930, -266.6164856, 187.2634583, -481.6814880, 472.7292786
2: -202.9298248, 349.8145447, -181.1727295, 315.7156372, -518.6454468, 530.9872437
3: -246.6327820, 509.9415894, -220.5318604, 462.8116150, -709.4443970, 730.4733887
4: -198.1674500, 358.7556458, -177.1924896, 324.4970703, -522.6644897, 535.9480591

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297601
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297620
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3500.5656738, 2763.1679688, -3328.9506836, 2597.5971680, -6098.1630859, 6092.1186523
1: -294.6142273, 206.0465240, -277.7698059, 194.9701691, -489.5843811, 483.8163452
2: -203.1135559, 350.1225281, -188.8735046, 328.8377686, -531.9512329, 538.9960327
3: -246.8815155, 510.0438843, -229.8262634, 481.5760193, -728.4575195, 739.8701172
4: -198.3611450, 359.0274048, -184.5046539, 337.8272095, -536.1883545, 543.5319824

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297697
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297698
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3532.4047852, 2786.0427246, -3181.5979004, 2478.7053223, -6011.1098633, 5967.6406250
1: -297.1766357, 208.0440063, -265.1635742, 186.1546936, -483.3312378, 473.2074890
2: -204.4458313, 352.9710999, -180.0768433, 313.9613953, -518.4072266, 533.0479736
3: -248.4431305, 514.5592041, -219.2167816, 460.2477722, -708.6909180, 733.7759399
4: -199.6256714, 361.9318237, -176.1553345, 322.6781921, -522.3038330, 538.0871582

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297665
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297674
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3531.4074707, 2790.1840820, -3308.7861328, 2584.0610352, -6115.4682617, 6098.9702148
1: -297.4456482, 208.0559845, -276.2791443, 193.8118896, -491.2575378, 484.3350220
2: -204.6710968, 353.3355713, -187.7561340, 327.0481873, -531.7192383, 541.0916138
3: -248.7470398, 514.8554688, -228.4906311, 478.9238586, -727.6708984, 743.3460693
4: -199.8825073, 362.2765503, -183.4465027, 335.9752197, -535.8577271, 545.7230225

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297712
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297727
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3487.4089355, 2750.5463867, -3300.6071777, 2568.0632324, -6055.4716797, 6051.1533203
1: -293.3680115, 205.2028046, -274.7120667, 193.0437622, -486.4117126, 479.9148560
2: -202.1395874, 348.6048279, -187.0105896, 325.6140137, -527.7535400, 535.6154175
3: -245.7180481, 507.9241638, -227.6125488, 477.1114502, -722.8294678, 735.5366821
4: -197.4035645, 357.4767456, -182.9010620, 334.6625366, -532.0660400, 540.3777466

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297647
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297656
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3482.8596191, 2752.7687988, -3417.0583496, 2666.8093262, -6149.6689453, 6169.8266602
1: -293.4207458, 205.0235596, -285.0947876, 200.0697021, -493.4904175, 490.1183472
2: -202.2115173, 348.7422180, -194.0800476, 337.8158875, -540.0274048, 542.8222656
3: -245.8299103, 507.7668152, -236.1561127, 494.5251160, -740.3550415, 743.9229126
4: -197.4963379, 357.5583496, -189.6313934, 346.9880066, -544.4843140, 547.1897583

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297689
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297689
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3516.5959473, 2776.8955078, -3268.7287598, 2546.4526367, -6063.0488281, 6045.6240234
1: -296.1163025, 207.1180878, -272.3489380, 191.2155762, -487.3318787, 479.4669800
2: -203.6427002, 351.7446899, -185.1996765, 322.7571716, -526.3997803, 536.9443359
3: -247.5104370, 512.5231323, -225.4497528, 472.8846436, -720.3948975, 737.9729004
4: -198.8594666, 360.6245728, -181.1885529, 331.7024841, -530.5619507, 541.8131104

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297671
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297676
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3512.6645508, 2778.8278809, -3389.6784668, 2648.1728516, -6160.8374023, 6168.5063477
1: -296.1579285, 206.9607391, -283.0579834, 198.4990997, -494.6570129, 490.0187378
2: -203.7049408, 351.8287659, -192.5449371, 335.3660278, -539.0709839, 544.3734741
3: -247.6120453, 512.4058838, -234.3175659, 490.8702698, -738.4820557, 746.7233887
4: -198.9581909, 360.6814270, -188.1708221, 344.4447632, -543.4028320, 548.8522339

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297712
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297727
time: 0.61 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.77 seconds
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297670, upper bound: 5693.8297706
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297670, upper bound: 5693.8297748
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297732, upper bound: 5693.8297705
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297732, upper bound: 5693.8297752
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297510, upper bound: 5693.8297582
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297510, upper bound: 5693.8297751
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297680, upper bound: 5693.8297706
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297680, upper bound: 5693.8297770
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297519, upper bound: 5693.8297505
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297519, upper bound: 5693.8297732
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297678, upper bound: 5693.8297503
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297678, upper bound: 5693.8297735
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297601
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297620
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297697
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297698
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297665
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297674
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297712
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297727
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297647
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297656
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297689
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297689
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297671
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297507, upper bound: 5693.8297676
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297712
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.77
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297727

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3090.6147461, 2428.5747070, -3207.4638672, 2507.9367676, -5598.5507812, 5636.0380859
1: -259.2886353, 181.1012268, -268.0381470, 187.9055634, -447.1942139, 449.1393433
2: -175.7523041, 306.6698608, -182.3226624, 317.2563782, -493.0086670, 488.9925232
3: -213.9884033, 448.7809143, -221.8622131, 464.6018066, -678.5900879, 670.6431274
4: -172.0180969, 314.9531250, -178.2803650, 325.9203796, -497.9384766, 493.2334900

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297433, upper bound: 5693.8297525
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297654, upper bound: 5693.8297688
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3173.1174316, 2494.0681152, -3207.4638672, 2507.9367676, -5681.0541992, 5701.5302734
1: -266.1759338, 185.9175110, -268.0381470, 187.9055634, -454.0814819, 453.9556274
2: -180.6437988, 315.1248779, -182.3226624, 317.2563782, -497.9000854, 497.4475403
3: -219.9475098, 460.8734741, -221.8622131, 464.6018066, -684.5492554, 682.7357178
4: -176.8218384, 323.6166687, -178.2803650, 325.9203796, -502.7422180, 501.8970337

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297433, upper bound: 5693.8297695
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297654, upper bound: 5693.8297727
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3071.3906250, 2415.2365723, -3226.4780273, 2528.9558105, -5600.3466797, 5641.7138672
1: -257.8356018, 180.0034180, -270.1625671, 189.1931763, -447.0286865, 450.1659851
2: -174.6832733, 304.9447327, -183.3457489, 319.7029724, -494.3862000, 488.2904358
3: -212.6924591, 446.2235107, -223.1637421, 467.8512573, -680.5436401, 669.3872681
4: -170.9938049, 313.1463623, -179.3373260, 328.3417664, -499.3355103, 492.4837036

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297360, upper bound: 5693.8297488
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297688
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3149.1745605, 2478.1323242, -3226.4780273, 2528.9558105, -5678.1293945, 5704.6103516
1: -264.4230347, 184.5718079, -270.1625671, 189.1931763, -453.6161804, 454.7343445
2: -179.3404083, 313.0447388, -183.3457489, 319.7029724, -499.0433960, 496.3904724
3: -218.3814240, 457.7559204, -223.1637421, 467.8512573, -686.2326660, 680.9195557
4: -175.5913086, 321.4310913, -179.3373260, 328.3417664, -503.9330444, 500.7684326

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297360, upper bound: 5693.8297671
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297728
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2945.9621582, 2322.5600586, -3171.2382812, 2481.2685547, -5427.2290039, 5493.7983398
1: -247.8460999, 172.3127289, -265.1055603, 185.4508057, -433.2969055, 437.4182739
2: -166.8696442, 292.9187317, -179.6564178, 313.9827576, -480.8524170, 472.5751038
3: -203.4680939, 428.6735229, -218.8523407, 459.6930542, -663.1611328, 647.5258789
4: -163.6853485, 300.8114929, -175.9422302, 322.6051331, -486.2904663, 476.7537231

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297414, upper bound: 5693.8297502
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297446, upper bound: 5693.8297509
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297367, upper bound: 5693.8297528
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297367, upper bound: 5693.8297581
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3025.2590332, 2384.8046875, -3171.2382812, 2481.2685547, -5506.5263672, 5556.0429688
1: -254.4228058, 176.8934021, -265.1055603, 185.4508057, -439.8735962, 441.9989624
2: -171.5940704, 300.9878235, -179.6564178, 313.9827576, -485.5768433, 480.6441650
3: -209.2149506, 440.1971130, -218.8523407, 459.6930542, -668.9080200, 659.0493774
4: -168.3269501, 309.0900269, -175.9422302, 322.6051331, -490.9320679, 485.0322571

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297414, upper bound: 5693.8297690
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297445, upper bound: 5693.8297498
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297367, upper bound: 5693.8297517
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297369, upper bound: 5693.8297746
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2943.4648438, 2325.3586426, -3264.5488281, 2565.2673340, -5508.7309570, 5589.9072266
1: -247.9661255, 172.2543945, -273.8424988, 191.1464539, -439.1125793, 446.0968933
2: -166.9467468, 293.1173401, -185.3799744, 324.1945801, -491.1413269, 478.4972229
3: -203.5538940, 428.7386780, -225.8269348, 474.0284119, -677.5822754, 654.5656128
4: -163.7559814, 300.9371948, -181.3990784, 332.8953552, -496.6513367, 482.3362122

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297649, upper bound: 5693.8297677
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297503, upper bound: 5693.8297628
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297505, upper bound: 5693.8297706
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3014.9707031, 2383.1479492, -3264.5488281, 2565.2673340, -5580.2373047, 5647.6967773
1: -254.0436707, 176.4174805, -273.8424988, 191.1464539, -445.1901245, 450.2599792
2: -171.2649078, 300.5686951, -185.3799744, 324.1945801, -495.4594727, 485.9486694
3: -208.8358917, 439.3384705, -225.8269348, 474.0284119, -682.8643188, 665.1654053
4: -168.0306396, 308.5655518, -181.3990784, 332.8953552, -500.9259949, 489.9645691

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297649, upper bound: 5693.8297752
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297505, upper bound: 5693.8297751
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297505, upper bound: 5693.8297769
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3178.2758789, 2503.0295410, -3192.4287109, 2494.5925293, -5672.8681641, 5695.4580078
1: -267.0198975, 186.6692657, -266.5606995, 186.7411804, -453.7610779, 453.2299194
2: -180.8055878, 316.3440857, -180.7002106, 315.7951965, -496.6007690, 497.0443115
3: -220.0504913, 463.1165771, -220.0727692, 462.4605713, -682.5109863, 683.1893311
4: -177.2169647, 324.9309692, -176.9029236, 324.5188599, -501.7358093, 501.8338928

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297368, upper bound: 5693.8297506
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297368, upper bound: 5693.8297503
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3265.5422363, 2570.0827637, -3192.4287109, 2494.5925293, -5760.1347656, 5762.5117188
1: -274.1297607, 191.7851562, -266.5606995, 186.7411804, -460.8708801, 458.3458557
2: -185.8871460, 325.0429688, -180.7002106, 315.7951965, -501.6823425, 505.7431641
3: -226.2059631, 475.7932739, -220.0727692, 462.4605713, -688.6665039, 695.8660278
4: -182.2168121, 333.8335876, -176.9029236, 324.5188599, -506.7356262, 510.7365112

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297368, upper bound: 5693.8297735
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297368, upper bound: 5693.8297735
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3202.0676270, 2522.9003906, -3292.7404785, 2583.1418457, -5785.2084961, 5815.6401367
1: -269.0587769, 188.1008148, -275.7785645, 192.8536835, -461.9123535, 463.8793945
2: -182.2489319, 318.8060608, -186.8198242, 326.6194153, -508.8682861, 505.6258850
3: -221.8019562, 466.5598755, -227.5098267, 477.7369995, -699.5389404, 694.0697021
4: -178.5707397, 327.4272461, -182.7255859, 335.4310303, -514.0017700, 510.1528320

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297506, upper bound: 5693.8297506
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297503, upper bound: 5693.8297503
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3295.7048340, 2594.3872070, -3292.7404785, 2583.1418457, -5878.8466797, 5887.1279297
1: -276.6621704, 193.5854492, -275.7785645, 192.8536835, -469.5157471, 469.3640137
2: -187.7403717, 328.0992737, -186.8198242, 326.6194153, -514.3597412, 514.9190674
3: -228.4466858, 480.1055908, -227.5098267, 477.7369995, -706.1837158, 707.6154175
4: -183.9619293, 336.9352417, -182.7255859, 335.4310303, -519.3929443, 519.6607666

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297503, upper bound: 5693.8297732
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297503, upper bound: 5693.8297734
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3423.1242676, 2698.3342285, -3200.8122559, 2492.0700684, -5915.1933594, 5899.1459961
1: -287.9113159, 201.2695770, -266.6164856, 187.2634583, -475.1747742, 467.8860474
2: -197.8938141, 341.8537292, -181.1727295, 315.7156372, -513.6094360, 523.0263062
3: -240.5560608, 498.4641113, -220.5318604, 462.8116150, -703.3676758, 718.9958496
4: -193.3300476, 350.6156921, -177.1924896, 324.4970703, -517.8271484, 527.8081665

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297215, upper bound: 5693.8297453
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297216, upper bound: 5693.8297597
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3521.3881836, 2776.0539551, -3200.8122559, 2492.0700684, -6013.4575195, 5976.8652344
1: -296.0671082, 206.9956665, -266.6164856, 187.2634583, -483.3305664, 473.6121521
2: -203.5337830, 351.8211060, -181.1727295, 315.7156372, -519.2493896, 532.9938354
3: -247.4381561, 512.8487549, -220.5318604, 462.8116150, -710.2497559, 733.3806152
4: -198.8710022, 360.8362122, -177.1924896, 324.4970703, -523.3680420, 538.0285645

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297215, upper bound: 5693.8297580
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297216, upper bound: 5693.8297622
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3422.1242676, 2702.7954102, -3328.9506836, 2597.5971680, -6019.7211914, 6031.7460938
1: -288.2008667, 201.2764587, -277.7698059, 194.9701691, -483.1709900, 479.0462646
2: -198.1415253, 342.2611084, -188.8735046, 328.8377686, -526.9793091, 531.1346436
3: -240.8829193, 498.7223511, -229.8262634, 481.5760193, -722.4588623, 728.5485840
4: -193.5966187, 350.9875183, -184.5046539, 337.8272095, -531.4238281, 535.4921265

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297718, upper bound: 5693.8297691
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297686, upper bound: 5693.8297654
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297675, upper bound: 5693.8297654
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3513.5354004, 2776.0678711, -3328.9506836, 2597.5971680, -6111.1323242, 6105.0185547
1: -295.8893738, 206.6088715, -277.7698059, 194.9701691, -490.8594666, 484.3786621
2: -203.3980560, 351.6396484, -188.8735046, 328.8377686, -532.2358398, 540.5130615
3: -247.3071136, 512.2778320, -229.8262634, 481.5760193, -728.8831177, 742.1041260
4: -198.7847595, 360.5822144, -184.5046539, 337.8272095, -536.6119385, 545.0866699

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297718, upper bound: 5693.8297679
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297686, upper bound: 5693.8297693
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297675, upper bound: 5693.8297690
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3452.5529785, 2725.0349121, -3181.5979004, 2478.7053223, -5931.2583008, 5906.6328125
1: -290.6806946, 203.1942139, -265.1635742, 186.1546936, -476.8352966, 468.3576965
2: -199.3946686, 345.0222778, -180.0768433, 313.9613953, -513.3559570, 525.0991211
3: -242.3578491, 503.0397644, -219.2167816, 460.2477722, -702.6055908, 722.2565308
4: -194.7559052, 353.8064880, -176.1553345, 322.6781921, -517.4340820, 529.9617920

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8295697, upper bound: 5693.8297056
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297466, upper bound: 5693.8297610
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3546.0996094, 2799.5058594, -3181.5979004, 2478.7053223, -6024.8046875, 5981.1030273
1: -298.4955444, 208.7005615, -265.1635742, 186.1546936, -484.6501770, 473.8640442
2: -204.7602844, 354.5985718, -180.0768433, 313.9613953, -518.7215576, 534.6754150
3: -248.8956451, 516.8376465, -219.2167816, 460.2477722, -709.1434326, 736.0544434
4: -200.0314636, 363.6328125, -176.1553345, 322.6781921, -522.7096558, 539.7881470

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8295697, upper bound: 5693.8297056
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297466, upper bound: 5693.8297638
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3453.3281250, 2730.2009277, -3308.7861328, 2584.0610352, -6037.3886719, 6038.9873047
1: -291.0779419, 203.3083496, -276.2791443, 193.8118896, -484.8898315, 479.5874329
2: -199.7337494, 345.5274353, -187.7561340, 327.0481873, -526.7818604, 533.2834473
3: -242.7936859, 503.5822449, -228.4906311, 478.9238586, -721.7175293, 732.0728149
4: -195.1296997, 354.2846680, -183.4465027, 335.9752197, -531.1049194, 537.7310791

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8296706, upper bound: 5693.8297535
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297679, upper bound: 5693.8297681
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3541.7651367, 2801.1098633, -3308.7861328, 2584.0610352, -6125.8247070, 6109.8959961
1: -298.5216675, 208.5278931, -276.2791443, 193.8118896, -492.3334961, 484.8070068
2: -204.7967987, 354.6605225, -187.7561340, 327.0481873, -531.8449707, 542.4165039
3: -248.9675446, 516.7344971, -228.4906311, 478.9238586, -727.8914185, 745.2250977
4: -200.1216431, 363.6117249, -183.4465027, 335.9752197, -536.0968628, 547.0581665

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8296706, upper bound: 5693.8297538
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297679, upper bound: 5693.8297687
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3423.1242676, 2698.3342285, -3300.6071777, 2568.0632324, -5991.1875000, 5998.9414062
1: -287.9113159, 201.2695770, -274.7120667, 193.0437622, -480.9549866, 475.9816284
2: -197.8938141, 341.8537292, -187.0105896, 325.6140137, -523.5078125, 528.8641968
3: -240.5560608, 498.4641113, -227.6125488, 477.1114502, -717.6674805, 726.0765381
4: -193.3300476, 350.6156921, -182.9010620, 334.6625366, -527.9925537, 533.5166626

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297216, upper bound: 5693.8297539
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297216, upper bound: 5693.8297647
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3521.3881836, 2776.0539551, -3300.6071777, 2568.0632324, -6089.4511719, 6076.6611328
1: -296.0671082, 206.9956665, -274.7120667, 193.0437622, -489.1108093, 481.7077332
2: -203.5337830, 351.8211060, -187.0105896, 325.6140137, -529.1478271, 538.8316650
3: -247.4381561, 512.8487549, -227.6125488, 477.1114502, -724.5496216, 740.4612427
4: -198.8710022, 360.8362122, -182.9010620, 334.6625366, -533.5335693, 543.7371826

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297217, upper bound: 5693.8297651
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297216, upper bound: 5693.8297656
time: 1.16 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3422.1242676, 2702.7954102, -3417.0583496, 2666.8093262, -6088.9335938, 6119.8535156
1: -288.2008667, 201.2764587, -285.0947876, 200.0697021, -488.2705383, 486.3712463
2: -198.1415253, 342.2611084, -194.0800476, 337.8158875, -535.9573975, 536.3411865
3: -240.8829193, 498.7223511, -236.1561127, 494.5251160, -735.4080200, 734.8784790
4: -193.5966187, 350.9875183, -189.6313934, 346.9880066, -540.5845947, 540.6188965

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297679, upper bound: 5693.8297671
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297678, upper bound: 5693.8297696
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3513.5354004, 2776.0678711, -3417.0583496, 2666.8093262, -6180.3447266, 6193.1254883
1: -295.8893738, 206.6088715, -285.0947876, 200.0697021, -495.9590454, 491.7036743
2: -203.3980560, 351.6396484, -194.0800476, 337.8158875, -541.2139282, 545.7196045
3: -247.3071136, 512.2778320, -236.1561127, 494.5251160, -741.8322144, 748.4339600
4: -198.7847595, 360.5822144, -189.6313934, 346.9880066, -545.7725830, 550.2135010

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297679, upper bound: 5693.8297683
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297678, upper bound: 5693.8297696
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3452.5529785, 2725.0349121, -3268.7287598, 2546.4526367, -5999.0058594, 5993.7636719
1: -290.6806946, 203.1942139, -272.3489380, 191.2155762, -481.8962402, 475.5430298
2: -199.3946686, 345.0222778, -185.1996765, 322.7571716, -522.1518555, 530.2219238
3: -242.3578491, 503.0397644, -225.4497528, 472.8846436, -715.2423706, 728.4895020
4: -194.7559052, 353.8064880, -181.1885529, 331.7024841, -526.4583740, 534.9950562

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297489, upper bound: 5693.8297648
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297186, upper bound: 5693.8297416
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3546.0996094, 2799.5058594, -3268.7287598, 2546.4526367, -6092.5522461, 6068.2338867
1: -298.4955444, 208.7005615, -272.3489380, 191.2155762, -489.7110901, 481.0494385
2: -204.7602844, 354.5985718, -185.1996765, 322.7571716, -527.5173340, 539.7982178
3: -248.8956451, 516.8376465, -225.4497528, 472.8846436, -721.7801514, 742.2874146
4: -200.0314636, 363.6328125, -181.1885529, 331.7024841, -531.7338867, 544.8213501

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297489, upper bound: 5693.8297651
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297186, upper bound: 5693.8297555
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3453.3281250, 2730.2009277, -3389.6784668, 2648.1728516, -6101.5009766, 6119.8789062
1: -291.0779419, 203.3083496, -283.0579834, 198.4990997, -489.5770264, 486.3662415
2: -199.7337494, 345.5274353, -192.5449371, 335.3660278, -535.0996094, 538.0722046
3: -242.7936859, 503.5822449, -234.3175659, 490.8702698, -733.6638184, 737.8996582
4: -195.1296997, 354.2846680, -188.1708221, 344.4447632, -539.5744629, 542.4555054

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297694, upper bound: 5693.8297705
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297704, upper bound: 5693.8297705
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3541.7651367, 2801.1098633, -3389.6784668, 2648.1728516, -6189.9375000, 6190.7880859
1: -298.5216675, 208.5278931, -283.0579834, 198.4990997, -497.0206909, 491.5858459
2: -204.7967987, 354.6605225, -192.5449371, 335.3660278, -540.1627197, 547.2052002
3: -248.9675446, 516.7344971, -234.3175659, 490.8702698, -739.8377686, 751.0520020
4: -200.1216431, 363.6117249, -188.1708221, 344.4447632, -544.5664062, 551.7825317

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297694, upper bound: 5693.8297712
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297704, upper bound: 5693.8297713
time: 0.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.57 seconds
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297433, upper bound: 5693.8297525
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297654, upper bound: 5693.8297688
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297433, upper bound: 5693.8297695
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297654, upper bound: 5693.8297727
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297360, upper bound: 5693.8297488
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297688
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297360, upper bound: 5693.8297671
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297728
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297367, upper bound: 5693.8297528
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297367, upper bound: 5693.8297581
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297367, upper bound: 5693.8297517
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297369, upper bound: 5693.8297746
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297503, upper bound: 5693.8297628
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297505, upper bound: 5693.8297706
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297505, upper bound: 5693.8297751
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297505, upper bound: 5693.8297769
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297368, upper bound: 5693.8297506
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297368, upper bound: 5693.8297503
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297368, upper bound: 5693.8297735
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297368, upper bound: 5693.8297735
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297506, upper bound: 5693.8297506
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297503, upper bound: 5693.8297503
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297503, upper bound: 5693.8297732
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297503, upper bound: 5693.8297734
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297215, upper bound: 5693.8297453
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297216, upper bound: 5693.8297597
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297215, upper bound: 5693.8297580
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297216, upper bound: 5693.8297622
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297686, upper bound: 5693.8297654
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297675, upper bound: 5693.8297654
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297686, upper bound: 5693.8297693
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297675, upper bound: 5693.8297690
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8295697, upper bound: 5693.8297056
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297466, upper bound: 5693.8297610
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8295697, upper bound: 5693.8297056
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297466, upper bound: 5693.8297638
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8296706, upper bound: 5693.8297535
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297679, upper bound: 5693.8297681
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8296706, upper bound: 5693.8297538
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297679, upper bound: 5693.8297687
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297216, upper bound: 5693.8297539
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297216, upper bound: 5693.8297647
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297217, upper bound: 5693.8297651
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297216, upper bound: 5693.8297656
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297679, upper bound: 5693.8297671
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297678, upper bound: 5693.8297696
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297679, upper bound: 5693.8297683
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297678, upper bound: 5693.8297696
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297489, upper bound: 5693.8297648
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297186, upper bound: 5693.8297416
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297489, upper bound: 5693.8297651
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297186, upper bound: 5693.8297555
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297694, upper bound: 5693.8297705
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297704, upper bound: 5693.8297705
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297694, upper bound: 5693.8297712
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.57
Output dim: 0, lower bound: -5693.8297704, upper bound: 5693.8297713

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3015.6596680, 2371.9321289, -3096.6042480, 2423.7744141, -5439.4335938, 5468.5361328
1: -253.2371979, 176.5710754, -259.0422058, 181.2160034, -434.4531555, 435.6131897
2: -171.0115356, 299.3089905, -175.3394623, 306.3741455, -477.3856812, 474.6484375
3: -208.3713684, 438.1641235, -213.5798798, 448.8952637, -657.2666016, 651.7437134
4: -167.5410309, 307.4345093, -171.6853943, 314.8195496, -482.3605652, 479.1198425

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297192, upper bound: 5693.8296505
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297388, upper bound: 5693.8297477
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3008.1113281, 2370.4157715, -3198.6035156, 2511.9204102, -5520.0317383, 5569.0185547
1: -252.9086151, 176.1971283, -268.2534790, 187.3858490, -440.2944336, 444.4505920
2: -170.7766724, 298.9758301, -181.4248199, 317.1923828, -487.9690552, 480.4006042
3: -208.0793457, 437.4748535, -221.0010986, 464.1204834, -672.1998291, 658.4759521
4: -167.3165436, 307.0135803, -177.4664001, 325.8106079, -493.1271362, 484.4799805

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297415, upper bound: 5693.8296885
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297607, upper bound: 5693.8297643
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3105.0327148, 2441.7121582, -3096.6042480, 2423.7744141, -5528.8071289, 5538.3164062
1: -260.6072083, 181.7746735, -259.0422058, 181.2160034, -441.8232117, 440.8167725
2: -176.3225403, 308.3625793, -175.3394623, 306.3741455, -482.6966858, 483.7020264
3: -214.8164520, 451.1150818, -213.5798798, 448.8952637, -663.7117310, 664.6949463
4: -172.7338257, 316.7129517, -171.6853943, 314.8195496, -487.5532532, 488.3983154

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297126, upper bound: 5693.8297669
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297120, upper bound: 5693.8297636
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3087.9150391, 2434.4375000, -3198.6035156, 2511.9204102, -5599.8354492, 5633.0410156
1: -259.6409302, 180.8637695, -268.2534790, 187.3858490, -447.0267639, 449.1172485
2: -175.5465698, 307.2256775, -181.4248199, 317.1923828, -492.7389526, 488.6504211
3: -213.9075623, 449.2587280, -221.0010986, 464.1204834, -678.0279541, 670.2598267
4: -172.0268250, 315.4552002, -177.4664001, 325.8106079, -497.8374023, 492.9216003

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297577, upper bound: 5693.8297691
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297563, upper bound: 5693.8297712
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2994.9816895, 2357.5554199, -3111.4833984, 2441.1555176, -5436.1362305, 5469.0390625
1: -251.6638336, 175.3860626, -260.8005066, 182.2313995, -433.8952332, 436.1865845
2: -169.8437500, 297.4277344, -176.0322876, 308.3531189, -478.1968689, 473.4599609
3: -206.9602203, 435.3807678, -214.4815216, 451.5268555, -658.4870605, 649.8623047
4: -166.4230194, 305.4705200, -172.4324799, 316.7261353, -483.1491699, 477.9029846

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297214, upper bound: 5693.8296498
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297317, upper bound: 5693.8297439
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2993.4738770, 2360.2836914, -3248.9179688, 2553.3801270, -5546.8540039, 5609.2016602
1: -251.8010712, 175.3710327, -272.6679993, 190.5388641, -442.3399353, 448.0390015
2: -169.9819641, 297.6770935, -184.1393127, 322.3106384, -492.2925110, 481.8164062
3: -207.1185150, 435.5352783, -224.2706299, 471.6904297, -678.8087769, 659.8059082
4: -166.5634766, 305.6457825, -180.1349335, 330.9239807, -497.4874573, 485.7806702

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297583, upper bound: 5693.8296907
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297681, upper bound: 5693.8297644
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3075.5605469, 2421.6535645, -3111.4833984, 2441.1555176, -5516.7148438, 5533.1367188
1: -258.4177246, 180.0883331, -260.8005066, 182.2313995, -440.6491089, 440.8888550
2: -174.6683350, 305.7177429, -176.0322876, 308.3531189, -483.0214539, 481.7499390
3: -212.8330841, 447.2228088, -214.4815216, 451.5268555, -664.3598633, 661.7043457
4: -171.1762695, 313.9580994, -172.4324799, 316.7261353, -487.9024048, 486.3905640

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297187, upper bound: 5693.8297627
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297185, upper bound: 5693.8297628
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3072.9555664, 2425.0615234, -3248.9179688, 2553.3801270, -5626.3344727, 5673.9794922
1: -258.5787048, 180.0696106, -272.6679993, 190.5388641, -449.1175537, 452.7376099
2: -174.7911072, 306.0111694, -184.1393127, 322.3106384, -497.1016846, 490.1504822
3: -212.9917450, 447.3568115, -224.2706299, 471.6904297, -684.6820679, 671.6274414
4: -171.3060913, 314.1651917, -180.1349335, 330.9239807, -502.2300720, 494.3001099

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297705, upper bound: 5693.8297691
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297704, upper bound: 5693.8297713
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2945.9621582, 2322.5600586, -3039.6013184, 2393.7299805, -5339.6918945, 5362.1611328
1: -247.8460999, 172.3127289, -255.4215393, 177.7558594, -425.6019592, 427.7342529
2: -166.8696442, 292.9187317, -172.2669983, 302.1781616, -469.0477905, 465.1857300
3: -203.4680939, 428.6735229, -210.0196533, 442.0452271, -645.5132446, 638.6931763
4: -163.6853485, 300.8114929, -168.9473572, 310.3593750, -474.0447388, 469.7586975

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297684, upper bound: 5693.8297331
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297713, upper bound: 5693.8297529
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2945.9621582, 2322.5600586, -3386.8078613, 2662.9218750, -5608.8823242, 5709.3681641
1: -247.8460999, 172.3127289, -284.1209412, 198.7053070, -446.5513306, 456.4336548
2: -166.8696442, 292.9187317, -193.9574585, 337.2680054, -504.1376343, 486.8761902
3: -203.4680939, 428.6735229, -236.0388336, 492.2661438, -695.7342529, 664.7123413
4: -163.6853485, 300.8114929, -189.5996246, 346.0918274, -509.7771606, 490.4111023

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.30 + 415.09 = 420.39 seconds
