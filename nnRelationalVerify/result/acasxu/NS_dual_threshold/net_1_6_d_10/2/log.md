## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.0038562720000000004


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551)
1: (-0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071)
2: (-0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934)
3: (-0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260)
4: (-0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.74 + 0.69 = 1.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0041916, upper bound: 0.0041916

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040912, upper bound: 0.0040844
time: 0.16 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040967, upper bound: 0.0040967
time: 0.17 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.41 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.41
Output dim: 0, lower bound: -0.0040912, upper bound: 0.0040844
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.41
Output dim: 0, lower bound: -0.0040967, upper bound: 0.0040967

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0145263, -0.0097033, -0.0142306, -0.0095089, -0.0050173, 0.0045273
1: -0.0195155, -0.0148325, -0.0193849, -0.0169080, -0.0026075, 0.0045523
2: -0.0204564, -0.0157830, -0.0199511, -0.0158257, -0.0046307, 0.0041681
3: -0.0200862, -0.0064727, -0.0188104, -0.0065534, -0.0135328, 0.0123377
4: -0.0189324, -0.0069033, -0.0184893, -0.0068395, -0.0120929, 0.0115859

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040796, upper bound: 0.0040796
time: 0.16 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040796, upper bound: 0.0040844
time: 0.16 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0140180, -0.0116259, -0.0142358, -0.0094807, -0.0045372, 0.0026099
1: -0.0192708, -0.0176558, -0.0193883, -0.0168813, -0.0023896, 0.0017325
2: -0.0196338, -0.0181951, -0.0199593, -0.0157658, -0.0038679, 0.0017641
3: -0.0180964, -0.0115294, -0.0188365, -0.0065106, -0.0115858, 0.0073072
4: -0.0181498, -0.0130073, -0.0184974, -0.0067645, -0.0113853, 0.0054901

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040844, upper bound: 0.0040912
time: 0.15 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040844, upper bound: 0.0040912
time: 0.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.06 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.06
Output dim: 0, lower bound: -0.0040796, upper bound: 0.0040796
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.06
Output dim: 0, lower bound: -0.0040796, upper bound: 0.0040844
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.06
Output dim: 0, lower bound: -0.0040844, upper bound: 0.0040912
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.06
Output dim: 0, lower bound: -0.0040844, upper bound: 0.0040912

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0145263, -0.0097033, -0.0145263, -0.0097042, -0.0048221, 0.0048229
1: -0.0195155, -0.0148325, -0.0195155, -0.0148374, -0.0046781, 0.0046829
2: -0.0204564, -0.0157830, -0.0204564, -0.0157962, -0.0046602, 0.0046734
3: -0.0200862, -0.0064727, -0.0200862, -0.0064803, -0.0136058, 0.0136135
4: -0.0189324, -0.0069033, -0.0189324, -0.0069200, -0.0120124, 0.0120291

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039299, upper bound: 0.0037010
time: 0.15 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040796, upper bound: 0.0040796
time: 0.17 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0145263, -0.0097033, -0.0140180, -0.0116259, -0.0029003, 0.0043146
1: -0.0195155, -0.0148325, -0.0192708, -0.0176558, -0.0018597, 0.0044383
2: -0.0204564, -0.0157830, -0.0196338, -0.0181951, -0.0022612, 0.0038508
3: -0.0200862, -0.0064727, -0.0180964, -0.0115294, -0.0085568, 0.0116237
4: -0.0189324, -0.0069033, -0.0181498, -0.0130073, -0.0059251, 0.0112465

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038568, upper bound: 0.0040565
time: 0.18 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038612, upper bound: 0.0038781
time: 0.15 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0140180, -0.0116259, -0.0145263, -0.0097033, -0.0043146, 0.0029003
1: -0.0192708, -0.0176558, -0.0195155, -0.0148325, -0.0044383, 0.0018597
2: -0.0196338, -0.0181951, -0.0204564, -0.0157830, -0.0038508, 0.0022612
3: -0.0180964, -0.0115294, -0.0200862, -0.0064727, -0.0116237, 0.0085568
4: -0.0181498, -0.0130073, -0.0189324, -0.0069033, -0.0112465, 0.0059251

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040565, upper bound: 0.0040263
time: 0.16 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038781, upper bound: 0.0040320
time: 0.19 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0140180, -0.0116259, -0.0140180, -0.0116259, -0.0023920, 0.0023920
1: -0.0192708, -0.0176558, -0.0192708, -0.0176558, -0.0016150, 0.0016150
2: -0.0196338, -0.0181951, -0.0196338, -0.0181951, -0.0014386, 0.0014386
3: -0.0180964, -0.0115294, -0.0180964, -0.0115294, -0.0065670, 0.0065670
4: -0.0181498, -0.0130073, -0.0181498, -0.0130073, -0.0051425, 0.0051425

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038544, upper bound: 0.0040633
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038781, upper bound: 0.0040320
time: 0.17 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.29 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.29
Output dim: 0, lower bound: -0.0039299, upper bound: 0.0037010
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.29
Output dim: 0, lower bound: -0.0040796, upper bound: 0.0040796
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.29
Output dim: 0, lower bound: -0.0038568, upper bound: 0.0040565
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.29
Output dim: 0, lower bound: -0.0038612, upper bound: 0.0038781
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 1.29
Output dim: 0, lower bound: -0.0040565, upper bound: 0.0040263
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 1.29
Output dim: 0, lower bound: -0.0038781, upper bound: 0.0040320
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.29
Output dim: 0, lower bound: -0.0038544, upper bound: 0.0040633
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.29
Output dim: 0, lower bound: -0.0038781, upper bound: 0.0040320

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0149168, -0.0090286, -0.0145263, -0.0097042, -0.0052127, 0.0054976
1: -0.0197052, -0.0119089, -0.0195155, -0.0148374, -0.0048678, 0.0076066
2: -0.0212128, -0.0145394, -0.0204564, -0.0157962, -0.0054167, 0.0059170
3: -0.0216718, -0.0058419, -0.0200862, -0.0064803, -0.0151915, 0.0142442
4: -0.0195063, -0.0059430, -0.0189324, -0.0069200, -0.0125863, 0.0129895

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0036771, upper bound: 0.0036771
time: 0.18 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0036771, upper bound: 0.0037010
time: 0.15 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0145260, -0.0097058, -0.0145263, -0.0097042, -0.0048218, 0.0048205
1: -0.0195152, -0.0148351, -0.0195155, -0.0148374, -0.0046778, 0.0046804
2: -0.0204560, -0.0157863, -0.0204564, -0.0157962, -0.0046598, 0.0046701
3: -0.0200844, -0.0064782, -0.0200862, -0.0064803, -0.0136041, 0.0136079
4: -0.0189318, -0.0069102, -0.0189324, -0.0069200, -0.0120118, 0.0120223

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038568, upper bound: 0.0040506
time: 0.17 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038611, upper bound: 0.0038612
time: 0.15 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0141037, -0.0107954, -0.0140180, -0.0116259, -0.0024778, 0.0032226
1: -0.0193187, -0.0172946, -0.0192708, -0.0176558, -0.0016629, 0.0019762
2: -0.0197876, -0.0172505, -0.0196338, -0.0181951, -0.0015925, 0.0023833
3: -0.0183508, -0.0089160, -0.0180964, -0.0115294, -0.0068214, 0.0091804
4: -0.0182811, -0.0097315, -0.0181498, -0.0130073, -0.0052737, 0.0084184

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040278, upper bound: 0.0038544
time: 0.16 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040278, upper bound: 0.0038781
time: 0.17 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0144098, -0.0101722, -0.0140180, -0.0116259, -0.0027838, 0.0038458
1: -0.0194430, -0.0156642, -0.0192708, -0.0176558, -0.0017872, 0.0036066
2: -0.0203044, -0.0168291, -0.0196338, -0.0181951, -0.0021093, 0.0028047
3: -0.0194587, -0.0072416, -0.0180964, -0.0115294, -0.0079293, 0.0108548
4: -0.0187139, -0.0081810, -0.0181498, -0.0130073, -0.0057066, 0.0099688

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040312, upper bound: 0.0038544
time: 0.17 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040312, upper bound: 0.0038781
time: 0.17 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0140180, -0.0116259, -0.0141037, -0.0107954, -0.0032226, 0.0024778
1: -0.0192708, -0.0176558, -0.0193187, -0.0172946, -0.0019762, 0.0016629
2: -0.0196338, -0.0181951, -0.0197876, -0.0172505, -0.0023833, 0.0015925
3: -0.0180964, -0.0115294, -0.0183508, -0.0089160, -0.0091804, 0.0068214
4: -0.0181498, -0.0130073, -0.0182811, -0.0097315, -0.0084184, 0.0052737

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038544, upper bound: 0.0040278
time: 0.19 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038544, upper bound: 0.0040287
time: 0.16 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0140180, -0.0116259, -0.0144098, -0.0101722, -0.0038458, 0.0027838
1: -0.0192708, -0.0176558, -0.0194430, -0.0156642, -0.0036066, 0.0017872
2: -0.0196338, -0.0181951, -0.0203044, -0.0168291, -0.0028047, 0.0021093
3: -0.0180964, -0.0115294, -0.0194587, -0.0072416, -0.0108548, 0.0079293
4: -0.0181498, -0.0130073, -0.0187139, -0.0081810, -0.0099688, 0.0057066

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038544, upper bound: 0.0040312
time: 0.16 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038544, upper bound: 0.0040320
time: 0.17 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0137114, -0.0122631, -0.0140180, -0.0116259, -0.0020854, 0.0017548
1: -0.0191517, -0.0181855, -0.0192708, -0.0176558, -0.0014959, 0.0010853
2: -0.0192194, -0.0185801, -0.0196338, -0.0181951, -0.0010242, 0.0010537
3: -0.0173103, -0.0128427, -0.0180964, -0.0115294, -0.0057809, 0.0052537
4: -0.0176720, -0.0144565, -0.0181498, -0.0130073, -0.0046647, 0.0036933

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039874, upper bound: 0.0040255
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039874, upper bound: 0.0040453
time: 0.19 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0139039, -0.0117751, -0.0140180, -0.0116259, -0.0022780, 0.0022428
1: -0.0192256, -0.0178401, -0.0192708, -0.0176558, -0.0015698, 0.0014307
2: -0.0194921, -0.0184853, -0.0196338, -0.0181951, -0.0012970, 0.0011485
3: -0.0177658, -0.0119961, -0.0180964, -0.0115294, -0.0062364, 0.0061003
4: -0.0179563, -0.0137371, -0.0181498, -0.0130073, -0.0049490, 0.0044127

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040478, upper bound: 0.0040263
time: 0.17 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040478, upper bound: 0.0040504
time: 0.18 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.11 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0036771, upper bound: 0.0036771
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0036771, upper bound: 0.0037010
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0038568, upper bound: 0.0040506
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0038611, upper bound: 0.0038612
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0040278, upper bound: 0.0038544
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0040278, upper bound: 0.0038781
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0040312, upper bound: 0.0038544
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0040312, upper bound: 0.0038781
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0038544, upper bound: 0.0040278
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0038544, upper bound: 0.0040287
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0038544, upper bound: 0.0040312
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0038544, upper bound: 0.0040320
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0039874, upper bound: 0.0040255
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0039874, upper bound: 0.0040453
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0040478, upper bound: 0.0040263
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.11
Output dim: 0, lower bound: -0.0040478, upper bound: 0.0040504

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0141034, -0.0107976, -0.0145263, -0.0097042, -0.0043992, 0.0037287
1: -0.0193185, -0.0172954, -0.0195155, -0.0148374, -0.0044811, 0.0022201
2: -0.0197871, -0.0172526, -0.0204564, -0.0157962, -0.0039909, 0.0032038
3: -0.0183496, -0.0089213, -0.0200862, -0.0064803, -0.0118692, 0.0111649
4: -0.0182804, -0.0097376, -0.0189324, -0.0069200, -0.0113604, 0.0091949

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038568, upper bound: 0.0038568
time: 0.15 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038568, upper bound: 0.0038612
time: 0.17 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0144096, -0.0101740, -0.0145263, -0.0097042, -0.0047054, 0.0043523
1: -0.0194428, -0.0156652, -0.0195155, -0.0148374, -0.0046053, 0.0038503
2: -0.0203042, -0.0168313, -0.0204564, -0.0157962, -0.0045080, 0.0036250
3: -0.0194576, -0.0072461, -0.0200862, -0.0064803, -0.0129773, 0.0128401
4: -0.0187134, -0.0081862, -0.0189324, -0.0069200, -0.0117934, 0.0107462

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038611, upper bound: 0.0038568
time: 0.18 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038611, upper bound: 0.0038611
time: 0.17 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0141037, -0.0107954, -0.0137114, -0.0122631, -0.0018406, 0.0029160
1: -0.0193187, -0.0172946, -0.0191517, -0.0181855, -0.0011331, 0.0018570
2: -0.0197876, -0.0172505, -0.0192194, -0.0185801, -0.0012075, 0.0019689
3: -0.0183508, -0.0089160, -0.0173103, -0.0128427, -0.0055081, 0.0083943
4: -0.0182811, -0.0097315, -0.0176720, -0.0144565, -0.0038246, 0.0079405

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0141037, -0.0107954, -0.0139039, -0.0117751, -0.0023286, 0.0031085
1: -0.0193187, -0.0172946, -0.0192256, -0.0178401, -0.0014785, 0.0019310
2: -0.0197876, -0.0172505, -0.0194921, -0.0184853, -0.0013023, 0.0022416
3: -0.0183508, -0.0089160, -0.0177658, -0.0119961, -0.0063547, 0.0088498
4: -0.0182811, -0.0097315, -0.0179563, -0.0137371, -0.0045439, 0.0082248

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0144098, -0.0101722, -0.0137114, -0.0122631, -0.0021467, 0.0035392
1: -0.0194430, -0.0156642, -0.0191517, -0.0181855, -0.0012574, 0.0034874
2: -0.0203044, -0.0168291, -0.0192194, -0.0185801, -0.0017243, 0.0023903
3: -0.0194587, -0.0072416, -0.0173103, -0.0128427, -0.0066160, 0.0100687
4: -0.0187139, -0.0081810, -0.0176720, -0.0144565, -0.0042574, 0.0094910

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0144098, -0.0101722, -0.0139039, -0.0117751, -0.0026346, 0.0037317
1: -0.0194430, -0.0156642, -0.0192256, -0.0178401, -0.0016028, 0.0035614
2: -0.0203044, -0.0168291, -0.0194921, -0.0184853, -0.0018191, 0.0026630
3: -0.0194587, -0.0072416, -0.0177658, -0.0119961, -0.0074626, 0.0105242
4: -0.0187139, -0.0081810, -0.0179563, -0.0137371, -0.0049768, 0.0097753

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0137114, -0.0122631, -0.0141037, -0.0107954, -0.0029160, 0.0018406
1: -0.0191517, -0.0181855, -0.0193187, -0.0172946, -0.0018570, 0.0011331
2: -0.0192194, -0.0185801, -0.0197876, -0.0172505, -0.0019689, 0.0012075
3: -0.0173103, -0.0128427, -0.0183508, -0.0089160, -0.0083943, 0.0055081
4: -0.0176720, -0.0144565, -0.0182811, -0.0097315, -0.0079405, 0.0038246

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0139039, -0.0117751, -0.0141037, -0.0107954, -0.0031085, 0.0023286
1: -0.0192256, -0.0178401, -0.0193187, -0.0172946, -0.0019310, 0.0014785
2: -0.0194921, -0.0184853, -0.0197876, -0.0172505, -0.0022416, 0.0013023
3: -0.0177658, -0.0119961, -0.0183508, -0.0089160, -0.0088498, 0.0063547
4: -0.0179563, -0.0137371, -0.0182811, -0.0097315, -0.0082248, 0.0045439

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0137114, -0.0122631, -0.0144098, -0.0101722, -0.0035392, 0.0021467
1: -0.0191517, -0.0181855, -0.0194430, -0.0156642, -0.0034874, 0.0012574
2: -0.0192194, -0.0185801, -0.0203044, -0.0168291, -0.0023903, 0.0017243
3: -0.0173103, -0.0128427, -0.0194587, -0.0072416, -0.0100687, 0.0066160
4: -0.0176720, -0.0144565, -0.0187139, -0.0081810, -0.0094910, 0.0042574

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0139039, -0.0117751, -0.0144098, -0.0101722, -0.0037317, 0.0026346
1: -0.0192256, -0.0178401, -0.0194430, -0.0156642, -0.0035614, 0.0016028
2: -0.0194921, -0.0184853, -0.0203044, -0.0168291, -0.0026630, 0.0018191
3: -0.0177658, -0.0119961, -0.0194587, -0.0072416, -0.0105242, 0.0074626
4: -0.0179563, -0.0137371, -0.0187139, -0.0081810, -0.0097753, 0.0049768

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0137114, -0.0122631, -0.0137114, -0.0122631, -0.0014483, 0.0014483
1: -0.0191517, -0.0181855, -0.0191517, -0.0181855, -0.0009661, 0.0009661
2: -0.0192194, -0.0185801, -0.0192194, -0.0185801, -0.0006393, 0.0006393
3: -0.0173103, -0.0128427, -0.0173103, -0.0128427, -0.0044676, 0.0044676
4: -0.0176720, -0.0144565, -0.0176720, -0.0144565, -0.0032155, 0.0032155

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0137114, -0.0122631, -0.0139039, -0.0117751, -0.0019363, 0.0016408
1: -0.0191517, -0.0181855, -0.0192256, -0.0178401, -0.0013115, 0.0010401
2: -0.0192194, -0.0185801, -0.0194921, -0.0184853, -0.0007341, 0.0009120
3: -0.0173103, -0.0128427, -0.0177658, -0.0119961, -0.0053142, 0.0049231
4: -0.0176720, -0.0144565, -0.0179563, -0.0137371, -0.0039349, 0.0034998

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0139039, -0.0117751, -0.0137114, -0.0122631, -0.0016408, 0.0019363
1: -0.0192256, -0.0178401, -0.0191517, -0.0181855, -0.0010401, 0.0013115
2: -0.0194921, -0.0184853, -0.0192194, -0.0185801, -0.0009120, 0.0007341
3: -0.0177658, -0.0119961, -0.0173103, -0.0128427, -0.0049231, 0.0053142
4: -0.0179563, -0.0137371, -0.0176720, -0.0144565, -0.0034998, 0.0039349

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0139039, -0.0117751, -0.0139039, -0.0117751, -0.0021288, 0.0021288
1: -0.0192256, -0.0178401, -0.0192256, -0.0178401, -0.0013855, 0.0013855
2: -0.0194921, -0.0184853, -0.0194921, -0.0184853, -0.0010068, 0.0010068
3: -0.0177658, -0.0119961, -0.0177658, -0.0119961, -0.0057697, 0.0057697
4: -0.0179563, -0.0137371, -0.0179563, -0.0137371, -0.0042191, 0.0042191

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.43 + 28.40 = 29.82 seconds
