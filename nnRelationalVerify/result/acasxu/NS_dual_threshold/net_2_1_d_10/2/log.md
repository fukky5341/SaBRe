## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 2.5152854478


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671)
1: (-1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297)
2: (-1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728)
3: (-1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408)
4: (-2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.15 = 2.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.5203261, upper bound: 2.5203261

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5201938, upper bound: 2.5194575
time: 0.41 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5203102, upper bound: 2.5203102
time: 0.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.90 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.90
Output dim: 0, lower bound: -2.5201938, upper bound: 2.5194575
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.90
Output dim: 0, lower bound: -2.5203102, upper bound: 2.5203102

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.6754932, 1.3466308, -1.2401047, 1.8555624, -2.5310552, 2.5867350
1: -0.7610564, 1.3813244, -1.4851047, 2.0027251, -2.7637815, 2.8664291
2: -0.8425393, 1.3817630, -1.4969569, 1.8678157, -2.7103548, 2.8787198
3: -0.9229488, 1.6725746, -1.7819047, 2.4911361, -3.4140849, 3.4544792
4: -1.3650292, 1.9489899, -2.2745841, 2.7287276, -4.0937567, 4.2235737

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5194265, upper bound: 2.5194265
time: 0.43 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5194265, upper bound: 2.5194575
time: 0.35 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7.0239887, 6.3770490, -1.2156670, 1.8306491, -8.6132889, 7.5927148
1: -9.7314625, 6.2030640, -1.4499383, 1.9785936, -11.3731203, 7.6530023
2: -8.2329721, 6.1651368, -1.4698651, 1.8486125, -9.7795572, 7.6350012
3: -9.6892891, 9.5362921, -1.7465651, 2.4514024, -11.8599110, 11.2159281
4: -10.0884438, 8.2303915, -2.2383173, 2.6959260, -12.5598803, 10.4687090

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176988, upper bound: 2.5191698
time: 0.46 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176614, upper bound: 2.5177285
time: 0.38 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.16 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -2.5194265, upper bound: 2.5194265
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -2.5194265, upper bound: 2.5194575
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -2.5176988, upper bound: 2.5191698
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -2.5176614, upper bound: 2.5177285

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.6754932, 1.3466308, -0.6754932, 1.3466308, -2.0221241, 2.0221241
1: -0.7610564, 1.3813244, -0.7610564, 1.3813244, -2.1423807, 2.1423807
2: -0.8425393, 1.3817630, -0.8425393, 1.3817630, -2.2243023, 2.2243023
3: -0.9229488, 1.6725746, -0.9229488, 1.6725746, -2.5955234, 2.5955234
4: -1.3650292, 1.9489899, -1.3650292, 1.9489899, -3.3140187, 3.3140192

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176988, upper bound: 2.5171454
time: 0.35 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176614, upper bound: 2.5181351
time: 0.39 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.6754932, 1.3466308, -5.6146450, 5.0941525, -5.7696457, 6.8465161
1: -0.7610564, 1.3813244, -7.6962872, 4.9879494, -5.7490048, 8.9284763
2: -0.8425393, 1.3817630, -6.5841451, 4.9601183, -5.8026576, 7.8192558
3: -0.9229488, 1.6725746, -7.7310028, 7.5750513, -8.4979992, 9.2390118
4: -1.3650292, 1.9489899, -8.1130047, 6.6544933, -8.0195227, 9.9792252

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171454, upper bound: 2.5181707
time: 0.36 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157044, upper bound: 2.5182023
time: 0.37 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -6.8317237, 6.1549883, -1.2156670, 1.8306491, -8.3768749, 7.3706551
1: -9.5345364, 5.9877911, -1.4499383, 1.9785936, -11.0978041, 7.4377294
2: -8.0127945, 5.9601860, -1.4698651, 1.8486125, -9.4994431, 7.4300499
3: -9.4746609, 9.2591133, -1.7465651, 2.4514024, -11.5927916, 10.8748436
4: -9.7967768, 7.9399056, -2.2383173, 2.6959260, -12.2030201, 10.1782227

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176988, upper bound: 2.5184178
time: 0.36 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5178146, upper bound: 2.5191698
time: 0.44 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -8.6723347, 7.5149980, -1.0979275, 1.7306205, -9.5498505, 8.4613600
1: -12.1343040, 7.3268261, -1.2880454, 1.8538647, -12.8064575, 8.4770679
2: -10.1676092, 7.2262888, -1.3343790, 1.7548029, -10.8919601, 8.4667616
3: -11.9992504, 11.4338579, -1.5765719, 2.2678235, -13.2599020, 12.4183826
4: -12.2650061, 9.4761868, -2.0556393, 2.5396948, -13.7874699, 11.4841080

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5177285, upper bound: 2.5176613
time: 0.43 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5177285, upper bound: 2.5176634
time: 0.43 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.19 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -2.5176988, upper bound: 2.5171454
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -2.5176614, upper bound: 2.5181351
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -2.5171454, upper bound: 2.5181707
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -2.5157044, upper bound: 2.5182023
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -2.5176988, upper bound: 2.5184178
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -2.5178146, upper bound: 2.5191698
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -2.5177285, upper bound: 2.5176613
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.19
Output dim: 0, lower bound: -2.5177285, upper bound: 2.5176634

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4422624, 1.2128479, -0.6754932, 1.3466308, -1.7888932, 1.8883404
1: -0.4775845, 1.2234946, -0.7610564, 1.3813244, -1.8589088, 1.9845511
2: -0.5712293, 1.2429589, -0.8425393, 1.3817630, -1.9529923, 2.0854976
3: -0.6305043, 1.4943078, -0.9229488, 1.6725746, -2.3030784, 2.4172566
4: -1.0105994, 1.7292284, -1.3650292, 1.9489899, -2.9595892, 3.0942566

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5163191, upper bound: 2.5163191
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5163191, upper bound: 2.5165288
time: 0.39 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.7761545, 2.0858898, -0.5813313, 1.2901593, -4.0598140, 2.6672211
1: -3.5152841, 2.2868772, -0.6484846, 1.2993938, -4.8146777, 2.9353619
2: -3.2617290, 2.2376070, -0.7322218, 1.3254614, -4.5583735, 2.9698286
3: -3.8889868, 2.9340467, -0.7853723, 1.5810685, -5.4449573, 3.7194190
4: -4.1486120, 3.0661988, -1.2051851, 1.8473226, -5.9959331, 4.2713833

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5181276, upper bound: 2.5181351
time: 0.42 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5181276, upper bound: 2.5181351
time: 0.39 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.6754932, 1.3466308, -5.4223442, 4.8656864, -5.5411797, 6.6222854
1: -0.7610564, 1.3813244, -7.4714098, 4.7590437, -5.5201001, 8.6534920
2: -0.8425393, 1.3817630, -6.3586473, 4.7498059, -5.5923452, 7.5501757
3: -0.9229488, 1.6725746, -7.4818916, 7.2867489, -8.1869955, 8.9604530
4: -1.3650292, 1.9489899, -7.8199315, 6.3481336, -7.7131624, 9.6333427

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5167223, upper bound: 2.5156954
time: 0.34 seconds

## Relational analysis of NS_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5167223, upper bound: 2.5181707
time: 0.47 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.5813313, 1.2901593, -7.2269616, 6.1518574, -6.6469636, 7.8701968
1: -0.6484846, 1.2993938, -10.0624990, 6.0498457, -6.6158299, 10.4864464
2: -0.7322218, 1.3254614, -8.4758892, 5.9668713, -6.6471634, 9.0183640
3: -0.7853723, 1.5810685, -9.9799557, 9.4115915, -9.7646160, 10.7848320
4: -1.2051851, 1.8473226, -10.2488499, 7.8198605, -9.0091438, 11.3221750

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171882, upper bound: 2.5181948
time: 0.45 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176613, upper bound: 2.5182023
time: 0.42 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -5.4223442, 4.8656864, -0.6754932, 1.3466308, -6.6222858, 5.5411797
1: -7.4714098, 4.7590437, -0.7610564, 1.3813244, -8.6534929, 5.5201001
2: -6.3586473, 4.7498059, -0.8425393, 1.3817630, -7.5501747, 5.5923452
3: -7.4818916, 7.2867489, -0.9229488, 1.6725746, -8.9604511, 8.1869946
4: -7.8199315, 6.3481336, -1.3650292, 1.9489899, -9.6333427, 7.7131624

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5156954, upper bound: 2.5167223
time: 0.41 seconds

## Relational analysis of NS_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5178146, upper bound: 2.5184178
time: 0.44 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -6.8317237, 6.1549883, -7.0239887, 5.4972200, -11.8993034, 12.7636528
1: -9.5345364, 5.9877911, -9.7314625, 5.7922802, -14.7868433, 15.2189102
2: -8.0127945, 5.9601860, -8.2329721, 5.4104624, -12.9344292, 13.7302160
3: -9.4746609, 9.2591133, -9.6892891, 8.9427605, -17.7707443, 18.2989902
4: -9.7967768, 7.9399056, -10.0884438, 7.2663670, -16.6237526, 17.6008739

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5156954, upper bound: 2.5183199
time: 0.37 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5178146, upper bound: 2.5191698
time: 0.42 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -7.2269616, 6.1518574, -0.5813313, 1.2901593, -7.8701973, 6.6469636
1: -10.0624990, 6.0498457, -0.6484846, 1.2993938, -10.4864483, 6.6158299
2: -8.4758892, 5.9668713, -0.7322218, 1.3254614, -9.0183630, 6.6471639
3: -9.9799557, 9.4115915, -0.7853723, 1.5810685, -10.7848320, 9.7646160
4: -10.2488499, 7.8198605, -1.2051851, 1.8473226, -11.3221760, 9.0091448

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174542, upper bound: 2.5176613
time: 0.43 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5177285, upper bound: 2.5176613
time: 0.50 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -8.6723347, 7.5149980, -6.9182963, 5.4134960, -13.0818834, 13.8194675
1: -12.1343040, 7.3268261, -9.6091242, 5.6988430, -16.5152969, 16.1945972
2: -10.1676092, 7.2262888, -8.1107140, 5.3302689, -14.3340206, 14.7079191
3: -11.9992504, 11.4338579, -9.5505629, 8.8136482, -19.4883480, 19.8748875
4: -12.2650061, 9.4761868, -9.9370432, 7.1514578, -18.2386169, 18.8061256

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174603, upper bound: 2.5176634
time: 0.44 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5177285, upper bound: 2.5176634
time: 0.44 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.78 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5163191, upper bound: 2.5163191
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5163191, upper bound: 2.5165288
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5181276, upper bound: 2.5181351
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5181276, upper bound: 2.5181351
NS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5167223, upper bound: 2.5156954
NS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5167223, upper bound: 2.5181707
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5171882, upper bound: 2.5181948
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5176613, upper bound: 2.5182023
NS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5156954, upper bound: 2.5167223
NS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5178146, upper bound: 2.5184178
NS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5156954, upper bound: 2.5183199
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5178146, upper bound: 2.5191698
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5174542, upper bound: 2.5176613
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5177285, upper bound: 2.5176613
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5174603, upper bound: 2.5176634
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 0, lower bound: -2.5177285, upper bound: 2.5176634

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.4422624, 1.2128479, -0.4422624, 1.2128479, -1.6551099, 1.6551100
1: -0.4775845, 1.2234946, -0.4775845, 1.2234946, -1.7010791, 1.7010789
2: -0.5712293, 1.2429589, -0.5712293, 1.2429589, -1.8141878, 1.8141880
3: -0.6305043, 1.4943078, -0.6305043, 1.4943078, -2.1248121, 2.1248121
4: -1.0105994, 1.7292284, -1.0105994, 1.7292284, -2.7398276, 2.7398276

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5137103, upper bound: 2.5129936
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5149973, upper bound: 2.5155751
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.4422624, 1.2128479, -2.7761545, 2.0858898, -2.5281522, 3.9660876
1: -0.4775845, 1.2234946, -3.5152841, 2.2868772, -2.7644615, 4.7168760
2: -0.5712293, 1.2429589, -3.2617290, 2.2376070, -2.8088362, 4.4667535
3: -0.6305043, 1.4943078, -3.8889868, 2.9340467, -3.5645509, 5.3455725
4: -1.0105994, 1.7292284, -4.1486120, 3.0661988, -4.0767980, 5.8574972

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5126425, upper bound: 2.5171454
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5087777, upper bound: 2.5126050
time: 0.35 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.7761545, 2.0858898, -0.5327253, 1.2670929, -4.0363030, 2.6186152
1: -3.5152841, 2.2868772, -0.5889323, 1.2715299, -4.7868137, 2.8758094
2: -3.2617290, 2.2376070, -0.6767374, 1.3011088, -4.5341396, 2.9143443
3: -3.8889868, 2.9340467, -0.7285765, 1.5475914, -5.4113350, 3.6626232
4: -4.1486120, 3.0661988, -1.1370524, 1.8091310, -5.9577422, 4.2032509

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5181276, upper bound: 2.5181275
time: 0.41 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143002, upper bound: 2.5126050
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.7761545, 2.0858898, -0.6121423, 1.3185565, -4.0888996, 2.6980321
1: -3.5152841, 2.2868772, -0.6785367, 1.3333414, -4.8486252, 2.9654136
2: -3.2617290, 2.2376070, -0.7678517, 1.3544348, -4.5884871, 3.0054588
3: -3.8889868, 2.9340467, -0.8427736, 1.6363766, -5.4963212, 3.7768204
4: -4.1486120, 3.0661988, -1.2806345, 1.8991282, -6.0477390, 4.3468332

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5181352, upper bound: 2.5181275
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5181352, upper bound: 2.5181351
time: 0.41 seconds

## BFS NS instance: NS_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.5434484, 1.2865694, -3.7818506, 3.2864940, -3.8299410, 4.9810772
1: -0.5922467, 1.3072410, -5.1267018, 3.2934995, -3.8857462, 6.3287392
2: -0.6897216, 1.3179048, -4.4467726, 3.2573531, -3.9470716, 5.6448374
3: -0.7491769, 1.5855213, -5.2075033, 4.9574633, -5.7066402, 6.6680632
4: -1.1828482, 1.8542707, -5.5099936, 4.3841066, -5.5669546, 7.3232555

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5167223, upper bound: 2.5153838
time: 0.38 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5167223, upper bound: 2.5156954
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.6754932, 1.3466308, -4.2336016, 3.8432238, -4.5187168, 5.5458946
1: -0.7610564, 1.3813244, -5.7718782, 3.7863760, -4.5474324, 7.1108971
2: -0.8425393, 1.3817630, -4.9727058, 3.7797599, -4.6222987, 6.3047476
3: -0.9229488, 1.6725746, -5.8492413, 5.7042847, -6.6272335, 7.4579806
4: -1.3650292, 1.9489899, -6.1542435, 5.0755715, -6.4406009, 8.1032333

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5179243, upper bound: 2.5180880
time: 0.48 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180966
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.5327253, 1.2670929, -7.2161198, 6.1416631, -6.5854478, 7.8376064
1: -0.5889323, 1.2715299, -10.0469379, 6.0403008, -6.5428314, 10.4449863
2: -0.6767374, 1.3011088, -8.4631863, 5.9574442, -6.5792580, 8.9834938
3: -0.7285765, 1.5475914, -9.9647789, 9.3964367, -9.6896887, 10.7381620
4: -1.1370524, 1.8091310, -10.2337093, 7.8074903, -8.9234772, 11.2699614

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5181942
time: 0.41 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5181948
time: 0.38 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.6121423, 1.3185565, -7.2168169, 6.1423159, -6.6745563, 7.8907857
1: -0.6785367, 1.3333414, -10.0479355, 6.0409126, -6.6434026, 10.5095749
2: -0.7678517, 1.3544348, -8.4640007, 5.9580498, -6.6817102, 9.0385246
3: -0.8427736, 1.6363766, -9.9657536, 9.3974075, -9.8058233, 10.8239851
4: -1.2806345, 1.8991282, -10.2346821, 7.8082838, -9.0662251, 11.3626032

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5182018
time: 0.47 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5182023
time: 0.41 seconds

## BFS NS instance: NS_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.7818506, 3.2864940, -0.5434484, 1.2865694, -4.9810767, 3.8299415
1: -5.1267018, 3.2934995, -0.5922467, 1.3072410, -6.3287392, 3.8857462
2: -4.4467726, 3.2573531, -0.6897216, 1.3179048, -5.6448379, 3.9470723
3: -5.2075033, 4.9574633, -0.7491769, 1.5855213, -6.6680636, 5.7066402
4: -5.5099936, 4.3841066, -1.1828482, 1.8542707, -7.3232555, 5.5669546

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A1_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153838, upper bound: 2.5167223
time: 0.41 seconds

## Relational analysis of NS_A2_A1_B1_A1_B2

### Relational analysis result of NS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153838, upper bound: 2.5167223
time: 0.36 seconds

## BFS NS instance: NS_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.2336016, 3.8432238, -0.6754932, 1.3466308, -5.5458941, 4.5187163
1: -5.7718782, 3.7863760, -0.7610564, 1.3813244, -7.1108966, 4.5474324
2: -4.9727058, 3.7797599, -0.8425393, 1.3817630, -6.3047476, 4.6222992
3: -5.8492413, 5.7042847, -0.9229488, 1.6725746, -7.4579792, 6.6272326
4: -6.1542435, 5.0755715, -1.3650292, 1.9489899, -8.1032333, 6.4406009

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180880, upper bound: 2.5179243
time: 0.46 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180966, upper bound: 2.5176983
time: 0.42 seconds

## BFS NS instance: NS_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.6932468, 5.0561733, -6.9166260, 5.4123621, -10.6588306, 11.5766268
1: -7.8776979, 4.9461617, -9.5792789, 5.7006278, -13.0415182, 14.0403671
2: -6.6848583, 4.9140401, -8.1078739, 5.3281469, -11.5097733, 12.5687542
3: -7.8654175, 7.6193733, -9.5416346, 8.8018169, -15.9950037, 16.5323391
4: -8.1991749, 6.5510368, -9.9360313, 7.1549931, -14.9068508, 16.0704021

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A1_B2_A1_B1

### Relational analysis result of NS_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5152688, upper bound: 2.5183166
time: 0.37 seconds

## Relational analysis of NS_A2_A1_B2_A1_B2

### Relational analysis result of NS_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5152506, upper bound: 2.5183199
time: 0.43 seconds

## BFS NS instance: NS_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.9547625, 5.4191532, -7.0239887, 5.4972200, -11.1315899, 12.0578537
1: -8.2864189, 5.2587056, -9.7314625, 5.7922802, -13.6924992, 14.5277472
2: -6.9889183, 5.2616816, -8.2329721, 5.4104624, -12.0492325, 13.0590448
3: -8.2509079, 8.0973358, -9.6892891, 8.9427605, -16.6760082, 17.2161713
4: -8.5724125, 7.0173569, -10.0884438, 7.2663670, -15.5436230, 16.7085381

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A1_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174864, upper bound: 2.5191666
time: 0.42 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174864, upper bound: 2.5191698
time: 0.57 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -7.2161198, 6.1416631, -0.5327253, 1.2670929, -7.8376069, 6.5854468
1: -10.0469379, 6.0403008, -0.5889323, 1.2715299, -10.4449873, 6.5428309
2: -8.4631863, 5.9574442, -0.6767374, 1.3011088, -8.9834929, 6.5792584
3: -9.9647789, 9.3964367, -0.7285765, 1.5475914, -10.7381620, 9.6896877
4: -10.2337093, 7.8074903, -1.1370524, 1.8091310, -11.2699623, 8.9234772

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5181942, upper bound: 2.5173931
time: 0.48 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5181942, upper bound: 2.5176613
time: 0.41 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -7.2168169, 6.1423159, -0.6121423, 1.3185565, -7.8907857, 6.6745563
1: -10.0479355, 6.0409126, -0.6785367, 1.3333414, -10.5095758, 6.6434026
2: -8.4640007, 5.9580498, -0.7678517, 1.3544348, -9.0385227, 6.6817098
3: -9.9657536, 9.3974075, -0.8427736, 1.6363766, -10.8239851, 9.8058233
4: -10.2346821, 7.8082838, -1.2806345, 1.8991282, -11.3626032, 9.0662260

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180966, upper bound: 2.5173931
time: 0.48 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180966, upper bound: 2.5171882
time: 0.46 seconds

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -8.6723347, 7.5149980, -6.8907700, 5.3909569, -13.0584593, 13.7922554
1: -12.1343040, 7.3268261, -9.5755825, 5.6734295, -16.4882317, 16.1602211
2: -10.1676092, 7.2262888, -8.0785522, 5.3082790, -14.3114614, 14.6761017
3: -11.9992504, 11.4338579, -9.5144176, 8.7780876, -19.4516907, 19.8391914
4: -12.2650061, 9.4761868, -9.8975525, 7.1216726, -18.2069340, 18.7668953

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174597, upper bound: 2.5173931
time: 0.42 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174597, upper bound: 2.5176634
time: 0.42 seconds

## BFS NS instance: NS_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -8.6723347, 7.5149980, -6.8577528, 5.3887672, -13.0512638, 13.7547150
1: -12.1343040, 7.3268261, -9.5304012, 5.6773558, -16.4884472, 16.1097488
2: -10.1676092, 7.2262888, -8.0426435, 5.3084478, -14.3055696, 14.6350336
3: -11.9992504, 11.4338579, -9.4721928, 8.7725086, -19.4371834, 19.7937813
4: -12.2650061, 9.4761868, -9.8567257, 7.1323280, -18.2082443, 18.7209263

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5177279, upper bound: 2.5173931
time: 0.50 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5177279, upper bound: 2.5176634
time: 0.51 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.37 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5137103, upper bound: 2.5129936
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5149973, upper bound: 2.5155751
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5126425, upper bound: 2.5171454
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5087777, upper bound: 2.5126050
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5181276, upper bound: 2.5181275
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5143002, upper bound: 2.5126050
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5181352, upper bound: 2.5181275
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5181352, upper bound: 2.5181351
NS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5167223, upper bound: 2.5153838
NS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5167223, upper bound: 2.5156954
NS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5179243, upper bound: 2.5180880
NS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180966
NS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5181942
NS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5181948
NS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5182018
NS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5182023
NS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5153838, upper bound: 2.5167223
NS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5153838, upper bound: 2.5167223
NS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5180880, upper bound: 2.5179243
NS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5180966, upper bound: 2.5176983
NS_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5152688, upper bound: 2.5183166
NS_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5152506, upper bound: 2.5183199
NS_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5174864, upper bound: 2.5191666
NS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5174864, upper bound: 2.5191698
NS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5181942, upper bound: 2.5173931
NS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5181942, upper bound: 2.5176613
NS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5180966, upper bound: 2.5173931
NS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5180966, upper bound: 2.5171882
NS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5174597, upper bound: 2.5173931
NS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5174597, upper bound: 2.5176634
NS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5177279, upper bound: 2.5173931
NS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.37
Output dim: 0, lower bound: -2.5177279, upper bound: 2.5176634

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3562506, 1.1616743, -0.4422624, 1.2128479, -1.5690982, 1.6039366
1: -0.3772616, 1.1587057, -0.4775845, 1.2234946, -1.6007562, 1.6362900
2: -0.4859813, 1.1899573, -0.5712293, 1.2429589, -1.7289398, 1.7611864
3: -0.5473121, 1.4236727, -0.6305043, 1.4943078, -2.0416198, 2.0541768
4: -0.8734660, 1.6446971, -1.0105994, 1.7292284, -2.6026943, 2.6552963

Time for backsubstitution: 1.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5137691, upper bound: 2.5150329
time: 0.36 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5156909, upper bound: 2.5156909
time: 0.40 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3970322, 1.1921725, -2.7761545, 2.0858898, -2.4829221, 3.9449866
1: -0.4233569, 1.1999031, -3.5152841, 2.2868772, -2.7102339, 4.6936140
2: -0.5255913, 1.2211068, -3.2617290, 2.2376070, -2.7631984, 4.4450455
3: -0.5881065, 1.4675878, -3.8889868, 2.9340467, -3.5221531, 5.3192520
4: -0.9486377, 1.6975809, -4.1486120, 3.0661988, -4.0148358, 5.8252525

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5146847, upper bound: 2.5115680
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5173550, upper bound: 2.5158553
time: 0.39 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.7539735, 2.0701859, -0.5327253, 1.2670929, -4.0131674, 2.6029112
1: -3.4861541, 2.2661319, -0.5889323, 1.2715299, -4.7576838, 2.8550642
2: -3.2356453, 2.2212410, -0.6767374, 1.3011088, -4.5069461, 2.8979783
3: -3.8560767, 2.9098785, -0.7285765, 1.5475914, -5.3774548, 3.6384549
4: -4.1139598, 3.0412693, -1.1370524, 1.8091310, -5.9230909, 4.1783218

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5165288, upper bound: 2.5179179
time: 0.41 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5165288, upper bound: 2.5179179
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.7539735, 2.0701859, -0.6121423, 1.3185565, -4.0657640, 2.6823282
1: -3.4861541, 2.2661319, -0.6785367, 1.3333414, -4.8194952, 2.9446685
2: -3.2356453, 2.2212410, -0.7678517, 1.3544348, -4.5612931, 2.9890928
3: -3.8560767, 2.9098785, -0.8427736, 1.6363766, -5.4624400, 3.7526522
4: -4.1139598, 3.0412693, -1.2806345, 1.8991282, -6.0130882, 4.3219037

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5143001
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5181276
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.8519592, 2.1517203, -0.6121423, 1.3185565, -4.1371112, 2.7638626
1: -3.6142616, 2.3707585, -0.6785367, 1.3333414, -4.9247966, 3.0492952
2: -3.3539767, 2.3088281, -0.7678517, 1.3544348, -4.6483407, 3.0766792
3: -4.0117965, 3.0343909, -0.8427736, 1.6363766, -5.5861859, 3.8771646
4: -4.2689424, 3.1599696, -1.2806345, 1.8991282, -6.1374044, 4.4406042

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5143077
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5143078
time: 0.42 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3473581, 1.1569773, -3.6753128, 3.1928968, -3.5402551, 4.7377644
1: -0.3669539, 1.1623970, -4.9753809, 3.2069921, -3.5739460, 6.0187211
2: -0.4716939, 1.1842246, -4.3222294, 3.1690164, -3.6407104, 5.3877673
3: -0.5512449, 1.4223603, -5.0598135, 4.8140874, -5.3653321, 6.3589211
4: -0.8731118, 1.6474792, -5.3608537, 4.2708254, -5.1439362, 6.9524179

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 6

## BFS NS instance: NS_A1_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -2.7369781, 1.8885986, -3.8730516, 3.3689032, -6.0745125, 5.6446743
1: -3.3847063, 2.0899162, -5.2563772, 3.3689287, -6.7422457, 7.2008600
2: -3.1705444, 2.0146182, -4.5533957, 3.3352652, -6.4462423, 6.4230928
3: -3.4999008, 2.7175813, -5.3340368, 5.0819759, -8.5069818, 7.8681116
4: -4.0265732, 2.8546615, -5.6377096, 4.4836979, -8.4791346, 8.4136906

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

## BFS NS instance: NS_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.6261075, 1.3201134, -4.2215748, 3.8321548, -4.4582620, 5.5076962
1: -0.6983421, 1.3477855, -5.7547536, 3.7762322, -4.4745736, 7.0617871
2: -0.7848513, 1.3538117, -4.9586401, 3.7693810, -4.5542321, 6.2631702
3: -0.8523064, 1.6343193, -5.8325591, 5.6877012, -6.5400076, 7.4043260
4: -1.2940460, 1.9056911, -6.1373882, 5.0621881, -6.3562336, 8.0430794

Time for backsubstitution: 1.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_B2_A1_B1

### Relational analysis result of NS_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180880
time: 0.44 seconds

## Relational analysis of NS_A1_B2_B1_B2_A1_B2

### Relational analysis result of NS_A1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180880
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.7102028, 1.3766686, -4.2223492, 3.8328681, -4.5430708, 5.5643010
1: -0.7987146, 1.4216498, -5.7558594, 3.7768831, -4.5755978, 7.1334333
2: -0.8826466, 1.4119534, -4.9595451, 3.7700500, -4.6526957, 6.3233223
3: -1.0016927, 1.7330508, -5.8336329, 5.6887684, -6.6904602, 7.4992485
4: -1.4446356, 2.0084071, -6.1384716, 5.0630498, -6.5076857, 8.1468792

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1_B2_A2_B1

### Relational analysis result of NS_A1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180966
time: 0.46 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2_B2

### Relational analysis result of NS_A1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180966
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.5327253, 1.2670929, -7.1787310, 6.1044397, -6.5476007, 7.8015471
1: -0.5889323, 1.2715299, -9.9948807, 6.0036626, -6.5054741, 10.3949804
2: -0.6767374, 1.3011088, -8.4195042, 5.9224820, -6.5438490, 8.9414921
3: -0.7285765, 1.5475914, -9.9142075, 9.3413544, -9.6347837, 10.6893425
4: -1.1370524, 1.8091310, -10.1804905, 7.7595658, -8.8744841, 11.2182570

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169116, upper bound: 2.5165953
time: 0.38 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169116, upper bound: 2.5165898
time: 0.42 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.5327253, 1.2670929, -7.2000875, 6.1167450, -6.5468040, 7.8011122
1: -0.5889323, 1.2715299, -10.0263863, 6.0230365, -6.5091014, 10.3974018
2: -0.6767374, 1.3011088, -8.4461441, 5.9346275, -6.5432315, 8.9431705
3: -0.7285765, 1.5475914, -9.9446583, 9.3778877, -9.6443958, 10.6944580
4: -1.1370524, 1.8091310, -10.2124462, 7.7835112, -8.8796244, 11.2227802

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169116, upper bound: 2.5165959
time: 0.37 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169116, upper bound: 2.5165898
time: 0.45 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.6121423, 1.3185565, -7.1794319, 6.1050982, -6.6367145, 7.8547287
1: -0.6785367, 1.3333414, -9.9958830, 6.0042820, -6.6060524, 10.4595728
2: -0.7678517, 1.3544348, -8.4203224, 5.9230928, -6.6463051, 8.9965267
3: -0.8427736, 1.6363766, -9.9151897, 9.3423347, -9.7509279, 10.7751713
4: -1.2806345, 1.8991282, -10.1814671, 7.7603636, -9.0172348, 11.3109035

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5132939, upper bound: 2.5126716
time: 0.43 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5132939, upper bound: 2.5126660
time: 0.41 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.6121423, 1.3185565, -7.2516923, 6.1653605, -6.6801043, 7.8969994
1: -0.6785367, 1.3333414, -10.1002665, 6.0686531, -6.6510506, 10.5233002
2: -0.7678517, 1.3544348, -8.5064888, 5.9795647, -6.6868467, 9.0480967
3: -0.8427736, 1.6363766, -10.0167522, 9.4500895, -9.8244734, 10.8414793
4: -1.2806345, 1.8991282, -10.2843866, 7.8429604, -9.0776520, 11.3760853

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5132939, upper bound: 2.5126716
time: 0.39 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5132939, upper bound: 2.5126660
time: 0.41 seconds

## BFS NS instance: NS_A2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.6753128, 3.1928968, -0.3473581, 1.1569773, -4.7377644, 3.5402551
1: -4.9753809, 3.2069921, -0.3669539, 1.1623970, -6.0187206, 3.5739460
2: -4.3222294, 3.1690164, -0.4716939, 1.1842246, -5.3877668, 3.6407104
3: -5.0598135, 4.8140874, -0.5512449, 1.4223603, -6.3589215, 5.3653321
4: -5.3608537, 4.2708254, -0.8731118, 1.6474792, -6.9524174, 5.1439371

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 6

## BFS NS instance: NS_A2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.8730516, 3.3689032, -2.7369781, 1.8885986, -5.6446743, 6.0745125
1: -5.2563772, 3.3689287, -3.3847063, 2.0899162, -7.2008591, 6.7422471
2: -4.5533957, 3.3352652, -3.1705444, 2.0146182, -6.4230919, 6.4462433
3: -5.3340368, 5.0819759, -3.4999008, 2.7175813, -7.8681102, 8.5069818
4: -5.6377096, 4.4836979, -4.0265732, 2.8546615, -8.4136915, 8.4791327

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

## BFS NS instance: NS_A2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.2215748, 3.8321548, -0.6261075, 1.3201134, -5.5076966, 4.4582624
1: -5.7547536, 3.7762322, -0.6983421, 1.3477855, -7.0617867, 4.4745741
2: -4.9586401, 3.7693810, -0.7848513, 1.3538117, -6.2631702, 4.5542321
3: -5.8325591, 5.6877012, -0.8523064, 1.6343193, -7.4043264, 6.5400076
4: -6.1373882, 5.0621881, -1.2940460, 1.9056911, -8.0430794, 6.3562341

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180880, upper bound: 2.5176983
time: 0.42 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180880, upper bound: 2.5176983
time: 0.46 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -4.2223492, 3.8328681, -0.7102028, 1.3766686, -5.5643024, 4.5430708
1: -5.7558594, 3.7768831, -0.7987146, 1.4216498, -7.1334338, 4.5755978
2: -4.9595451, 3.7700500, -0.8826466, 1.4119534, -6.3233228, 4.6526961
3: -5.8336329, 5.6887684, -1.0016927, 1.7330508, -7.4992504, 6.6904612
4: -6.1384716, 5.0630498, -1.4446356, 2.0084071, -8.1468792, 6.5076857

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180966, upper bound: 2.5176983
time: 0.48 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180966, upper bound: 2.5176983
time: 0.40 seconds

## BFS NS instance: NS_A2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.6932468, 5.0561733, -6.7242684, 5.2281952, -10.4439087, 11.3351803
1: -7.8776979, 4.9461617, -9.3834820, 5.5021696, -12.7979336, 13.7583246
2: -6.6848583, 4.9140401, -7.8873725, 5.1528721, -11.3073931, 12.2824593
3: -7.8654175, 7.6193733, -9.3261652, 8.5464020, -15.6776333, 16.2584934
4: -8.1991749, 6.5510368, -9.6453066, 6.9007783, -14.6101990, 15.7080212

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## BFS NS instance: NS_A2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.6932468, 5.0561733, -8.5870895, 6.3626013, -11.4132042, 12.6202564
1: -7.8776979, 4.9461617, -12.0149221, 6.7367802, -13.8249111, 15.6111307
2: -6.6848583, 4.9140401, -10.0683823, 6.2628808, -12.2764730, 13.7820683
3: -7.8654175, 7.6193733, -11.8813295, 10.5592909, -17.2643909, 18.1386147
4: -8.1991749, 6.5510368, -12.1446266, 8.1916656, -15.7752733, 17.4673500

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## BFS NS instance: NS_A2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.9547625, 5.4191532, -6.8317237, 5.3128982, -10.9185486, 11.8214417
1: -8.2864189, 5.2587056, -9.5345364, 5.5934081, -13.4511089, 14.2524271
2: -6.9889183, 5.2616816, -8.0127945, 5.2354393, -11.8488111, 12.7789288
3: -8.2509079, 8.0973358, -9.4746609, 8.6866598, -16.3622475, 16.9490528
4: -8.5724125, 7.0173569, -9.7967768, 7.0117779, -15.2486496, 16.3516808

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174864, upper bound: 2.5191666
time: 0.48 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174275, upper bound: 2.5185804
time: 0.50 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.9547625, 5.4191532, -8.6723347, 6.4275594, -11.8736486, 13.0967979
1: -8.2864189, 5.2587056, -12.1343040, 6.8071079, -14.4645596, 16.0902328
2: -6.9889183, 5.2616816, -10.1676092, 6.3258405, -12.8028898, 14.2668343
3: -8.2509079, 8.0973358, -11.9992504, 10.6687088, -17.9311619, 18.8128929
4: -8.5724125, 7.0173569, -12.2650061, 8.2761087, -16.3905792, 18.0970974

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174864, upper bound: 2.5191698
time: 0.44 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174275, upper bound: 2.5185836
time: 0.48 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -7.1787329, 6.1044397, -0.5327253, 1.2670929, -7.8015485, 6.5476003
1: -9.9948807, 6.0036631, -0.5889323, 1.2715299, -10.3949795, 6.5054746
2: -8.4195042, 5.9224825, -0.6767374, 1.3011088, -8.9414930, 6.5438495
3: -9.9142084, 9.3413525, -0.7285765, 1.5475914, -10.6893435, 9.6347809
4: -10.1804905, 7.7595654, -1.1370524, 1.8091310, -11.2182570, 8.8744822

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5165953, upper bound: 2.5171835
time: 0.47 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5165953, upper bound: 2.5173925
time: 0.42 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -7.2000875, 6.1167450, -0.5327253, 1.2670929, -7.8011117, 6.5468035
1: -10.0263863, 6.0230365, -0.5889323, 1.2715299, -10.3974009, 6.5091014
2: -8.4461441, 5.9346275, -0.6767374, 1.3011088, -8.9431715, 6.5432315
3: -9.9446583, 9.3778877, -0.7285765, 1.5475914, -10.6944590, 9.6443949
4: -10.2124462, 7.7835112, -1.1370524, 1.8091310, -11.2227802, 8.8796234

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5165953, upper bound: 2.5174517
time: 0.43 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5165953, upper bound: 2.5176614
time: 0.49 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -7.1794324, 6.1050982, -0.6121423, 1.3185565, -7.8547301, 6.6367149
1: -9.9958858, 6.0042810, -0.6785367, 1.3333414, -10.4595728, 6.6060524
2: -8.4203224, 5.9230928, -0.7678517, 1.3544348, -8.9965267, 6.6463065
3: -9.9151878, 9.3423347, -0.8427736, 1.6363766, -10.7751713, 9.7509289
4: -10.1814661, 7.7603655, -1.2806345, 1.8991282, -11.3109035, 9.0172358

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126716, upper bound: 2.5135657
time: 0.40 seconds

## Relational analysis of NS_A2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5126716, upper bound: 2.5173739
time: 0.42 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -7.2007794, 6.1173949, -0.6121423, 1.3185565, -7.8542905, 6.6359105
1: -10.0273771, 6.0236473, -0.6785367, 1.3333414, -10.4619856, 6.6096735
2: -8.4469509, 5.9352303, -0.7678517, 1.3544348, -8.9981956, 6.6456800
3: -9.9456234, 9.3788528, -0.8427736, 1.6363766, -10.7802725, 9.7605267
4: -10.2134123, 7.7843046, -1.2806345, 1.8991282, -11.3154154, 9.0223742

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5087777, upper bound: 2.5138339
time: 0.42 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126716, upper bound: 2.5138340
time: 0.37 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -8.6478014, 7.4907303, -6.8907700, 5.3909569, -13.0326967, 13.7662697
1: -12.1003590, 7.3022265, -9.5755825, 5.6734295, -16.4527264, 16.1336040
2: -10.1389942, 7.2032175, -8.0785522, 5.3082790, -14.2815332, 14.6516647
3: -11.9664698, 11.3971157, -9.5144176, 8.7780876, -19.4176064, 19.8000793
4: -12.2298355, 9.4446411, -9.8975525, 7.1216726, -18.1700020, 18.7334213

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A2_B2_B1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5171847
time: 0.42 seconds

## Relational analysis of NS_A2_A2_B2_B1_A1_B2

### Relational analysis result of NS_A2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5173925
time: 0.44 seconds

## BFS NS instance: NS_A2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -8.6124039, 7.4529805, -6.8907700, 5.3909569, -12.9874401, 13.7200451
1: -12.0485582, 7.2757049, -9.5755825, 5.6734295, -16.3882084, 16.0966682
2: -10.0973291, 7.1685996, -8.0785522, 5.3082790, -14.2288313, 14.6083059
3: -11.9168777, 11.3555002, -9.5144176, 8.7780876, -19.3567753, 19.7417011
4: -12.1806126, 9.4135494, -9.8975525, 7.1216726, -18.1084080, 18.6886864

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5174573
time: 0.46 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5176634
time: 0.46 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -8.6478014, 7.4907303, -6.8577528, 5.3887672, -13.0254984, 13.7287312
1: -12.1003590, 7.3022265, -9.5304012, 5.6773558, -16.4529400, 16.0831299
2: -10.1389942, 7.2032175, -8.0426435, 5.3084478, -14.2756414, 14.6105995
3: -11.9664698, 11.3971157, -9.4721928, 8.7725086, -19.4030952, 19.7546692
4: -12.2298355, 9.4446411, -9.8567257, 7.1323280, -18.1713142, 18.6874504

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5171336
time: 0.49 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5173739
time: 0.43 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -8.6124039, 7.4529805, -6.8577528, 5.3887672, -12.9802437, 13.6825066
1: -12.0485582, 7.2757049, -9.5304012, 5.6773558, -16.3884220, 16.0461922
2: -10.0973291, 7.1685996, -8.0426435, 5.3084478, -14.2229395, 14.5672388
3: -11.9168777, 11.3555002, -9.4721928, 8.7725086, -19.3422680, 19.6962929
4: -12.1806126, 9.4135494, -9.8567257, 7.1323280, -18.1097202, 18.6427135

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5172759
time: 0.48 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5172759
time: 0.46 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.37 seconds
NS_A1_B1_A1_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5137691, upper bound: 2.5150329
NS_A1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5156909, upper bound: 2.5156909
NS_A1_B1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5146847, upper bound: 2.5115680
NS_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5173550, upper bound: 2.5158553
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5165288, upper bound: 2.5179179
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5165288, upper bound: 2.5179179
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5143001
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5181276
NS_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5143077
NS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5143078
NS_A1_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180880
NS_A1_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180880
NS_A1_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180966
NS_A1_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180966
NS_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5169116, upper bound: 2.5165953
NS_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5169116, upper bound: 2.5165898
NS_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5169116, upper bound: 2.5165959
NS_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5169116, upper bound: 2.5165898
NS_A1_B2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5132939, upper bound: 2.5126716
NS_A1_B2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5132939, upper bound: 2.5126660
NS_A1_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5132939, upper bound: 2.5126716
NS_A1_B2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5132939, upper bound: 2.5126660
NS_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5180880, upper bound: 2.5176983
NS_A2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5180880, upper bound: 2.5176983
NS_A2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5180966, upper bound: 2.5176983
NS_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5180966, upper bound: 2.5176983
NS_A2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5174864, upper bound: 2.5191666
NS_A2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5174275, upper bound: 2.5185804
NS_A2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5174864, upper bound: 2.5191698
NS_A2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5174275, upper bound: 2.5185836
NS_A2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5165953, upper bound: 2.5171835
NS_A2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5165953, upper bound: 2.5173925
NS_A2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5165953, upper bound: 2.5174517
NS_A2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5165953, upper bound: 2.5176614
NS_A2_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5126716, upper bound: 2.5135657
NS_A2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5126716, upper bound: 2.5173739
NS_A2_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5087777, upper bound: 2.5138339
NS_A2_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5126716, upper bound: 2.5138340
NS_A2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5171847
NS_A2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5173925
NS_A2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5174573
NS_A2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5176634
NS_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5171336
NS_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5173739
NS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5172759
NS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.37
Output dim: 0, lower bound: -2.5171831, upper bound: 2.5172759

## BFS NS instance: NS_A1_B1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.2901578, 1.0792370, -0.4422624, 1.2128479, -1.5030057, 1.5214993
1: -0.3188496, 1.0616143, -0.4775845, 1.2234946, -1.5423441, 1.5391988
2: -0.4029438, 1.1045909, -0.5712293, 1.2429589, -1.6459024, 1.6758201
3: -0.4859847, 1.3131876, -0.6305043, 1.4943078, -1.9802924, 1.9436917
4: -0.7336688, 1.5171063, -1.0105994, 1.7292284, -2.4628973, 2.5277057

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5150329, upper bound: 2.5148702
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5150329, upper bound: 2.5156909
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.3426556, 1.1422892, -2.7761545, 2.0858898, -2.4285450, 3.8941319
1: -0.3654389, 1.1365834, -3.5152841, 2.2868772, -2.6523161, 4.6322985
2: -0.4687397, 1.1695594, -3.2617290, 2.2376070, -2.7063463, 4.3942943
3: -0.5346321, 1.3988347, -3.8889868, 2.9340467, -3.4686790, 5.2465672
4: -0.8442517, 1.6151515, -4.1486120, 3.0661988, -3.9104502, 5.7407217

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5172993, upper bound: 2.5158553
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5172993, upper bound: 2.5158553
time: 0.39 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.7539735, 2.0701859, -0.3950344, 1.1911188, -3.9207866, 2.4652205
1: -3.4861541, 2.2661319, -0.4210930, 1.1985672, -4.6620526, 2.6872249
2: -3.2356453, 2.2212410, -0.5236790, 1.2200419, -4.4168029, 2.7449198
3: -3.8560767, 2.9098785, -0.5861320, 1.4661070, -5.2839174, 3.4960105
4: -4.1139598, 3.0412693, -0.9453007, 1.6958464, -5.7876706, 3.9865699

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.7539735, 2.0701859, -2.7448788, 1.8917532, -4.6232176, 4.7878666
1: -3.4861541, 2.2661319, -3.3953857, 2.0928764, -5.5631871, 5.6510744
2: -3.2356453, 2.2212410, -3.1794271, 2.0177436, -5.2150993, 5.3517447
3: -3.8560767, 2.9098785, -3.5107236, 2.7228885, -6.5136600, 6.3775930
4: -4.1139598, 3.0412693, -4.0373936, 2.8596537, -6.9588995, 7.0635610

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.7539735, 2.0701859, -2.8416548, 1.9358467, -4.6694465, 4.8580341
1: -3.4861541, 2.2661319, -3.5171494, 2.1649544, -5.6342912, 5.7420297
2: -3.2356453, 2.2212410, -3.2936087, 2.0705359, -5.2670007, 5.4351840
3: -3.8560767, 2.9098785, -3.6439588, 2.8189430, -6.5998440, 6.4824963
4: -4.1139598, 3.0412693, -4.1866851, 2.9397149, -7.0429592, 7.1794472

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

## BFS NS instance: NS_A1_B2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.6261075, 1.3201134, -4.1580791, 3.7753544, -4.4014606, 5.4472160
1: -0.6983421, 1.3477855, -5.6661081, 3.7206717, -4.4190140, 6.9777040
2: -0.7848513, 1.3538117, -4.8842783, 3.7158217, -4.5006723, 6.1919069
3: -0.8523064, 1.6343193, -5.7462406, 5.6020117, -6.4543176, 7.3219223
4: -1.2940460, 1.9056911, -6.0474215, 4.9918461, -6.2858915, 7.9531121

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5179243, upper bound: 2.5166353
time: 0.50 seconds

## Relational analysis of NS_A1_B2_B1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5179243, upper bound: 2.5180880
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.6261075, 1.3201134, -4.2877989, 3.9085767, -4.5346837, 5.5602798
1: -0.6983421, 1.3477855, -5.8487091, 3.8567965, -4.5551381, 7.1389976
2: -0.7848513, 1.3538117, -5.0372524, 3.8418114, -4.6266618, 6.3235350
3: -0.8523064, 1.6343193, -5.9219580, 5.8015747, -6.6538806, 7.4792061
4: -1.2940460, 1.9056911, -6.2315407, 5.1583791, -6.4524240, 8.1312323

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5179243, upper bound: 2.5166353
time: 0.47 seconds

## Relational analysis of NS_A1_B2_B1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5179243, upper bound: 2.5180880
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.7102028, 1.3766686, -4.1588955, 3.7760954, -4.4862981, 5.5038548
1: -0.7987146, 1.4216498, -5.6672697, 3.7213535, -4.5200682, 7.0493984
2: -0.8826466, 1.4119534, -4.8852324, 3.7165174, -4.5991626, 6.2521048
3: -1.0016927, 1.7330508, -5.7473712, 5.6031294, -6.6048222, 7.4168940
4: -1.4446356, 2.0084071, -6.0485659, 4.9927406, -6.4373760, 8.0569725

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143058, upper bound: 2.5128443
time: 0.44 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143058, upper bound: 2.5128443
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.7102028, 1.3766686, -4.3443012, 3.9596505, -4.6698532, 5.6676846
1: -0.7987146, 1.4216498, -5.9307003, 3.9038832, -4.7025976, 7.2844615
2: -0.8826466, 1.4119534, -5.1033545, 3.8897028, -4.7723484, 6.4431477
3: -1.0016927, 1.7330508, -6.0003395, 5.8788414, -6.8805342, 7.6457081
4: -1.4446356, 2.0084071, -6.3107352, 5.2201557, -6.6647911, 8.3059654

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143058, upper bound: 2.5128443
time: 0.43 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143058, upper bound: 2.5128443
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3950344, 1.1911188, -7.0893564, 6.0203209, -6.3214283, 7.6342740
1: -0.4210930, 1.1985672, -9.8665905, 5.9249167, -6.2542510, 10.1844501
2: -0.5236790, 1.2200419, -8.3147917, 5.8446922, -6.3077183, 8.7636509
3: -0.5861320, 1.4661070, -9.7891293, 9.2163086, -9.3652687, 10.4882126
4: -0.9453007, 1.6958464, -10.0556669, 7.6575012, -8.5676346, 10.9742832

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## BFS NS instance: NS_A1_B2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.7448788, 1.8917532, -7.2545805, 6.1759610, -8.6988029, 8.4750957
1: -3.3953857, 2.0928764, -10.1037340, 6.0705919, -9.2399025, 11.2849503
2: -3.1794271, 2.0177436, -8.5083370, 5.9885983, -8.9549217, 9.7239704
3: -3.5107236, 2.7228885, -10.0202885, 9.4476271, -12.4126244, 11.9167099
4: -4.0373936, 2.8596537, -10.2864513, 7.8462467, -11.6854782, 12.3426189

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## BFS NS instance: NS_A1_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3950344, 1.1911188, -7.1118965, 6.0336528, -6.3215585, 7.6346788
1: -0.4210930, 1.1985672, -9.9001293, 5.9450679, -6.2585382, 10.1884632
2: -0.5236790, 1.2200419, -8.3430052, 5.8578138, -6.3079815, 8.7665062
3: -0.5861320, 1.4661070, -9.8214340, 9.2544785, -9.3763208, 10.4947948
4: -0.9453007, 1.6958464, -10.0895014, 7.6818609, -8.5730267, 10.9802723

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## BFS NS instance: NS_A1_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.7448788, 1.8917532, -7.2748618, 6.1871948, -8.6970358, 8.4738874
1: -3.3953857, 2.0928764, -10.1334352, 6.0891399, -9.2428207, 11.2859516
2: -3.1794271, 2.0177436, -8.5335836, 5.9997454, -8.9533939, 9.7246008
3: -3.5107236, 2.7228885, -10.0491142, 9.4825125, -12.4207802, 11.9205389
4: -4.0373936, 2.8596537, -10.3166800, 7.8696575, -11.6902313, 12.3457775

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -4.1580791, 3.7753544, -0.6261075, 1.3201134, -5.4472151, 4.4014606
1: -5.6661081, 3.7206717, -0.6983421, 1.3477855, -6.9777021, 4.4190140
2: -4.8842783, 3.7158217, -0.7848513, 1.3538117, -6.1919060, 4.5006723
3: -5.7462406, 5.6020117, -0.8523064, 1.6343193, -7.3219228, 6.4543171
4: -6.0474215, 4.9918461, -1.2940460, 1.9056911, -7.9531121, 6.2858920

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166353, upper bound: 2.5179243
time: 0.48 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166353, upper bound: 2.5179243
time: 0.44 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.2877989, 3.9085767, -0.6261075, 1.3201134, -5.5602798, 4.5346832
1: -5.8487091, 3.8567965, -0.6983421, 1.3477855, -7.1389976, 4.5551386
2: -5.0372524, 3.8418114, -0.7848513, 1.3538117, -6.3235340, 4.6266623
3: -5.9219580, 5.8015747, -0.8523064, 1.6343193, -7.4792061, 6.6538801
4: -6.2315407, 5.1583791, -1.2940460, 1.9056911, -8.1312313, 6.4524250

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166353, upper bound: 2.5179243
time: 0.50 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166353, upper bound: 2.5179243
time: 0.48 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.1588955, 3.7760954, -0.7102028, 1.3766686, -5.5038567, 4.4862981
1: -5.6672697, 3.7213535, -0.7987146, 1.4216498, -7.0493979, 4.5200682
2: -4.8852324, 3.7165174, -0.8826466, 1.4119534, -6.2521043, 4.5991626
3: -5.7473712, 5.6031294, -1.0016927, 1.7330508, -7.4168944, 6.6048222
4: -6.0485659, 4.9927406, -1.4446356, 2.0084071, -8.0569725, 6.4373760

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5128443, upper bound: 2.5143058
time: 0.45 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5128443, upper bound: 2.5176983
time: 0.46 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -4.2885561, 3.9092617, -0.7102028, 1.3766686, -5.6168671, 4.6194644
1: -5.8498087, 3.8574264, -0.7987146, 1.4216498, -7.2106347, 4.6561413
2: -5.0381389, 3.8424530, -0.8826466, 1.4119534, -6.3836689, 4.7250996
3: -5.9230089, 5.8026075, -1.0016927, 1.7330508, -7.5741072, 6.8043003
4: -6.2326002, 5.1592073, -1.4446356, 2.0084071, -8.2337627, 6.6038427

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5128443, upper bound: 2.5143058
time: 0.50 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5128443, upper bound: 2.5176983
time: 0.43 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.9150581, 5.3805423, -6.8317237, 5.3128982, -10.8794346, 11.7810802
1: -8.2361355, 5.2199984, -9.5345364, 5.5934081, -13.4002819, 14.2119417
2: -6.9425473, 5.2252007, -8.0127945, 5.2354393, -11.8031187, 12.7411261
3: -8.2022467, 8.0410023, -9.4746609, 8.6866598, -16.3138790, 16.8912582
4: -8.5155449, 6.9682760, -9.7967768, 7.0117779, -15.1923790, 16.3004875

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5182432, upper bound: 2.5187587
time: 0.43 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5182432, upper bound: 2.5187587
time: 0.45 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.9392948, 5.4248514, -6.8317237, 5.3128982, -10.8975821, 11.8207712
1: -8.2653294, 5.2730703, -9.5345364, 5.5934081, -13.4236450, 14.2585678
2: -6.9720888, 5.2667742, -8.0127945, 5.2354393, -11.8255606, 12.7778931
3: -8.2341700, 8.1017342, -9.4746609, 8.6866598, -16.3433018, 16.9447994
4: -8.5526466, 7.0323858, -9.7967768, 7.0117779, -15.2221966, 16.3568096

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5182432, upper bound: 2.5187587
time: 0.42 seconds

## Relational analysis of NS_A2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5182432, upper bound: 2.5187587
time: 0.41 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.9150581, 5.3805423, -8.6723347, 6.4275594, -11.8345385, 13.0564404
1: -8.2361355, 5.2199984, -12.1343040, 6.8071079, -14.4137325, 16.0497417
2: -6.9425473, 5.2252007, -10.1676092, 6.3258405, -12.7571955, 14.2290344
3: -8.2022467, 8.0410023, -11.9992504, 10.6687088, -17.8827915, 18.7550983
4: -8.5155449, 6.9682760, -12.2650061, 8.2761087, -16.3343086, 18.0459080

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176056, upper bound: 2.5185824
time: 0.44 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176056, upper bound: 2.5185836
time: 0.44 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.9392948, 5.4248514, -8.6723347, 6.4275594, -11.8526850, 13.0961304
1: -8.2653294, 5.2730703, -12.1343040, 6.8071079, -14.4370937, 16.0963726
2: -6.9720888, 5.2667742, -10.1676092, 6.3258405, -12.7796392, 14.2658005
3: -8.2341700, 8.1017342, -11.9992504, 10.6687088, -17.9122143, 18.8086395
4: -8.5526466, 7.0323858, -12.2650061, 8.2761087, -16.3641243, 18.1022263

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176056, upper bound: 2.5185824
time: 0.47 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176056, upper bound: 2.5185836
time: 0.47 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.0893574, 6.0203209, -0.3950344, 1.1911188, -7.6342745, 6.3214288
1: -9.8665924, 5.9249191, -0.4210930, 1.1985672, -10.1844521, 6.2542534
2: -8.3147945, 5.8446941, -0.5236790, 1.2200419, -8.7636509, 6.3077192
3: -9.7891293, 9.2163086, -0.5861320, 1.4661070, -10.4882145, 9.3652678
4: -10.0556707, 7.6575031, -0.9453007, 1.6958464, -10.9742880, 8.5676355

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## BFS NS instance: NS_A2_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.3557491, 6.2714081, -2.7448788, 1.8917532, -8.5597029, 8.7866268
1: -10.2489128, 6.1598883, -3.3953857, 2.0928764, -11.4067984, 9.3218555
2: -8.6268635, 6.0768437, -3.1794271, 2.0177436, -9.8230448, 9.0367575
3: -10.1617470, 9.5894079, -3.5107236, 2.7228885, -12.0381565, 12.5397186
4: -10.4278126, 7.9619126, -4.0373936, 2.8596537, -12.4631748, 11.7942390

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## BFS NS instance: NS_A2_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.1118975, 6.0336542, -0.3950344, 1.1911188, -7.6346788, 6.3215599
1: -9.9001322, 5.9450684, -0.4210930, 1.1985672, -10.1884661, 6.2585382
2: -8.3430071, 5.8578167, -0.5236790, 1.2200419, -8.7665071, 6.3079844
3: -9.8214369, 9.2544823, -0.5861320, 1.4661070, -10.4947977, 9.3763237
4: -10.0895023, 7.6818619, -0.9453007, 1.6958464, -10.9802771, 8.5730257

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## BFS NS instance: NS_A2_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.3745861, 6.2811937, -2.7448788, 1.8917532, -8.5574579, 8.7835541
1: -10.2761984, 6.1773214, -3.3953857, 2.0928764, -11.4059067, 9.3238125
2: -8.6501970, 6.0866270, -3.1794271, 2.0177436, -9.8222332, 9.0339937
3: -10.1883926, 9.6220760, -3.5107236, 2.7228885, -12.0402641, 12.5459194
4: -10.4556828, 7.9845376, -4.0373936, 2.8596537, -12.4644699, 11.7984066

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## BFS NS instance: NS_A2_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.3563466, 6.2719703, -2.8416548, 1.9358467, -8.6064281, 8.8573103
1: -10.2497673, 6.1604161, -3.5171494, 2.1649544, -11.4786234, 9.4132938
2: -8.6275625, 6.0773644, -3.2936087, 2.0705359, -9.8755274, 9.1206808
3: -10.1625843, 9.5902443, -3.6439588, 2.8189430, -12.1250610, 12.6453705
4: -10.4286461, 7.9625940, -4.1866851, 2.9397149, -12.5479460, 11.9107666

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## BFS NS instance: NS_A2_A2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.6478014, 7.4907303, -6.8069677, 5.2901254, -12.9124699, 13.6478882
1: -12.1003590, 7.3022265, -9.5007324, 5.5681057, -16.3213444, 16.0012226
2: -10.1389942, 7.2032175, -7.9836950, 5.2133727, -14.1693506, 14.5082207
3: -11.9664698, 11.3971157, -9.4416971, 8.6506729, -19.2495136, 19.6855068
4: -12.2298355, 9.4446411, -9.7563725, 6.9820871, -18.0059052, 18.5403461

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

## BFS NS instance: NS_A2_A2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.6478014, 7.4907303, -8.6478014, 6.4080291, -13.8701973, 14.9224024
1: -12.1003590, 7.3022265, -12.1003590, 6.7848253, -17.3374138, 17.8375206
2: -10.1389942, 7.2032175, -10.1389942, 6.3066001, -15.1258535, 15.9954357
3: -11.9664698, 11.3971157, -11.9664698, 10.6353474, -20.8202133, 21.5483665
4: -12.2298355, 9.4446411, -12.2298355, 8.2497396, -19.1510601, 20.2891312

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

## BFS NS instance: NS_A2_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.6124039, 7.4529805, -6.8069677, 5.2901254, -12.8672142, 13.6016636
1: -12.0485582, 7.2757049, -9.5007324, 5.5681057, -16.2568283, 15.9642868
2: -10.0973291, 7.1685996, -7.9836950, 5.2133727, -14.1166487, 14.4648581
3: -11.9168777, 11.3555002, -9.4416971, 8.6506729, -19.1886883, 19.6271286
4: -12.1806126, 9.4135494, -9.7563725, 6.9820871, -17.9443130, 18.4956112

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

## BFS NS instance: NS_A2_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.6124039, 7.4529805, -8.6478014, 6.4080291, -13.8249416, 14.8761749
1: -12.0485582, 7.2757049, -12.1003590, 6.7848253, -17.2728958, 17.8005829
2: -10.0973291, 7.1685996, -10.1389942, 6.3066001, -15.0731516, 15.9520741
3: -11.9168777, 11.3555002, -11.9664698, 10.6353474, -20.7593899, 21.4899902
4: -12.1806126, 9.4135494, -12.2298355, 8.2497396, -19.0894699, 20.2443962

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 6

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.6478014, 7.4907303, -6.7702093, 5.2816124, -12.8991165, 13.6099720
1: -12.1003590, 7.3022265, -9.4442530, 5.5647058, -16.3116302, 15.9452963
2: -10.1389942, 7.2032175, -7.9425573, 5.2075701, -14.1581612, 14.4662504
3: -11.9664698, 11.3971157, -9.3902512, 8.6345081, -19.2267342, 19.6338272
4: -12.2298355, 9.4446411, -9.7027397, 6.9911861, -18.0059834, 18.4841576

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.6478014, 7.4907303, -8.6124039, 6.3862586, -13.8392334, 14.8771458
1: -12.1003590, 7.3022265, -12.0485582, 6.7657232, -17.3076000, 17.7730045
2: -10.1389942, 7.2032175, -10.0973291, 6.2844644, -15.0945930, 15.9427338
3: -11.9664698, 11.3971157, -11.9168777, 10.6043463, -20.7724152, 21.4875412
4: -12.2298355, 9.4446411, -12.1806126, 8.2345695, -19.1220036, 20.2275352

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.6124039, 7.4529805, -6.7702093, 5.2816124, -12.8538609, 13.5637474
1: -12.0485582, 7.2757049, -9.4442530, 5.5647058, -16.2471123, 15.9083595
2: -10.0973291, 7.1685996, -7.9425573, 5.2075701, -14.1054611, 14.4228878
3: -11.9168777, 11.3555002, -9.3902512, 8.6345081, -19.1659088, 19.5754490
4: -12.1806126, 9.4135494, -9.7027397, 6.9911861, -17.9443874, 18.4394207

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.6124039, 7.4529805, -8.6124039, 6.3862586, -13.7939777, 14.8309193
1: -12.0485582, 7.2757049, -12.0485582, 6.7657232, -17.2430859, 17.7360687
2: -10.0973291, 7.1685996, -10.0973291, 6.2844644, -15.0418911, 15.8993692
3: -11.9168777, 11.3555002, -11.9168777, 10.6043463, -20.7115898, 21.4291630
4: -12.1806126, 9.4135494, -12.1806126, 8.2345695, -19.0604115, 20.1828003

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.47 + 197.01 = 199.48 seconds
