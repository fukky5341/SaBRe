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
execution time: IAR + RelationalAnalysis = 1.37 + 1.12 = 2.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.5203261, upper bound: 2.5203261

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6

Time for candidate selection: 0.09 seconds

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
- Time for NS candidates: 0.89 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.89
Output dim: 0, lower bound: -2.5201938, upper bound: 2.5194575
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.89
Output dim: 0, lower bound: -2.5203102, upper bound: 2.5203102

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.6754932, 1.3466308, -1.2401047, 1.8555624, -2.5310552, 2.5867350
1: -0.7610564, 1.3813244, -1.4851047, 2.0027251, -2.7637815, 2.8664291
2: -0.8425393, 1.3817630, -1.4969569, 1.8678157, -2.7103548, 2.8787198
3: -0.9229488, 1.6725746, -1.7819047, 2.4911361, -3.4140849, 3.4544792
4: -1.3650292, 1.9489899, -2.2745841, 2.7287276, -4.0937567, 4.2235737

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.09 seconds

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
time: 0.36 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7.0239887, 6.3770490, -1.2156670, 1.8306491, -8.6132889, 7.5927148
1: -9.7314625, 6.2030640, -1.4499383, 1.9785936, -11.3731203, 7.6530023
2: -8.2329721, 6.1651368, -1.4698651, 1.8486125, -9.7795572, 7.6350012
3: -9.6892891, 9.5362921, -1.7465651, 2.4514024, -11.8599110, 11.2159281
4: -10.0884438, 8.2303915, -2.2383173, 2.6959260, -12.5598803, 10.4687090

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5194575, upper bound: 2.5201938
time: 0.37 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5194575, upper bound: 2.5203102
time: 0.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.13 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -2.5194265, upper bound: 2.5194265
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -2.5194265, upper bound: 2.5194575
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -2.5194575, upper bound: 2.5201938
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.13
Output dim: 0, lower bound: -2.5194575, upper bound: 2.5203102

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.6754932, 1.3466308, -0.6754932, 1.3466308, -2.0221241, 2.0221241
1: -0.7610564, 1.3813244, -0.7610564, 1.3813244, -2.1423807, 2.1423807
2: -0.8425393, 1.3817630, -0.8425393, 1.3817630, -2.2243023, 2.2243023
3: -0.9229488, 1.6725746, -0.9229488, 1.6725746, -2.5955234, 2.5955234
4: -1.3650292, 1.9489899, -1.3650292, 1.9489899, -3.3140187, 3.3140192

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

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

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176988, upper bound: 2.5172126
time: 0.39 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176614, upper bound: 2.5182023
time: 0.41 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -5.6146450, 5.0941525, -0.6754932, 1.3466308, -6.8465166, 5.7696457
1: -7.6962872, 4.9879494, -0.7610564, 1.3813244, -8.9284773, 5.7490044
2: -6.5841451, 4.9601183, -0.8425393, 1.3817630, -7.8192558, 5.8026576
3: -7.7310028, 7.5750513, -0.9229488, 1.6725746, -9.2390108, 8.4979992
4: -8.1130047, 6.6544933, -1.3650292, 1.9489899, -9.9792252, 8.0195227

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176988, upper bound: 2.5184178
time: 0.37 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5177285, upper bound: 2.5176613
time: 0.43 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7.0239887, 6.3770490, -7.0239887, 5.4972200, -12.1357155, 13.0267391
1: -9.7314625, 6.2030640, -9.7314625, 5.7922802, -15.0621614, 15.4811602
2: -8.2329721, 6.1651368, -8.2329721, 5.4104624, -13.2145462, 13.9734955
3: -9.6892891, 9.5362921, -9.6892891, 8.9427605, -18.0378666, 18.6400681
4: -10.0884438, 8.2303915, -10.0884438, 7.2663670, -16.9806099, 17.9345722

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5178146, upper bound: 2.5191698
time: 0.45 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5177224, upper bound: 2.5176634
time: 0.40 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.21 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -2.5176988, upper bound: 2.5171454
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -2.5176614, upper bound: 2.5181351
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -2.5176988, upper bound: 2.5172126
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -2.5176614, upper bound: 2.5182023
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -2.5176988, upper bound: 2.5184178
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -2.5177285, upper bound: 2.5176613
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -2.5178146, upper bound: 2.5191698
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.21
Output dim: 0, lower bound: -2.5177224, upper bound: 2.5176634

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4422624, 1.2128479, -0.6754932, 1.3466308, -1.7888932, 1.8883404
1: -0.4775845, 1.2234946, -0.7610564, 1.3813244, -1.8589088, 1.9845511
2: -0.5712293, 1.2429589, -0.8425393, 1.3817630, -1.9529923, 2.0854976
3: -0.6305043, 1.4943078, -0.9229488, 1.6725746, -2.3030784, 2.4172566
4: -1.0105994, 1.7292284, -1.3650292, 1.9489899, -2.9595892, 3.0942566

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

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

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5165288, upper bound: 2.5179255
time: 0.42 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5165288, upper bound: 2.5179255
time: 0.42 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.4422624, 1.2128479, -5.5278516, 5.0144835, -5.4567456, 6.6147838
1: -0.4775845, 1.2234946, -7.5726709, 4.9148855, -5.3924699, 8.6289883
2: -0.5712293, 1.2429589, -6.4826288, 4.8853893, -5.4566178, 7.5756626
3: -0.6305043, 1.4943078, -7.6104145, 7.4542704, -8.0847750, 8.9375706
4: -1.0105994, 1.7292284, -7.9915161, 6.5566998, -7.5672989, 9.6192169

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169786, upper bound: 2.5165898
time: 0.42 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169786, upper bound: 2.5165959
time: 0.42 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.7761545, 2.0858898, -5.6062417, 5.0744190, -7.8240452, 7.5371742
1: -3.5152841, 2.2868772, -7.6916857, 4.9607401, -8.4760246, 9.7880745
2: -3.2617290, 2.2376070, -6.5749116, 4.9428654, -8.1547556, 8.6212158
3: -3.8889868, 2.9340467, -7.7243729, 7.5589638, -11.3299065, 10.4502335
4: -4.1486120, 3.0661988, -8.0935211, 6.6196499, -10.7593870, 11.0298758

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171882, upper bound: 2.5181701
time: 0.40 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171882, upper bound: 2.5182023
time: 0.41 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.4223442, 4.8656864, -0.6754932, 1.3466308, -6.6222858, 5.5411797
1: -7.4714098, 4.7590437, -0.7610564, 1.3813244, -8.6534929, 5.5201001
2: -6.3586473, 4.7498059, -0.8425393, 1.3817630, -7.5501747, 5.5923452
3: -7.4818916, 7.2867489, -0.9229488, 1.6725746, -8.9604511, 8.1869946
4: -7.8199315, 6.3481336, -1.3650292, 1.9489899, -9.6333427, 7.7131624

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5165898, upper bound: 2.5169786
time: 0.40 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5165898, upper bound: 2.5169786
time: 0.39 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.2269616, 6.1518574, -0.5813313, 1.2901593, -7.8701973, 6.6469636
1: -10.0624990, 6.0498457, -0.6484846, 1.2993938, -10.4864483, 6.6158299
2: -8.4758892, 5.9668713, -0.7322218, 1.3254614, -9.0183630, 6.6471639
3: -9.9799557, 9.4115915, -0.7853723, 1.5810685, -10.7848320, 9.7646160
4: -10.2488499, 7.8198605, -1.2051851, 1.8473226, -11.3221760, 9.0091448

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5165288, upper bound: 2.5174517
time: 0.43 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5165288, upper bound: 2.5174517
time: 0.37 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.8317237, 6.1549883, -7.0239887, 5.4972200, -11.8993034, 12.7636528
1: -9.5345364, 5.9877911, -9.7314625, 5.7922802, -14.7868433, 15.2189102
2: -8.0127945, 5.9601860, -8.2329721, 5.4104624, -12.9344292, 13.7302160
3: -9.4746609, 9.2591133, -9.6892891, 8.9427605, -17.7707443, 18.2989902
4: -9.7967768, 7.9399056, -10.0884438, 7.2663670, -16.6237526, 17.6008739

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157654, upper bound: 2.5172492
time: 0.42 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5154948, upper bound: 2.5172554
time: 0.38 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.6723347, 7.5149980, -6.9182963, 5.4134960, -13.0818834, 13.8194675
1: -12.1343040, 7.3268261, -9.6091242, 5.6988430, -16.5152969, 16.1945972
2: -10.1676092, 7.2262888, -8.1107140, 5.3302689, -14.3340206, 14.7079191
3: -11.9992504, 11.4338579, -9.5505629, 8.8136482, -19.4883480, 19.8748875
4: -12.2650061, 9.4761868, -9.9370432, 7.1514578, -18.2386169, 18.8061256

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5172490, upper bound: 2.5174573
time: 0.46 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157044, upper bound: 2.5174573
time: 0.41 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.26 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5163191, upper bound: 2.5163191
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5163191, upper bound: 2.5165288
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5165288, upper bound: 2.5179255
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5165288, upper bound: 2.5179255
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5169786, upper bound: 2.5165898
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5169786, upper bound: 2.5165959
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5171882, upper bound: 2.5181701
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5171882, upper bound: 2.5182023
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5165898, upper bound: 2.5169786
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5165898, upper bound: 2.5169786
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5165288, upper bound: 2.5174517
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5165288, upper bound: 2.5174517
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5157654, upper bound: 2.5172492
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5154948, upper bound: 2.5172554
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5172490, upper bound: 2.5174573
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.26
Output dim: 0, lower bound: -2.5157044, upper bound: 2.5174573

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.4422624, 1.2128479, -0.4422624, 1.2128479, -1.6551099, 1.6551100
1: -0.4775845, 1.2234946, -0.4775845, 1.2234946, -1.7010791, 1.7010789
2: -0.5712293, 1.2429589, -0.5712293, 1.2429589, -1.8141878, 1.8141880
3: -0.6305043, 1.4943078, -0.6305043, 1.4943078, -2.1248121, 2.1248121
4: -1.0105994, 1.7292284, -1.0105994, 1.7292284, -2.7398276, 2.7398276

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157282, upper bound: 2.5164284
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5165662, upper bound: 2.5169358
time: 0.42 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.4422624, 1.2128479, -2.7761545, 2.0858898, -2.5281522, 3.9660876
1: -0.4775845, 1.2234946, -3.5152841, 2.2868772, -2.7644615, 4.7168760
2: -0.5712293, 1.2429589, -3.2617290, 2.2376070, -2.8088362, 4.4667535
3: -0.6305043, 1.4943078, -3.8889868, 2.9340467, -3.5645509, 5.3455725
4: -1.0105994, 1.7292284, -4.1486120, 3.0661988, -4.0767980, 5.8574972

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157282, upper bound: 2.5164899
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5165662, upper bound: 2.5171454
time: 0.39 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.7761545, 2.0858898, -0.4422624, 1.2128479, -3.9660873, 2.5281522
1: -3.5152841, 2.2868772, -0.4775845, 1.2234946, -4.7168756, 2.7644615
2: -3.2617290, 2.2376070, -0.5712293, 1.2429589, -4.4667540, 2.8088365
3: -3.8889868, 2.9340467, -0.6305043, 1.4943078, -5.3455734, 3.5645509
4: -4.1486120, 3.0661988, -1.0105994, 1.7292284, -5.8574982, 4.0767975

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5123954, upper bound: 2.5143001
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5143078
time: 0.42 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.7761545, 2.0858898, -2.7668662, 1.9054010, -4.6604733, 4.8274279
1: -3.5152841, 2.2868772, -3.4232130, 2.1117463, -5.6131725, 5.7020044
2: -3.2617290, 2.2376070, -3.2046628, 2.0320072, -5.2569766, 5.3953595
3: -3.8889868, 2.9340467, -3.5389442, 2.7443883, -6.5698938, 6.4323645
4: -4.1486120, 3.0661988, -4.0708928, 2.8829608, -7.0188360, 7.1248541

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5181276
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5181352
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.4422624, 1.2128479, -5.3324895, 4.7840176, -5.2262797, 6.3885760
1: -0.4775845, 1.2234946, -7.3400917, 4.6813440, -5.1589284, 8.3489552
2: -0.5712293, 1.2429589, -6.2535367, 4.6726990, -5.2439270, 7.3043118
3: -0.6305043, 1.4943078, -7.3570223, 7.1615653, -7.7525887, 8.6559486
4: -1.0105994, 1.7292284, -7.6936369, 6.2472916, -7.2578907, 9.2686701

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166005, upper bound: 2.5166826
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5172256, upper bound: 2.5172064
time: 0.42 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.4422624, 1.2128479, -7.1383510, 6.0685167, -6.4204588, 7.7022200
1: -0.4775845, 1.2234946, -9.9352951, 5.9718213, -6.3614292, 10.2730331
2: -0.5712293, 1.2429589, -8.3720541, 5.8897891, -6.4029961, 8.8397799
3: -0.6305043, 1.4943078, -9.8559189, 9.2876968, -9.4843740, 10.5787554
4: -1.0105994, 1.7292284, -10.1251249, 7.7187138, -8.7011728, 11.0738792

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166005, upper bound: 2.5166966
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5172256, upper bound: 2.5172126
time: 0.37 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.7761545, 2.0858898, -5.4986544, 4.9350543, -7.6481829, 7.4064684
1: -3.5152841, 2.2868772, -7.5829573, 4.8250828, -8.3211451, 9.6395493
2: -3.2617290, 2.2376070, -6.4479337, 4.8153105, -7.9954286, 8.4609737
3: -3.8889868, 2.9340467, -7.5879598, 7.3931212, -11.1282978, 10.2939320
4: -4.1486120, 3.0661988, -7.9271946, 6.4337778, -10.5336218, 10.8228569

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5123954, upper bound: 2.5165898
time: 0.41 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171882, upper bound: 2.5180966
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.7761545, 2.0858898, -7.3020964, 6.2225165, -8.7772951, 8.7098083
1: -3.5152841, 2.2868772, -10.1703568, 6.1159949, -9.4042988, 11.5458279
2: -3.2617290, 2.2376070, -8.5639334, 6.0322251, -9.0804043, 9.9847116
3: -3.8889868, 2.9340467, -10.0851030, 9.5166225, -12.8513470, 12.2048273
4: -4.1486120, 3.0661988, -10.3537340, 7.9055910, -11.8560123, 12.6126404

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171882, upper bound: 2.5181948
time: 0.46 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5182023
time: 0.46 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.3324895, 4.7840176, -0.4422624, 1.2128479, -6.3885756, 5.2262797
1: -7.3400917, 4.6813440, -0.4775845, 1.2234946, -8.3489571, 5.1589284
2: -6.2535367, 4.6726990, -0.5712293, 1.2429589, -7.3043122, 5.2439275
3: -7.3570223, 7.1615653, -0.6305043, 1.4943078, -8.6559505, 7.7525883
4: -7.6936369, 6.2472916, -1.0105994, 1.7292284, -9.2686701, 7.2578907

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153838, upper bound: 2.5167223
time: 0.41 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153838, upper bound: 2.5184178
time: 0.41 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.4986534, 4.9350529, -2.7761545, 2.0858898, -7.4064670, 7.6481819
1: -7.5829573, 4.8250809, -3.5152841, 2.2868772, -9.6395493, 8.3211441
2: -6.4479337, 4.8153100, -3.2617290, 2.2376070, -8.4609737, 7.9954281
3: -7.5879593, 7.3931189, -3.8889868, 2.9340467, -10.2939310, 11.1282978
4: -7.9271941, 6.4337764, -4.1486120, 3.0661988, -10.8228569, 10.5336208

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153838, upper bound: 2.5167223
time: 0.37 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153838, upper bound: 2.5184178
time: 0.43 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.1383510, 6.0685167, -0.4422624, 1.2128479, -7.7022200, 6.4204583
1: -9.9352951, 5.9718213, -0.4775845, 1.2234946, -10.2730331, 6.3614297
2: -8.3720541, 5.8897891, -0.5712293, 1.2429589, -8.8397779, 6.4029961
3: -9.8559189, 9.2876968, -0.6305043, 1.4943078, -10.5787554, 9.4843740
4: -10.1251249, 7.7187138, -1.0105994, 1.7292284, -11.0738802, 8.7011719

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126716, upper bound: 2.5135657
time: 0.45 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126722, upper bound: 2.5138339
time: 0.41 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.4024248, 6.3170948, -2.7668662, 1.9054010, -8.6176491, 8.8552074
1: -10.3143454, 6.2045007, -3.4232130, 2.1117463, -11.4877720, 9.3954525
2: -8.6814384, 6.1196551, -3.2046628, 2.0320072, -9.8888311, 9.1057663
3: -10.2254076, 9.6571398, -3.5389442, 2.7443883, -12.1205702, 12.6350584
4: -10.4938707, 8.0201883, -4.0708928, 2.8829608, -12.5499554, 11.8875847

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5126716, upper bound: 2.5173739
time: 0.41 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5126722, upper bound: 2.5176614
time: 0.48 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.8317237, 6.1549883, -6.8317237, 5.3128982, -11.6862621, 12.5272408
1: -9.5345364, 5.9877911, -9.5345364, 5.5934081, -14.5454521, 14.9435978
2: -8.0127945, 5.9601860, -8.0127945, 5.2354393, -12.7340097, 13.4501038
3: -9.4746609, 9.2591133, -9.4746609, 8.6866598, -17.4569855, 18.0318718
4: -9.7967768, 7.9399056, -9.7967768, 7.0117779, -16.3287754, 17.2440166

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5152688, upper bound: 2.5183166
time: 0.37 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174864, upper bound: 2.5191666
time: 0.43 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.8317237, 6.1549883, -8.6723347, 6.4275594, -12.6413641, 13.8025970
1: -9.5345364, 5.9877911, -12.1343040, 6.8071079, -15.5589056, 16.7814007
2: -8.0127945, 5.9601860, -10.1676092, 6.3258405, -13.6880884, 14.9380112
3: -9.4746609, 9.2591133, -11.9992504, 10.6687088, -19.0258999, 19.8957062
4: -9.7967768, 7.9399056, -12.2650061, 8.2761087, -17.4707031, 18.9894352

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5152506, upper bound: 2.5183199
time: 0.45 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174864, upper bound: 2.5191698
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.6723347, 7.5149980, -6.8317237, 5.3128982, -12.9616203, 13.6987934
1: -12.1343040, 7.3268261, -9.5345364, 5.5934081, -16.3832531, 16.0618477
2: -10.1676092, 7.2262888, -8.0127945, 5.2354393, -14.2219152, 14.5618935
3: -11.9992504, 11.4338579, -9.4746609, 8.6866598, -19.3208275, 19.7577305
4: -12.2650061, 9.4761868, -9.7967768, 7.0117779, -18.0741920, 18.6141148

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171988, upper bound: 2.5171336
time: 0.45 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5172490, upper bound: 2.5172759
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.6723347, 7.5149980, -8.6723347, 6.4275594, -13.9167204, 14.9741516
1: -12.1343040, 7.3268261, -12.1343040, 6.8071079, -17.3967056, 17.8996468
2: -10.1676092, 7.2262888, -10.1676092, 6.3258405, -15.1759939, 16.0497990
3: -11.9992504, 11.4338579, -11.9992504, 10.6687088, -20.8897419, 21.6215668
4: -12.2650061, 9.4761868, -12.2650061, 8.2761087, -19.2161198, 20.3595352

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171988, upper bound: 2.5173739
time: 0.46 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5172759
time: 0.53 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.02 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5157282, upper bound: 2.5164284
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5165662, upper bound: 2.5169358
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5157282, upper bound: 2.5164899
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5165662, upper bound: 2.5171454
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5123954, upper bound: 2.5143001
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5143078
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5181276
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5181352
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5166005, upper bound: 2.5166826
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5172256, upper bound: 2.5172064
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5166005, upper bound: 2.5166966
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5172256, upper bound: 2.5172126
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5123954, upper bound: 2.5165898
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5171882, upper bound: 2.5180966
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5171882, upper bound: 2.5181948
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5182023
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5153838, upper bound: 2.5167223
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5153838, upper bound: 2.5184178
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5153838, upper bound: 2.5167223
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5153838, upper bound: 2.5184178
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5126716, upper bound: 2.5135657
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5126722, upper bound: 2.5138339
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5126716, upper bound: 2.5173739
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5126722, upper bound: 2.5176614
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5152688, upper bound: 2.5183166
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5174864, upper bound: 2.5191666
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5152506, upper bound: 2.5183199
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5174864, upper bound: 2.5191698
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5171988, upper bound: 2.5171336
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5172490, upper bound: 2.5172759
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5171988, upper bound: 2.5173739
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.02
Output dim: 0, lower bound: -2.5126050, upper bound: 2.5172759

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0805989, 0.7764781, -0.3473581, 1.1569773, -1.2375762, 1.1238362
1: -0.1202015, 0.7816066, -0.3669539, 1.1623970, -1.2825985, 1.1485602
2: -0.1124061, 0.8064398, -0.4716939, 1.1842246, -1.2966305, 1.2781335
3: -0.3463325, 0.9331710, -0.5512449, 1.4223603, -1.7686927, 1.4844160
4: -0.4303014, 1.0714533, -0.8731118, 1.6474792, -2.0777793, 1.9445652

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5161842, upper bound: 2.5161842
time: 0.37 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5161842, upper bound: 2.5166755
time: 0.40 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3282641, 1.1417439, -0.4422624, 1.2128479, -1.5411117, 1.5840063
1: -0.3512052, 1.1404800, -0.4775845, 1.2234946, -1.5746999, 1.6180642
2: -0.4495943, 1.1684830, -0.5712293, 1.2429589, -1.6925528, 1.7397124
3: -0.5310645, 1.3953160, -0.6305043, 1.4943078, -2.0253723, 2.0258203
4: -0.8314856, 1.6191450, -1.0105994, 1.7292284, -2.5607140, 2.6297445

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166755, upper bound: 2.5166916
time: 0.41 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166755, upper bound: 2.5171829
time: 0.41 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0805989, 0.7764781, -2.7459164, 2.0636122, -2.1442111, 3.4772539
1: -0.1202015, 0.7816066, -3.4748855, 2.2599401, -2.3801413, 4.2000542
2: -0.1124061, 0.8064398, -3.2263663, 2.2144232, -2.3268292, 3.9716442
3: -0.3463325, 0.9331710, -3.8431349, 2.9015131, -3.2478456, 4.7263236
4: -0.4303014, 1.0714533, -4.1025858, 3.0325317, -3.4628325, 5.1299930

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3282641, 1.1417439, -2.7761545, 2.0858898, -2.4141538, 3.8933072
1: -0.3512052, 1.1404800, -3.5152841, 2.2868772, -2.6380823, 4.6296430
2: -0.4495943, 1.1684830, -3.2617290, 2.2376070, -2.6872010, 4.3944035
3: -0.5310645, 1.3953160, -3.8889868, 2.9340467, -3.4651113, 5.2431035
4: -0.8314856, 1.6191450, -4.1486120, 3.0661988, -3.8976839, 5.7438240

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143001, upper bound: 2.5126050
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143077, upper bound: 2.5126050
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.7539735, 2.0701859, -2.7668662, 1.9054010, -4.6373372, 4.8108115
1: -3.4861541, 2.2661319, -3.4232130, 2.1117463, -5.5829258, 5.6799927
2: -3.2356453, 2.2212410, -3.2046628, 2.0320072, -5.2297835, 5.3780789
3: -3.8560767, 2.9098785, -3.5389442, 2.7443883, -6.5360136, 6.4066920
4: -4.1139598, 3.0412693, -4.0708928, 2.8829608, -6.9830313, 7.0981965

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180865, upper bound: 2.5181275
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5087777, upper bound: 2.5087777
time: 0.42 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.8519592, 2.1517203, -2.7668662, 1.9054010, -4.7086844, 4.8805156
1: -3.6142616, 2.3707585, -3.4232130, 2.1117463, -5.6792927, 5.7742443
2: -3.3539767, 2.3088281, -3.2046628, 2.0320072, -5.3168302, 5.4537816
3: -4.0117965, 3.0343909, -3.5389442, 2.7443883, -6.6597576, 6.5169926
4: -4.2689424, 3.1599696, -4.0708928, 2.8829608, -7.1036320, 7.2047033

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180865, upper bound: 2.5181351
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180865, upper bound: 2.5181351
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0805989, 0.7764781, -4.2870603, 3.8229914, -3.9035900, 4.9971495
1: -0.1202015, 0.7816066, -5.8294711, 3.7846522, -3.9048538, 6.5178061
2: -0.1124061, 0.8064398, -5.0312433, 3.7692685, -3.8816743, 5.7548456
3: -0.3463325, 0.9331710, -5.9059887, 5.7125831, -6.0589156, 6.7593217
4: -0.4303014, 1.0714533, -6.2255478, 5.0567827, -5.4870844, 7.2569752

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5178019, upper bound: 2.5162238
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5178019, upper bound: 2.5169357
time: 0.41 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3282641, 1.1417439, -4.7858911, 4.2866125, -4.6148767, 5.8268127
1: -0.3512052, 1.1404800, -6.5435309, 4.2171412, -4.5683465, 7.5488167
2: -0.4495943, 1.1684830, -5.6140180, 4.2049055, -4.6544995, 6.6602974
3: -0.5310645, 1.3953160, -6.5977225, 6.4050088, -6.9360728, 7.8615575
4: -0.8314856, 1.6191450, -6.9262238, 5.6333418, -6.4648275, 8.4579906

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5182932, upper bound: 2.5167312
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5182932, upper bound: 2.5174436
time: 0.41 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0805989, 0.7764781, -6.1009212, 5.0927887, -5.1270375, 6.3613310
1: -0.1202015, 0.7816066, -8.4493818, 5.0548496, -5.1280437, 8.5245571
2: -0.1124061, 0.8064398, -7.1566792, 4.9842172, -5.0638580, 7.3471441
3: -0.3463325, 0.9331710, -8.4049129, 7.8391185, -7.8709154, 8.7376194
4: -0.4303014, 1.0714533, -8.6743870, 6.5290947, -6.9593964, 9.1337547

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3282641, 1.1417439, -6.5895252, 5.5531797, -5.8156176, 7.1675301
1: -0.3512052, 1.1404800, -9.1478996, 5.4884720, -5.7770967, 9.5205622
2: -0.4495943, 1.1684830, -7.7288651, 5.4122720, -5.8224888, 8.2263737
3: -0.5310645, 1.3953160, -9.0878143, 8.5223446, -8.6847410, 9.8129692
4: -0.8314856, 1.6191450, -9.3586760, 7.0918374, -7.9067011, 10.3026352

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5135658, upper bound: 2.5126716
time: 0.42 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5135657, upper bound: 2.5126722
time: 0.41 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.7539735, 2.0701859, -5.4879284, 4.9252863, -7.6159897, 7.3803501
1: -3.4861541, 2.2661319, -7.5672741, 4.8157802, -8.2823496, 9.6036530
2: -3.2356453, 2.2212410, -6.4353800, 4.8060827, -7.9595027, 8.4325781
3: -3.8560767, 2.9098785, -7.5730433, 7.3781557, -11.0805931, 10.2547808
4: -4.1139598, 3.0412693, -7.9121189, 6.4217196, -10.4864769, 10.7826223

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180879
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180879
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.8519592, 2.1517203, -5.4885526, 4.9258537, -7.6878653, 7.4506068
1: -3.6142616, 2.3707585, -7.5681882, 4.8163204, -8.3792114, 9.6987123
2: -3.3539767, 2.3088281, -6.4361105, 4.8066196, -8.0470572, 8.5089293
3: -4.0117965, 3.0343909, -7.5739112, 7.3790264, -11.2051430, 10.3658657
4: -4.2689424, 3.1599696, -7.9129925, 6.4224224, -10.6077385, 10.8899193

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180965
time: 0.44 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180965
time: 0.44 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.7539735, 2.0701859, -7.2915411, 6.2125916, -8.7450209, 8.6843605
1: -3.4861541, 2.2661319, -10.1552048, 6.1067038, -9.3655281, 11.5110931
2: -3.2356453, 2.2212410, -8.5515614, 6.0230465, -9.0446930, 9.9570847
3: -3.8560767, 2.9098785, -10.0703316, 9.5018682, -12.8042374, 12.1664667
4: -4.1139598, 3.0412693, -10.3390007, 7.8935518, -11.8088837, 12.5734138

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5181941
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5181947
time: 0.45 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.8519592, 2.1517203, -7.2921553, 6.2131681, -8.8168983, 8.7545757
1: -3.6142616, 2.3707585, -10.1560841, 6.1072421, -9.4623823, 11.6060839
2: -3.3539767, 2.3088281, -8.5522814, 6.0235782, -9.1322327, 10.0333891
3: -4.0117965, 3.0343909, -10.0711899, 9.5027266, -12.9287491, 12.2775059
4: -4.2689424, 3.1599696, -10.3398561, 7.8942509, -11.9301424, 12.6806526

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5182017
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5182023
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.6753128, 3.1928968, -0.3473581, 1.1569773, -4.7377644, 3.5402551
1: -4.9753809, 3.2069921, -0.3669539, 1.1623970, -6.0187206, 3.5739460
2: -4.3222294, 3.1690164, -0.4716939, 1.1842246, -5.3877668, 3.6407104
3: -5.0598135, 4.8140874, -0.5512449, 1.4223603, -6.3589215, 5.3653321
4: -5.3608537, 4.2708254, -0.8731118, 1.6474792, -6.9524174, 5.1439371

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5162238, upper bound: 2.5178019
time: 0.39 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5162238, upper bound: 2.5182932
time: 0.41 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.1351776, 3.7526097, -0.4422624, 1.2128479, -5.3018622, 4.1948719
1: -5.6317034, 3.7033772, -0.4775845, 1.2234946, -6.7940021, 4.1809616
2: -4.8576083, 3.6948140, -0.5712293, 1.2429589, -6.0467992, 4.2660432
3: -5.7126884, 5.5687056, -0.6305043, 1.4943078, -7.1398125, 6.1992097
4: -6.0163565, 4.9660902, -1.0105994, 1.7292284, -7.7378511, 5.9766893

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169357, upper bound: 2.5186518
time: 0.38 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169357, upper bound: 2.5191431
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.8730516, 3.3689032, -2.7459164, 2.0636122, -5.8155613, 6.0831938
1: -5.2563772, 3.3689287, -3.4748855, 2.2599401, -7.3744478, 6.8294277
2: -4.5533957, 3.3352652, -3.2263663, 2.2144232, -6.6114974, 6.5001078
3: -5.3340368, 5.0819759, -3.8431349, 2.9015131, -8.0672579, 8.8384838
4: -5.6377096, 4.4836979, -4.1025858, 3.0325317, -8.5904274, 8.5528660

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -4.3192077, 3.9213142, -2.7761545, 2.0858898, -6.3406262, 6.6781673
1: -5.8936520, 3.8581808, -3.5152841, 2.2868772, -8.1096506, 7.3734646
2: -5.0728083, 3.8531051, -3.2617290, 2.2376070, -7.2278643, 7.0714388
3: -5.9677629, 5.8218279, -3.8889868, 2.9340467, -8.8051929, 9.6372404
4: -6.2741413, 5.1704597, -4.1486120, 3.0661988, -9.3200951, 9.3120556

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180879, upper bound: 2.5179243
time: 0.42 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180965, upper bound: 2.5176983
time: 0.40 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.3659849, 6.2810764, -2.7668662, 1.9054010, -8.5823755, 8.8184624
1: -10.2635994, 6.1689324, -3.4232130, 2.1117463, -11.4388580, 9.3590689
2: -8.6388559, 6.0857844, -3.2046628, 2.0320072, -9.8477478, 9.0713816
3: -10.1760607, 9.6037617, -3.5389442, 2.7443883, -12.0727901, 12.5816736
4: -10.4421167, 7.9736271, -4.0708928, 2.8829608, -12.4994984, 11.8398819

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174125, upper bound: 2.5173738
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174125, upper bound: 2.5173738
time: 0.46 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.3846717, 6.2907071, -2.7668662, 1.9054010, -8.5800257, 8.8152504
1: -10.2906399, 6.1862459, -3.4232130, 2.1117463, -11.4377747, 9.3609257
2: -8.6619911, 6.0954227, -3.2046628, 2.0320072, -9.8467875, 9.0684862
3: -10.2024832, 9.6361961, -3.5389442, 2.7443883, -12.0747232, 12.5876703
4: -10.4697390, 7.9961643, -4.0708928, 2.8829608, -12.5005960, 11.8439846

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5178487, upper bound: 2.5176613
time: 0.43 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126660, upper bound: 2.5133608
time: 0.46 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.6932468, 5.0561733, -6.7242684, 5.2281952, -10.4439087, 11.3351803
1: -7.8776979, 4.9461617, -9.3834820, 5.5021696, -12.7979336, 13.7583246
2: -6.6848583, 4.9140401, -7.8873725, 5.1528721, -11.3073931, 12.2824593
3: -7.8654175, 7.6193733, -9.3261652, 8.5464020, -15.6776333, 16.2584934
4: -8.1991749, 6.5510368, -9.6453066, 6.9007783, -14.6101990, 15.7080212

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5162238, upper bound: 2.5178415
time: 0.40 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176845, upper bound: 2.5185078
time: 0.42 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.9547625, 5.4191532, -6.8317237, 5.3128982, -10.9185486, 11.8214417
1: -8.2864189, 5.2587056, -9.5345364, 5.5934081, -13.4511089, 14.2524271
2: -6.9889183, 5.2616816, -8.0127945, 5.2354393, -11.8488111, 12.7789288
3: -8.2509079, 8.0973358, -9.4746609, 8.6866598, -16.3622475, 16.9490528
4: -8.5724125, 7.0173569, -9.7967768, 7.0117779, -15.2486496, 16.3516808

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166755, upper bound: 2.5186914
time: 0.42 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5185539, upper bound: 2.5194037
time: 0.44 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.6932468, 5.0561733, -8.5870895, 6.3626013, -11.4132042, 12.6202564
1: -7.8776979, 4.9461617, -12.0149221, 6.7367802, -13.8249111, 15.6111307
2: -6.6848583, 4.9140401, -10.0683823, 6.2628808, -12.2764730, 13.7820683
3: -7.8654175, 7.6193733, -11.8813295, 10.5592909, -17.2643909, 18.1386147
4: -8.1991749, 6.5510368, -12.1446266, 8.1916656, -15.7752733, 17.4673500

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.9547625, 5.4191532, -8.6723347, 6.4275594, -11.8736486, 13.0967979
1: -8.2864189, 5.2587056, -12.1343040, 6.8071079, -14.4645596, 16.0902328
2: -6.9889183, 5.2616816, -10.1676092, 6.3258405, -12.8028898, 14.2668343
3: -8.2509079, 8.0973358, -11.9992504, 10.6687088, -17.9311619, 18.8128929
4: -8.5724125, 7.0173569, -12.2650061, 8.2761087, -16.3905792, 18.0970974

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176056, upper bound: 2.5185824
time: 0.45 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5178355, upper bound: 2.5185836
time: 0.43 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.6478014, 7.4907303, -6.8317237, 5.3128982, -12.9358549, 13.6728086
1: -12.1003590, 7.3022265, -9.5345364, 5.5934081, -16.3477459, 16.0352306
2: -10.1389942, 7.2032175, -8.0127945, 5.2354393, -14.1919870, 14.5374584
3: -11.9664698, 11.3971157, -9.4746609, 8.6866598, -19.2867393, 19.7186165
4: -12.2298355, 9.4446411, -9.7967768, 7.0117779, -18.0372639, 18.5806389

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5172307, upper bound: 2.5171336
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5172307, upper bound: 2.5171336
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.6124039, 7.4529805, -6.8317237, 5.3128982, -12.8905983, 13.6265850
1: -12.0485582, 7.2757049, -9.5345364, 5.5934081, -16.2832298, 15.9982920
2: -10.0973291, 7.1685996, -8.0127945, 5.2354393, -14.1392851, 14.4940977
3: -11.9168777, 11.3555002, -9.4746609, 8.6866598, -19.2259102, 19.6602402
4: -12.1806126, 9.4135494, -9.7967768, 7.0117779, -17.9756718, 18.5359020

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5173822, upper bound: 2.5172759
time: 0.45 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5173822, upper bound: 2.5172759
time: 0.45 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.6478014, 7.4907303, -8.6723347, 6.4275594, -13.8909559, 14.9481659
1: -12.1003590, 7.3022265, -12.1343040, 6.8071079, -17.3611984, 17.8730278
2: -10.1389942, 7.2032175, -10.1676092, 6.3258405, -15.1460657, 16.0253658
3: -11.9664698, 11.3971157, -11.9992504, 10.6687088, -20.8556538, 21.5824585
4: -12.2298355, 9.4446411, -12.2650061, 8.2761087, -19.1791935, 20.3260555

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5173951, upper bound: 2.5173738
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5173951, upper bound: 2.5173738
time: 0.45 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.6124039, 7.4529805, -8.6723347, 6.4275594, -13.8457012, 14.9019394
1: -12.0485582, 7.2757049, -12.1343040, 6.8071079, -17.2966843, 17.8360901
2: -10.0973291, 7.1685996, -10.1676092, 6.3258405, -15.0933619, 15.9820004
3: -11.9168777, 11.3555002, -11.9992504, 10.6687088, -20.7948246, 21.5240784
4: -12.1806126, 9.4135494, -12.2650061, 8.2761087, -19.1175995, 20.2813206

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174035, upper bound: 2.5176634
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5174035, upper bound: 2.5176634
time: 0.48 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.47 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5161842, upper bound: 2.5161842
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5161842, upper bound: 2.5166755
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5166755, upper bound: 2.5166916
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5166755, upper bound: 2.5171829
NS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5143001, upper bound: 2.5126050
NS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5143077, upper bound: 2.5126050
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5180865, upper bound: 2.5181275
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5087777, upper bound: 2.5087777
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5180865, upper bound: 2.5181351
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5180865, upper bound: 2.5181351
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5178019, upper bound: 2.5162238
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5178019, upper bound: 2.5169357
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5182932, upper bound: 2.5167312
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5182932, upper bound: 2.5174436
NS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5135658, upper bound: 2.5126716
NS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5135657, upper bound: 2.5126722
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180879
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180879
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180965
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5176983, upper bound: 2.5180965
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5181941
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5181947
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5182017
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5173931, upper bound: 2.5182023
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5162238, upper bound: 2.5178019
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5162238, upper bound: 2.5182932
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5169357, upper bound: 2.5186518
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5169357, upper bound: 2.5191431
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5180879, upper bound: 2.5179243
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5180965, upper bound: 2.5176983
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5174125, upper bound: 2.5173738
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5174125, upper bound: 2.5173738
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5178487, upper bound: 2.5176613
NS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5126660, upper bound: 2.5133608
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5162238, upper bound: 2.5178415
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5176845, upper bound: 2.5185078
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5166755, upper bound: 2.5186914
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5185539, upper bound: 2.5194037
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5176056, upper bound: 2.5185824
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5178355, upper bound: 2.5185836
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5172307, upper bound: 2.5171336
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5172307, upper bound: 2.5171336
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5173822, upper bound: 2.5172759
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5173822, upper bound: 2.5172759
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5173951, upper bound: 2.5173738
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5173951, upper bound: 2.5173738
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5174035, upper bound: 2.5176634
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.47
Output dim: 0, lower bound: -2.5174035, upper bound: 2.5176634

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0805989, 0.7764781, -0.0805989, 0.7764781, -0.8570769, 0.8570769
1: -0.1202015, 0.7816066, -0.1202015, 0.7816066, -0.9018080, 0.9018078
2: -0.1124061, 0.8064398, -0.1124061, 0.8064398, -0.9188458, 0.9188458
3: -0.3463325, 0.9331710, -0.3463325, 0.9331710, -1.2795036, 1.2795036
4: -0.4303014, 1.0714533, -0.4303014, 1.0714533, -1.5017543, 1.5017539

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5132446, upper bound: 2.5121649
time: 0.40 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142121, upper bound: 2.5142121
time: 0.41 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0805989, 0.7764781, -0.3282641, 1.1417439, -1.2223427, 1.1047421
1: -0.1202015, 0.7816066, -0.3512052, 1.1404800, -1.2606814, 1.1328115
2: -0.1124061, 0.8064398, -0.4495943, 1.1684830, -1.2808889, 1.2560340
3: -0.3463325, 0.9331710, -0.5310645, 1.3953160, -1.7416484, 1.4642355
4: -0.4303014, 1.0714533, -0.8314856, 1.6191450, -2.0494459, 1.9029386

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5132446, upper bound: 2.5127922
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142121, upper bound: 2.5150329
time: 0.43 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3282641, 1.1417439, -0.0805989, 0.7764781, -1.1047421, 1.2223427
1: -0.3512052, 1.1404800, -0.1202015, 0.7816066, -1.1328118, 1.2606815
2: -0.4495943, 1.1684830, -0.1124061, 0.8064398, -1.2560340, 1.2808890
3: -0.5310645, 1.3953160, -0.3463325, 0.9331710, -1.4642355, 1.7416484
4: -0.8314856, 1.6191450, -0.4303014, 1.0714533, -1.9029388, 2.0494456

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5133046, upper bound: 2.5121649
time: 0.38 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5150329, upper bound: 2.5148702
time: 0.42 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3282641, 1.1417439, -0.3282641, 1.1417439, -1.4700079, 1.4700078
1: -0.3512052, 1.1404800, -0.3512052, 1.1404800, -1.4916853, 1.4916853
2: -0.4495943, 1.1684830, -0.4495943, 1.1684830, -1.6180772, 1.6180773
3: -0.5310645, 1.3953160, -0.5310645, 1.3953160, -1.9263805, 1.9263805
4: -0.8314856, 1.6191450, -0.8314856, 1.6191450, -2.4506299, 2.4506299

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5133046, upper bound: 2.5129972
time: 0.39 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5133046, upper bound: 2.5154901
time: 0.41 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.7539735, 2.0701859, -2.7448788, 1.8917532, -4.6232176, 4.7878666
1: -3.4861541, 2.2661319, -3.3953857, 2.0928764, -5.5631871, 5.6510744
2: -3.2356453, 2.2212410, -3.1794271, 2.0177436, -5.2150993, 5.3517447
3: -3.8560767, 2.9098785, -3.5107236, 2.7228885, -6.5136600, 6.3775930
4: -4.1139598, 3.0412693, -4.0373936, 2.8596537, -6.9588995, 7.0635610

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.8519592, 2.1517203, -2.7448788, 1.8917532, -4.6945648, 4.8575706
1: -3.6142616, 2.3707585, -3.3953857, 2.0928764, -5.6595531, 5.7453241
2: -3.3539767, 2.3088281, -3.1794271, 2.0177436, -5.3021460, 5.4274473
3: -4.0117965, 3.0343909, -3.5107236, 2.7228885, -6.6374049, 6.4878945
4: -4.2689424, 3.1599696, -4.0373936, 2.8596537, -7.0795012, 7.1700678

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.8519592, 2.1517203, -2.8416548, 1.9358467, -4.7407937, 4.9277387
1: -3.6142616, 2.3707585, -3.5171494, 2.1649544, -5.7306576, 5.8362808
2: -3.3539767, 2.3088281, -3.2936087, 2.0705359, -5.3540473, 5.5108867
3: -4.0117965, 3.0343909, -3.6439588, 2.8189430, -6.7235899, 6.5927982
4: -4.2689424, 3.1599696, -4.1866851, 2.9397149, -7.1635609, 7.2859554

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0805989, 0.7764781, -2.7387879, 2.4044466, -2.4850454, 3.5152659
1: -0.1202015, 0.7816066, -3.6364954, 2.4991283, -2.6193292, 4.4181018
2: -0.1124061, 0.8064398, -3.2318120, 2.4207802, -2.5331862, 4.0382509
3: -0.3463325, 0.9331710, -3.7530384, 3.6096230, -3.9559553, 4.6862087
4: -0.4303014, 1.0714533, -4.0778370, 3.3187928, -3.7490942, 5.1492891

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5146740, upper bound: 2.5121846
time: 0.37 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5162477, upper bound: 2.5143274
time: 0.39 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0805989, 0.7764781, -3.7133267, 3.3684957, -3.4490945, 4.4727678
1: -0.1202015, 0.7816066, -5.0279217, 3.3576045, -3.4778061, 5.7794132
2: -0.1124061, 0.8064398, -4.3656483, 3.3344810, -3.4468870, 5.1543379
3: -0.3463325, 0.9331710, -5.1242142, 4.9981651, -5.3444977, 6.0345259
4: -0.4303014, 1.0714533, -5.4355564, 4.5006022, -4.9309034, 6.5070095

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5132446, upper bound: 2.5127922
time: 0.41 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5162477, upper bound: 2.5155410
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3282641, 1.1417439, -3.0573397, 2.6638999, -2.9921639, 4.1709089
1: -0.3512052, 1.1404800, -4.0954857, 2.7195361, -3.0707409, 5.1986322
2: -0.4495943, 1.1684830, -3.6014028, 2.6683187, -3.1179128, 4.7312517
3: -0.5310645, 1.3953160, -4.2009101, 4.0079241, -4.5389886, 5.5491257
4: -0.8314856, 1.6191450, -4.5045495, 3.6278658, -4.4593515, 6.1236944

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142011, upper bound: 2.5121874
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169537, upper bound: 2.5149855
time: 0.44 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3282641, 1.1417439, -3.9720745, 3.6037841, -3.9320478, 5.0795116
1: -0.3512052, 1.1404800, -5.3986263, 3.5680091, -3.9192142, 6.4919128
2: -0.4495943, 1.1684830, -4.6672277, 3.5552611, -4.0048552, 5.7998800
3: -0.5310645, 1.3953160, -5.4856615, 5.3467135, -5.8777776, 6.8262148
4: -0.8314856, 1.6191450, -5.7903986, 4.7859626, -5.6174479, 7.4095435

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142011, upper bound: 2.5130303
time: 0.42 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169537, upper bound: 2.5160332
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.7539735, 2.0701859, -5.4460998, 4.8855495, -7.5748062, 7.3412251
1: -3.4861541, 2.2661319, -7.5122128, 4.7757583, -8.2413769, 9.5513420
2: -3.2356453, 2.2212410, -6.3864594, 4.7686267, -7.9211993, 8.3868446
3: -3.8560767, 2.9098785, -7.5166707, 7.3188291, -11.0217180, 10.2016439
4: -4.1139598, 3.0412693, -7.8519735, 6.3709087, -10.4338512, 10.7247963

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.7539735, 2.0701859, -5.4859509, 4.9405117, -7.6206932, 7.3690166
1: -3.4861541, 2.2661319, -7.5702424, 4.8367043, -8.2923069, 9.5933838
2: -3.2356453, 2.2212410, -6.4354601, 4.8202071, -7.9638062, 8.4220581
3: -3.8560767, 2.9098785, -7.5710402, 7.4006052, -11.0869455, 10.2429361
4: -4.1139598, 3.0412693, -7.9110727, 6.4448872, -10.4952202, 10.7669563

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.8519592, 2.1517203, -5.4467287, 4.8861232, -7.6466832, 7.4114871
1: -3.6142616, 2.3707585, -7.5131335, 4.7763028, -8.3382397, 9.6464081
2: -3.3539767, 2.3088281, -6.3871956, 4.7691650, -8.0087547, 8.4632015
3: -4.0117965, 3.0343909, -7.5175481, 7.3197079, -11.1462746, 10.3127422
4: -4.2689424, 3.1599696, -7.8528585, 6.3716140, -10.5551147, 10.8321009

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.8519592, 2.1517203, -5.5357876, 4.9861193, -7.7342072, 7.4828858
1: -3.6142616, 2.3707585, -7.6432972, 4.8793731, -8.4282827, 9.7522745
2: -3.3539767, 2.3088281, -6.4938111, 4.8632946, -8.0915756, 8.5494318
3: -4.0117965, 3.0343909, -7.6406374, 7.4703727, -11.2750492, 10.4161625
4: -4.2689424, 3.1599696, -7.9810920, 6.5015054, -10.6690283, 10.9363384

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.7539735, 2.0701859, -7.2545805, 6.1759610, -8.7077141, 8.6486549
1: -3.4861541, 2.2661319, -10.1037340, 6.0705919, -9.3286457, 11.4615784
2: -3.2356453, 2.2212410, -8.5083370, 5.9885983, -9.0097656, 9.9154625
3: -3.8560767, 2.9098785, -10.0202885, 9.4476271, -12.7500792, 12.1180992
4: -4.1139598, 3.0412693, -10.2864513, 7.8462467, -11.7604828, 12.5222826

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.7539735, 2.0701859, -7.2748618, 6.1871948, -8.7059479, 8.6474476
1: -3.4861541, 2.2661319, -10.1334352, 6.0891399, -9.3315630, 11.4625778
2: -3.2356453, 2.2212410, -8.5335836, 5.9997454, -9.0082388, 9.9160929
3: -3.8560767, 2.9098785, -10.0491142, 9.4825125, -12.7582359, 12.1219273
4: -4.1139598, 3.0412693, -10.3166800, 7.8696575, -11.7652359, 12.5254402

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.8519592, 2.1517203, -7.2551985, 6.1765432, -8.7795954, 8.7188768
1: -3.6142616, 2.3707585, -10.1046200, 6.0711379, -9.4255104, 11.5565720
2: -3.3539767, 2.3088281, -8.5090609, 5.9891372, -9.0973120, 9.9917698
3: -4.0117965, 3.0343909, -10.0211487, 9.4484940, -12.8746033, 12.2291374
4: -4.2689424, 3.1599696, -10.2873154, 7.8469534, -11.8817482, 12.6295290

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.8519592, 2.1517203, -7.3250484, 6.2344961, -8.8208427, 8.7592239
1: -3.6142616, 2.3707585, -10.2052784, 6.1335168, -9.4686985, 11.6172152
2: -3.3539767, 2.3088281, -8.5922670, 6.0434661, -9.1358538, 10.0409431
3: -4.0117965, 3.0343909, -10.1192093, 9.5527439, -12.9449730, 12.2925005
4: -4.2689424, 3.1599696, -10.3866320, 7.9274740, -11.9402895, 12.6916981

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.7387879, 2.4044466, -0.0805989, 0.7764781, -3.5152655, 2.4850454
1: -3.6364954, 2.4991283, -0.1202015, 0.7816066, -4.4181018, 2.6193290
2: -3.2318120, 2.4207802, -0.1124061, 0.8064398, -4.0382514, 2.5331862
3: -3.7530384, 3.6096230, -0.3463325, 0.9331710, -4.6862078, 3.9559555
4: -4.0778370, 3.3187928, -0.4303014, 1.0714533, -5.1492901, 3.7490942

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.0573397, 2.6638999, -0.3282641, 1.1417439, -4.1709094, 2.9921637
1: -4.0954857, 2.7195361, -0.3512052, 1.1404800, -5.1986332, 3.0707407
2: -3.6014028, 2.6683187, -0.4495943, 1.1684830, -4.7312503, 3.1179128
3: -4.2009101, 4.0079241, -0.5310645, 1.3953160, -5.5491252, 4.5389886
4: -4.5045495, 3.6278658, -0.8314856, 1.6191450, -6.1236944, 4.4593515

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.7133267, 3.3684957, -0.0805989, 0.7764781, -4.4727678, 3.4490945
1: -5.0279217, 3.3576045, -0.1202015, 0.7816066, -5.7794132, 3.4778059
2: -4.3656483, 3.3344810, -0.1124061, 0.8064398, -5.1543374, 3.4468870
3: -5.1242142, 4.9981651, -0.3463325, 0.9331710, -6.0345259, 5.3444977
4: -5.4355564, 4.5006022, -0.4303014, 1.0714533, -6.5070095, 4.9309034

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.9720745, 3.6037841, -0.3282641, 1.1417439, -5.0795131, 3.9320481
1: -5.3986263, 3.5680091, -0.3512052, 1.1404800, -6.4919128, 3.9192142
2: -4.6672277, 3.5552611, -0.4495943, 1.1684830, -5.7998810, 4.0048547
3: -5.4856615, 5.3467135, -0.5310645, 1.3953160, -6.8262148, 5.8777781
4: -5.7903986, 4.7859626, -0.8314856, 1.6191450, -7.4095435, 5.6174474

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -4.3069487, 3.9102044, -2.7539735, 2.0701859, -6.3128181, 6.6444635
1: -5.8762131, 3.8479524, -3.4861541, 2.2661319, -8.0717058, 7.3341064
2: -5.0584722, 3.8426592, -3.2356453, 2.2212410, -7.1975040, 7.0342903
3: -5.9508057, 5.8050623, -3.8560767, 2.9098785, -8.7638617, 9.5875263
4: -6.2569833, 5.1569223, -4.1139598, 3.0412693, -9.2775822, 9.2631836

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180879, upper bound: 2.5176983
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180879, upper bound: 2.5176983
time: 0.45 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.3076577, 3.9108481, -2.8519592, 2.1517203, -6.3831716, 6.7164230
1: -5.8772240, 3.8485458, -3.6142616, 2.3707585, -8.1668787, 7.4464569
2: -5.0593023, 3.8432651, -3.3539767, 2.3088281, -7.2739615, 7.1219149
3: -5.9517870, 5.8060360, -4.0117965, 3.0343909, -8.8750687, 9.7121897
4: -6.2579775, 5.1577077, -4.2689424, 3.1599696, -9.3850088, 9.3845396

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180965, upper bound: 2.5176983
time: 0.40 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5180965, upper bound: 2.5176983
time: 0.42 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -7.3557491, 6.2714081, -2.7448788, 1.8917532, -8.5597029, 8.7866268
1: -10.2489128, 6.1598883, -3.3953857, 2.0928764, -11.4067984, 9.3218555
2: -8.6268635, 6.0768437, -3.1794271, 2.0177436, -9.8230448, 9.0367575
3: -10.1617470, 9.5894079, -3.5107236, 2.7228885, -12.0381565, 12.5397186
4: -10.4278126, 7.9619126, -4.0373936, 2.8596537, -12.4631748, 11.7942390

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -7.3563466, 6.2719703, -2.8416548, 1.9358467, -8.6064281, 8.8573103
1: -10.2497673, 6.1604161, -3.5171494, 2.1649544, -11.4786234, 9.4132938
2: -8.6275625, 6.0773644, -3.2936087, 2.0705359, -9.8755274, 9.1206808
3: -10.1625843, 9.5902443, -3.6439588, 2.8189430, -12.1250610, 12.6453705
4: -10.4286461, 7.9625940, -4.1866851, 2.9397149, -12.5479460, 11.9107666

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -7.3745861, 6.2811937, -2.7448788, 1.8917532, -8.5574579, 8.7835541
1: -10.2761984, 6.1773214, -3.3953857, 2.0928764, -11.4059067, 9.3238125
2: -8.6501970, 6.0866270, -3.1794271, 2.0177436, -9.8222332, 9.0339937
3: -10.1883926, 9.6220760, -3.5107236, 2.7228885, -12.0402641, 12.5459194
4: -10.4556828, 7.9845376, -4.0373936, 2.8596537, -12.4644699, 11.7984066

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.6932468, 5.0561733, -5.6932468, 4.4604254, -9.7097025, 10.3166523
1: -7.8776979, 4.9461617, -7.8776979, 4.6675220, -12.0074701, 12.2924967
2: -6.6848583, 4.9140401, -6.6848583, 4.4060106, -10.5890036, 11.1011267
3: -7.8654175, 7.6193733, -7.8654175, 7.2203083, -14.4035053, 14.8146162
4: -8.1991749, 6.5510368, -8.1991749, 5.8971386, -13.6405725, 14.2890577

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.6932468, 5.0561733, -5.9547625, 4.7075644, -9.9512892, 10.6734047
1: -7.8776979, 4.9461617, -8.2864189, 4.9360762, -12.2679739, 12.8135633
2: -6.6848583, 4.9140401, -6.9889183, 4.6621242, -10.8358860, 11.5204487
3: -7.8654175, 7.6193733, -8.2509079, 7.6366320, -14.8457575, 15.3108673
4: -8.1991749, 6.5510368, -8.5724125, 6.2593250, -13.9907169, 14.7774754

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.9547625, 5.4191532, -5.6932468, 4.4604254, -10.0664558, 10.6643734
1: -8.2864189, 5.2587056, -7.8776979, 4.6675220, -12.5285349, 12.5969219
2: -6.9889183, 5.2616816, -6.6848583, 4.4060106, -11.0083227, 11.4343767
3: -8.2509079, 8.0973358, -7.8654175, 7.2203083, -14.8997536, 15.3125372
4: -8.5724125, 7.0173569, -8.1991749, 5.8971386, -14.1289949, 14.7436323

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.9547625, 5.4191532, -5.9547625, 4.7075644, -10.3398781, 11.0537281
1: -8.2864189, 5.2587056, -8.2864189, 4.9360762, -12.8291368, 13.1580849
2: -6.9889183, 5.2616816, -6.9889183, 4.6621242, -11.2941380, 11.8937321
3: -8.2509079, 8.0973358, -8.2509079, 7.6366320, -15.3885794, 15.8543139
4: -8.5724125, 7.0173569, -8.5724125, 6.2593250, -14.5172997, 15.2715549

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.9547625, 5.4191532, -8.6478014, 6.4080291, -11.8528881, 13.0710354
1: -8.2864189, 5.2587056, -12.1003590, 6.7848253, -14.4407721, 16.0547256
2: -6.9889183, 5.2616816, -10.1389942, 6.3066001, -12.7826777, 14.2369061
3: -8.2509079, 8.0973358, -11.9664698, 10.6353474, -17.8957233, 18.7788067
4: -8.5724125, 7.0173569, -12.2298355, 8.2497396, -16.3624458, 18.0601711

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176056, upper bound: 2.5185824
time: 0.46 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176056, upper bound: 2.5185824
time: 0.47 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.9547625, 5.4191532, -8.6124039, 6.3862586, -11.8219261, 13.0257788
1: -8.2864189, 5.2587056, -12.0485582, 6.7657232, -14.4109612, 15.9902086
2: -6.9889183, 5.2616816, -10.0973291, 6.2844644, -12.7514172, 14.1842041
3: -8.2509079, 8.0973358, -11.9168777, 10.6043463, -17.8479233, 18.7179794
4: -8.5724125, 7.0173569, -12.1806126, 8.2345695, -16.3333874, 17.9985733

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5178355, upper bound: 2.5185836
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5178355, upper bound: 2.5185836
time: 0.44 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.6478014, 7.4907303, -6.8069677, 5.2901254, -12.9124699, 13.6478882
1: -12.1003590, 7.3022265, -9.5007324, 5.5681057, -16.3213444, 16.0012226
2: -10.1389942, 7.2032175, -7.9836950, 5.2133727, -14.1693506, 14.5082207
3: -11.9664698, 11.3971157, -9.4416971, 8.6506729, -19.2495136, 19.6855068
4: -12.2298355, 9.4446411, -9.7563725, 6.9820871, -18.0059052, 18.5403461

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.6478014, 7.4907303, -6.7702093, 5.2816124, -12.8991165, 13.6099720
1: -12.1003590, 7.3022265, -9.4442530, 5.5647058, -16.3116302, 15.9452963
2: -10.1389942, 7.2032175, -7.9425573, 5.2075701, -14.1581612, 14.4662504
3: -11.9664698, 11.3971157, -9.3902512, 8.6345081, -19.2267342, 19.6338272
4: -12.2298355, 9.4446411, -9.7027397, 6.9911861, -18.0059834, 18.4841576

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.6124039, 7.4529805, -6.8069677, 5.2901254, -12.8672142, 13.6016636
1: -12.0485582, 7.2757049, -9.5007324, 5.5681057, -16.2568283, 15.9642868
2: -10.0973291, 7.1685996, -7.9836950, 5.2133727, -14.1166487, 14.4648581
3: -11.9168777, 11.3555002, -9.4416971, 8.6506729, -19.1886883, 19.6271286
4: -12.1806126, 9.4135494, -9.7563725, 6.9820871, -17.9443130, 18.4956112

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.6124039, 7.4529805, -6.7702093, 5.2816124, -12.8538609, 13.5637474
1: -12.0485582, 7.2757049, -9.4442530, 5.5647058, -16.2471123, 15.9083595
2: -10.0973291, 7.1685996, -7.9425573, 5.2075701, -14.1054611, 14.4228878
3: -11.9168777, 11.3555002, -9.3902512, 8.6345081, -19.1659088, 19.5754490
4: -12.1806126, 9.4135494, -9.7027397, 6.9911861, -17.9443874, 18.4394207

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.6478014, 7.4907303, -8.6478014, 6.4080291, -13.8701973, 14.9224024
1: -12.1003590, 7.3022265, -12.1003590, 6.7848253, -17.3374138, 17.8375206
2: -10.1389942, 7.2032175, -10.1389942, 6.3066001, -15.1258535, 15.9954357
3: -11.9664698, 11.3971157, -11.9664698, 10.6353474, -20.8202133, 21.5483665
4: -12.2298355, 9.4446411, -12.2298355, 8.2497396, -19.1510601, 20.2891312

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.6478014, 7.4907303, -8.6124039, 6.3862586, -13.8392334, 14.8771458
1: -12.1003590, 7.3022265, -12.0485582, 6.7657232, -17.3076000, 17.7730045
2: -10.1389942, 7.2032175, -10.0973291, 6.2844644, -15.0945930, 15.9427338
3: -11.9664698, 11.3971157, -11.9168777, 10.6043463, -20.7724152, 21.4875412
4: -12.2298355, 9.4446411, -12.1806126, 8.2345695, -19.1220036, 20.2275352

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.6124039, 7.4529805, -8.6478014, 6.4080291, -13.8249416, 14.8761749
1: -12.0485582, 7.2757049, -12.1003590, 6.7848253, -17.2728958, 17.8005829
2: -10.0973291, 7.1685996, -10.1389942, 6.3066001, -15.0731516, 15.9520741
3: -11.9168777, 11.3555002, -11.9664698, 10.6353474, -20.7593899, 21.4899902
4: -12.1806126, 9.4135494, -12.2298355, 8.2497396, -19.0894699, 20.2443962

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.6124039, 7.4529805, -8.6124039, 6.3862586, -13.7939777, 14.8309193
1: -12.0485582, 7.2757049, -12.0485582, 6.7657232, -17.2430859, 17.7360687
2: -10.0973291, 7.1685996, -10.0973291, 6.2844644, -15.0418911, 15.8993692
3: -11.9168777, 11.3555002, -11.9168777, 10.6043463, -20.7115898, 21.4291630
4: -12.1806126, 9.4135494, -12.1806126, 8.2345695, -19.0604115, 20.1828003

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 31

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.49 + 218.47 = 220.96 seconds
