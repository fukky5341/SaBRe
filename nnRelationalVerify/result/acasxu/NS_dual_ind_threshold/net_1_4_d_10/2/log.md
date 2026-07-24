## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.055158916499999995


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0278593, 0.0309887, -0.0278593, 0.0309887, -0.0588479, 0.0588479)
1: (-0.0350513, 0.0705397, -0.0350513, 0.0705397, -0.1055910, 0.1055910)
2: (-0.0677722, 0.0423128, -0.0677722, 0.0423128, -0.1100850, 0.1100850)
3: (-0.0527389, 0.0981292, -0.0527389, 0.0981292, -0.1508681, 0.1508681)
4: (-0.0944105, 0.0499190, -0.0944105, 0.0499190, -0.1443295, 0.1443295)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.63 + 0.76 = 1.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0559989, upper bound: 0.0559989

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556559
time: 0.19 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0556391
time: 0.22 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.47 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.47
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556559
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.47
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0556391

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0206600, 0.0216224, -0.0278593, 0.0309887, -0.0516487, 0.0494817
1: -0.0226827, 0.0442280, -0.0350513, 0.0705397, -0.0932224, 0.0792793
2: -0.0535189, 0.0294451, -0.0677722, 0.0423128, -0.0958317, 0.0972173
3: -0.0368305, 0.0571968, -0.0527389, 0.0981292, -0.1349597, 0.1099357
4: -0.0685483, 0.0351860, -0.0944105, 0.0499190, -0.1184673, 0.1295964

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555568, upper bound: 0.0555568
time: 0.18 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555568, upper bound: 0.0556391
time: 0.19 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0213640, 0.0238672, -0.0278593, 0.0309887, -0.0523526, 0.0517265
1: -0.0256137, 0.0480500, -0.0350513, 0.0705397, -0.0961535, 0.0831014
2: -0.0530172, 0.0317309, -0.0677722, 0.0423128, -0.0953299, 0.0995031
3: -0.0398649, 0.0629934, -0.0527389, 0.0981292, -0.1379941, 0.1157324
4: -0.0719933, 0.0355104, -0.0944105, 0.0499190, -0.1219123, 0.1299209

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0555568
time: 0.20 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0556391
time: 0.18 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.02 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.02
Output dim: 0, lower bound: -0.0555568, upper bound: 0.0555568
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.02
Output dim: 0, lower bound: -0.0555568, upper bound: 0.0556391
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.02
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0555568
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.02
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0556391

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0206600, 0.0216224, -0.0206600, 0.0216224, -0.0422824, 0.0422824
1: -0.0226827, 0.0442280, -0.0226827, 0.0442280, -0.0669107, 0.0669107
2: -0.0535189, 0.0294451, -0.0535189, 0.0294451, -0.0829640, 0.0829640
3: -0.0368305, 0.0571968, -0.0368305, 0.0571968, -0.0940273, 0.0940273
4: -0.0685483, 0.0351860, -0.0685483, 0.0351860, -0.1037342, 0.1037342

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11

Time for candidate selection: 0.03 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556328
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556190
time: 0.21 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0206600, 0.0216224, -0.0213640, 0.0238672, -0.0445272, 0.0429864
1: -0.0226827, 0.0442280, -0.0256137, 0.0480500, -0.0707327, 0.0698417
2: -0.0535189, 0.0294451, -0.0530172, 0.0317309, -0.0852498, 0.0824623
3: -0.0368305, 0.0571968, -0.0398649, 0.0629934, -0.0998239, 0.0970616
4: -0.0685483, 0.0351860, -0.0719933, 0.0355104, -0.1040587, 0.1071792

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556485
time: 0.19 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556545
time: 0.19 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0213640, 0.0238672, -0.0206600, 0.0216224, -0.0429864, 0.0445272
1: -0.0256137, 0.0480500, -0.0226827, 0.0442280, -0.0698417, 0.0707327
2: -0.0530172, 0.0317309, -0.0535189, 0.0294451, -0.0824623, 0.0852498
3: -0.0398649, 0.0629934, -0.0368305, 0.0571968, -0.0970616, 0.0998239
4: -0.0719933, 0.0355104, -0.0685483, 0.0351860, -0.1071792, 0.1040587

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555548, upper bound: 0.0550900
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0555568
time: 0.20 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0213640, 0.0238672, -0.0213640, 0.0238672, -0.0452312, 0.0452312
1: -0.0256137, 0.0480500, -0.0256137, 0.0480500, -0.0736638, 0.0736638
2: -0.0530172, 0.0317309, -0.0530172, 0.0317309, -0.0847481, 0.0847481
3: -0.0398649, 0.0629934, -0.0398649, 0.0629934, -0.1028583, 0.1028583
4: -0.0719933, 0.0355104, -0.0719933, 0.0355104, -0.1075037, 0.1075037

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555548, upper bound: 0.0550900
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0555568
time: 0.23 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.09 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.09
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556328
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.09
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556190
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.09
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556485
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.09
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556545
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.09
Output dim: 0, lower bound: -0.0555548, upper bound: 0.0550900
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.09
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0555568
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.09
Output dim: 0, lower bound: -0.0555548, upper bound: 0.0550900
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.09
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0555568

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0174403, 0.0183881, -0.0206600, 0.0216224, -0.0390627, 0.0390481
1: -0.0179365, 0.0346897, -0.0226827, 0.0442280, -0.0621645, 0.0573724
2: -0.0468226, 0.0235523, -0.0535189, 0.0294451, -0.0762677, 0.0770712
3: -0.0312331, 0.0437208, -0.0368305, 0.0571968, -0.0884299, 0.0805513
4: -0.0597628, 0.0287614, -0.0685483, 0.0351860, -0.0949488, 0.0973097

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556204, upper bound: 0.0556204
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556204, upper bound: 0.0556204
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0277988, 0.0316155, -0.0206600, 0.0216224, -0.0494212, 0.0522756
1: -0.0329395, 0.0745442, -0.0226827, 0.0442280, -0.0771675, 0.0972269
2: -0.0723793, 0.0529296, -0.0535189, 0.0294451, -0.1018244, 0.1064486
3: -0.0475005, 0.0975785, -0.0368305, 0.0571968, -0.1046973, 0.1344090
4: -0.0997654, 0.0589750, -0.0685483, 0.0351860, -0.1349514, 0.1275233

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556204, upper bound: 0.0556204
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556204, upper bound: 0.0556204
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0174403, 0.0183881, -0.0213640, 0.0238672, -0.0413075, 0.0397520
1: -0.0179365, 0.0346897, -0.0256137, 0.0480500, -0.0659865, 0.0603034
2: -0.0468226, 0.0235523, -0.0530172, 0.0317309, -0.0785535, 0.0765695
3: -0.0312331, 0.0437208, -0.0398649, 0.0629934, -0.0942266, 0.0835857
4: -0.0597628, 0.0287614, -0.0719933, 0.0355104, -0.0952732, 0.1007547

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556167
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556485
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0277988, 0.0316155, -0.0213640, 0.0238672, -0.0516660, 0.0529795
1: -0.0329395, 0.0745442, -0.0256137, 0.0480500, -0.0809896, 0.1001579
2: -0.0723793, 0.0529296, -0.0530172, 0.0317309, -0.1041102, 0.1059468
3: -0.0475005, 0.0975785, -0.0398649, 0.0629934, -0.1104940, 0.1374434
4: -0.0997654, 0.0589750, -0.0719933, 0.0355104, -0.1352758, 0.1309683

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556167
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556545
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0193350, 0.0216215, -0.0206600, 0.0216224, -0.0409574, 0.0422815
1: -0.0221682, 0.0413110, -0.0226827, 0.0442280, -0.0663962, 0.0639937
2: -0.0484735, 0.0279975, -0.0535189, 0.0294451, -0.0779186, 0.0815165
3: -0.0360067, 0.0530757, -0.0368305, 0.0571968, -0.0932035, 0.0899062
4: -0.0654610, 0.0313325, -0.0685483, 0.0351860, -0.1006470, 0.0998808

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556167, upper bound: 0.0550900
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556167, upper bound: 0.0550900
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0249527, 0.0289441, -0.0206600, 0.0216224, -0.0465751, 0.0496042
1: -0.0302124, 0.0647534, -0.0226827, 0.0442280, -0.0744403, 0.0874361
2: -0.0640738, 0.0455330, -0.0535189, 0.0294451, -0.0935189, 0.0990519
3: -0.0429836, 0.0844141, -0.0368305, 0.0571968, -0.1001804, 0.1212446
4: -0.0914269, 0.0490248, -0.0685483, 0.0351860, -0.1266129, 0.1175731

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556485, upper bound: 0.0555625
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556485, upper bound: 0.0555625
time: 0.22 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0193350, 0.0216215, -0.0213640, 0.0238672, -0.0432022, 0.0429854
1: -0.0221682, 0.0413110, -0.0256137, 0.0480500, -0.0702183, 0.0669248
2: -0.0484735, 0.0279975, -0.0530172, 0.0317309, -0.0802044, 0.0810147
3: -0.0360067, 0.0530757, -0.0398649, 0.0629934, -0.0990001, 0.0929406
4: -0.0654610, 0.0313325, -0.0719933, 0.0355104, -0.1009714, 0.1033258

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0249527, 0.0289441, -0.0213640, 0.0238672, -0.0488199, 0.0503081
1: -0.0302124, 0.0647534, -0.0256137, 0.0480500, -0.0782624, 0.0903672
2: -0.0640738, 0.0455330, -0.0530172, 0.0317309, -0.0958047, 0.0985502
3: -0.0429836, 0.0844141, -0.0398649, 0.0629934, -0.1059770, 0.1242790
4: -0.0914269, 0.0490248, -0.0719933, 0.0355104, -0.1269373, 0.1210181

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555548
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555568
time: 0.23 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.12 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0556204, upper bound: 0.0556204
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0556204, upper bound: 0.0556204
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0556204, upper bound: 0.0556204
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0556204, upper bound: 0.0556204
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556167
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556485
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556167
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556545
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0556167, upper bound: 0.0550900
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0556167, upper bound: 0.0550900
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0556485, upper bound: 0.0555625
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0556485, upper bound: 0.0555625
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555548
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555568

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0174403, 0.0183881, -0.0174403, 0.0183881, -0.0358283, 0.0358283
1: -0.0179365, 0.0346897, -0.0179365, 0.0346897, -0.0526262, 0.0526262
2: -0.0468226, 0.0235523, -0.0468226, 0.0235523, -0.0703749, 0.0703749
3: -0.0312331, 0.0437208, -0.0312331, 0.0437208, -0.0749540, 0.0749540
4: -0.0597628, 0.0287614, -0.0597628, 0.0287614, -0.0885242, 0.0885242

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556133
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555681, upper bound: 0.0555621
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0174403, 0.0183881, -0.0277988, 0.0316155, -0.0490558, 0.0461869
1: -0.0179365, 0.0346897, -0.0329395, 0.0745442, -0.0924807, 0.0676292
2: -0.0468226, 0.0235523, -0.0723793, 0.0529296, -0.0997523, 0.0959316
3: -0.0312331, 0.0437208, -0.0475005, 0.0975785, -0.1288116, 0.0912214
4: -0.0597628, 0.0287614, -0.0997654, 0.0589750, -0.1187378, 0.1285269

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556133
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555681, upper bound: 0.0555621
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0277988, 0.0316155, -0.0174403, 0.0183881, -0.0461869, 0.0490558
1: -0.0329395, 0.0745442, -0.0179365, 0.0346897, -0.0676292, 0.0924807
2: -0.0723793, 0.0529296, -0.0468226, 0.0235523, -0.0959316, 0.0997523
3: -0.0475005, 0.0975785, -0.0312331, 0.0437208, -0.0912214, 0.1288116
4: -0.0997654, 0.0589750, -0.0597628, 0.0287614, -0.1285269, 0.1187378

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556044
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555513, upper bound: 0.0555513
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0277988, 0.0316155, -0.0277988, 0.0316155, -0.0594143, 0.0594143
1: -0.0329395, 0.0745442, -0.0329395, 0.0745442, -0.1074837, 0.1074837
2: -0.0723793, 0.0529296, -0.0723793, 0.0529296, -0.1253090, 0.1253090
3: -0.0475005, 0.0975785, -0.0475005, 0.0975785, -0.1450790, 0.1450790
4: -0.0997654, 0.0589750, -0.0997654, 0.0589750, -0.1587404, 0.1587404

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556044
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555513, upper bound: 0.0555513
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0174403, 0.0183881, -0.0193350, 0.0216215, -0.0390617, 0.0377230
1: -0.0179365, 0.0346897, -0.0221682, 0.0413110, -0.0592475, 0.0568580
2: -0.0468226, 0.0235523, -0.0484735, 0.0279975, -0.0748202, 0.0720258
3: -0.0312331, 0.0437208, -0.0360067, 0.0530757, -0.0843088, 0.0797275
4: -0.0597628, 0.0287614, -0.0654610, 0.0313325, -0.0910953, 0.0942224

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556110
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0555982
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0174403, 0.0183881, -0.0249527, 0.0289441, -0.0463844, 0.0433407
1: -0.0179365, 0.0346897, -0.0302124, 0.0647534, -0.0826899, 0.0649021
2: -0.0468226, 0.0235523, -0.0640738, 0.0455330, -0.0923556, 0.0876261
3: -0.0312331, 0.0437208, -0.0429836, 0.0844141, -0.1156472, 0.0867045
4: -0.0597628, 0.0287614, -0.0914269, 0.0490248, -0.1087876, 0.1201884

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556342
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556196
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0277988, 0.0316155, -0.0193350, 0.0216215, -0.0494203, 0.0509505
1: -0.0329395, 0.0745442, -0.0221682, 0.0413110, -0.0742506, 0.0967125
2: -0.0723793, 0.0529296, -0.0484735, 0.0279975, -0.1003769, 0.1014031
3: -0.0475005, 0.0975785, -0.0360067, 0.0530757, -0.1005763, 0.1335852
4: -0.0997654, 0.0589750, -0.0654610, 0.0313325, -0.1310980, 0.1244360

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556013
time: 0.19 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0555873
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0277988, 0.0316155, -0.0249527, 0.0289441, -0.0567429, 0.0565682
1: -0.0329395, 0.0745442, -0.0302124, 0.0647534, -0.0976930, 0.1047566
2: -0.0723793, 0.0529296, -0.0640738, 0.0455330, -0.1179123, 0.1170035
3: -0.0475005, 0.0975785, -0.0429836, 0.0844141, -0.1319146, 0.1405621
4: -0.0997654, 0.0589750, -0.0914269, 0.0490248, -0.1487902, 0.1504019

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556248
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0555917
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0193350, 0.0216215, -0.0174403, 0.0183881, -0.0377230, 0.0390617
1: -0.0221682, 0.0413110, -0.0179365, 0.0346897, -0.0568580, 0.0592475
2: -0.0484735, 0.0279975, -0.0468226, 0.0235523, -0.0720258, 0.0748202
3: -0.0360067, 0.0530757, -0.0312331, 0.0437208, -0.0797275, 0.0843088
4: -0.0654610, 0.0313325, -0.0597628, 0.0287614, -0.0942224, 0.0910953

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0193350, 0.0216215, -0.0277988, 0.0316155, -0.0509505, 0.0494203
1: -0.0221682, 0.0413110, -0.0329395, 0.0745442, -0.0967125, 0.0742506
2: -0.0484735, 0.0279975, -0.0723793, 0.0529296, -0.1014031, 0.1003769
3: -0.0360067, 0.0530757, -0.0475005, 0.0975785, -0.1335852, 0.1005763
4: -0.0654610, 0.0313325, -0.0997654, 0.0589750, -0.1244360, 0.1310980

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0249527, 0.0289441, -0.0174403, 0.0183881, -0.0433407, 0.0463844
1: -0.0302124, 0.0647534, -0.0179365, 0.0346897, -0.0649021, 0.0826899
2: -0.0640738, 0.0455330, -0.0468226, 0.0235523, -0.0876261, 0.0923556
3: -0.0429836, 0.0844141, -0.0312331, 0.0437208, -0.0867045, 0.1156472
4: -0.0914269, 0.0490248, -0.0597628, 0.0287614, -0.1201884, 0.1087876

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553731, upper bound: 0.0555625
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556196, upper bound: 0.0554980
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0249527, 0.0289441, -0.0277988, 0.0316155, -0.0565682, 0.0567429
1: -0.0302124, 0.0647534, -0.0329395, 0.0745442, -0.1047566, 0.0976930
2: -0.0640738, 0.0455330, -0.0723793, 0.0529296, -0.1170035, 0.1179123
3: -0.0429836, 0.0844141, -0.0475005, 0.0975785, -0.1405621, 0.1319146
4: -0.0914269, 0.0490248, -0.0997654, 0.0589750, -0.1504019, 0.1487902

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553731, upper bound: 0.0555625
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556196, upper bound: 0.0554980
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0193350, 0.0216215, -0.0193350, 0.0216215, -0.0409564, 0.0409564
1: -0.0221682, 0.0413110, -0.0221682, 0.0413110, -0.0634793, 0.0634793
2: -0.0484735, 0.0279975, -0.0484735, 0.0279975, -0.0764710, 0.0764710
3: -0.0360067, 0.0530757, -0.0360067, 0.0530757, -0.0890824, 0.0890824
4: -0.0654610, 0.0313325, -0.0654610, 0.0313325, -0.0967935, 0.0967935

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0193350, 0.0216215, -0.0249527, 0.0289441, -0.0482791, 0.0465741
1: -0.0221682, 0.0413110, -0.0302124, 0.0647534, -0.0869217, 0.0715234
2: -0.0484735, 0.0279975, -0.0640738, 0.0455330, -0.0940065, 0.0920714
3: -0.0360067, 0.0530757, -0.0429836, 0.0844141, -0.1204208, 0.0960593
4: -0.0654610, 0.0313325, -0.0914269, 0.0490248, -0.1144858, 0.1227595

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0249527, 0.0289441, -0.0193350, 0.0216215, -0.0465741, 0.0482791
1: -0.0302124, 0.0647534, -0.0221682, 0.0413110, -0.0715234, 0.0869217
2: -0.0640738, 0.0455330, -0.0484735, 0.0279975, -0.0920714, 0.0940065
3: -0.0429836, 0.0844141, -0.0360067, 0.0530757, -0.0960593, 0.1204208
4: -0.0914269, 0.0490248, -0.0654610, 0.0313325, -0.1227595, 0.1144858

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549628, upper bound: 0.0555548
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555037
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0249527, 0.0289441, -0.0249527, 0.0289441, -0.0538968, 0.0538968
1: -0.0302124, 0.0647534, -0.0302124, 0.0647534, -0.0949658, 0.0949658
2: -0.0640738, 0.0455330, -0.0640738, 0.0455330, -0.1096068, 0.1096068
3: -0.0429836, 0.0844141, -0.0429836, 0.0844141, -0.1273977, 0.1273977
4: -0.0914269, 0.0490248, -0.0914269, 0.0490248, -0.1404517, 0.1404517

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549628, upper bound: 0.0555568
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555037
time: 0.20 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.10 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556133
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0555681, upper bound: 0.0555621
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556133
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0555681, upper bound: 0.0555621
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556044
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0555513, upper bound: 0.0555513
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0556044
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0555513, upper bound: 0.0555513
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556110
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0555982
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556342
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0551976, upper bound: 0.0556196
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556013
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0555873
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0547159, upper bound: 0.0556248
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0555917
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0553731, upper bound: 0.0555625
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0556196, upper bound: 0.0554980
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0553731, upper bound: 0.0555625
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0556196, upper bound: 0.0554980
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0549628, upper bound: 0.0555548
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555037
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0549628, upper bound: 0.0555568
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.10
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555037

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0174846, 0.0184832, -0.0174403, 0.0183881, -0.0358727, 0.0359235
1: -0.0177545, 0.0346891, -0.0179365, 0.0346897, -0.0524442, 0.0526256
2: -0.0469266, 0.0241803, -0.0468226, 0.0235523, -0.0704789, 0.0710029
3: -0.0308810, 0.0437489, -0.0312331, 0.0437208, -0.0746018, 0.0749820
4: -0.0604094, 0.0290925, -0.0597628, 0.0287614, -0.0891708, 0.0888553

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0552061
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0555822
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0169136, 0.0178243, -0.0174403, 0.0183881, -0.0353017, 0.0352645
1: -0.0172633, 0.0331967, -0.0179365, 0.0346897, -0.0519530, 0.0511332
2: -0.0456765, 0.0224354, -0.0468226, 0.0235523, -0.0692288, 0.0692580
3: -0.0304100, 0.0416375, -0.0312331, 0.0437208, -0.0741309, 0.0728707
4: -0.0582723, 0.0274961, -0.0597628, 0.0287614, -0.0870337, 0.0872589

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555822, upper bound: 0.0552061
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555822, upper bound: 0.0555822
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0174846, 0.0184832, -0.0277988, 0.0316155, -0.0491002, 0.0462820
1: -0.0177545, 0.0346891, -0.0329395, 0.0745442, -0.0922987, 0.0676286
2: -0.0469266, 0.0241803, -0.0723793, 0.0529296, -0.0998563, 0.0965596
3: -0.0308810, 0.0437489, -0.0475005, 0.0975785, -0.1284595, 0.0912494
4: -0.0604094, 0.0290925, -0.0997654, 0.0589750, -0.1193844, 0.1288579

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0552061
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0555621
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0169136, 0.0178243, -0.0277988, 0.0316155, -0.0485292, 0.0456231
1: -0.0172633, 0.0331967, -0.0329395, 0.0745442, -0.0918075, 0.0661362
2: -0.0456765, 0.0224354, -0.0723793, 0.0529296, -0.0986062, 0.0948147
3: -0.0304100, 0.0416375, -0.0475005, 0.0975785, -0.1279885, 0.0891381
4: -0.0582723, 0.0274961, -0.0997654, 0.0589750, -0.1172473, 0.1272616

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555681, upper bound: 0.0552061
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555681, upper bound: 0.0555621
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0280088, 0.0324474, -0.0174403, 0.0183881, -0.0463969, 0.0498877
1: -0.0325964, 0.0775581, -0.0179365, 0.0346897, -0.0672861, 0.0954946
2: -0.0725141, 0.0540359, -0.0468226, 0.0235523, -0.0960664, 0.1008585
3: -0.0464773, 0.1009725, -0.0312331, 0.0437208, -0.0901981, 0.1322056
4: -0.1008744, 0.0594039, -0.0597628, 0.0287614, -0.1296358, 0.1191667

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0551828
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0555681
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0273379, 0.0311306, -0.0174403, 0.0183881, -0.0457259, 0.0485709
1: -0.0321509, 0.0732026, -0.0179365, 0.0346897, -0.0668406, 0.0911391
2: -0.0714296, 0.0520608, -0.0468226, 0.0235523, -0.0949819, 0.0988834
3: -0.0465936, 0.0955957, -0.0312331, 0.0437208, -0.0903144, 0.1268288
4: -0.0984768, 0.0579753, -0.0597628, 0.0287614, -0.1272383, 0.1177381

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555621, upper bound: 0.0551828
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555621, upper bound: 0.0555681
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0280088, 0.0324474, -0.0277988, 0.0316155, -0.0596244, 0.0602462
1: -0.0325964, 0.0775581, -0.0329395, 0.0745442, -0.1071406, 0.1104977
2: -0.0725141, 0.0540359, -0.0723793, 0.0529296, -0.1254437, 0.1264152
3: -0.0464773, 0.1009725, -0.0475005, 0.0975785, -0.1440558, 0.1484731
4: -0.1008744, 0.0594039, -0.0997654, 0.0589750, -0.1598494, 0.1591693

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0551828
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0555513
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0273379, 0.0311306, -0.0277988, 0.0316155, -0.0589534, 0.0589294
1: -0.0321509, 0.0732026, -0.0329395, 0.0745442, -0.1066951, 0.1061421
2: -0.0714296, 0.0520608, -0.0723793, 0.0529296, -0.1243593, 0.1244401
3: -0.0465936, 0.0955957, -0.0475005, 0.0975785, -0.1441721, 0.1430962
4: -0.0984768, 0.0579753, -0.0997654, 0.0589750, -0.1574518, 0.1577407

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555513, upper bound: 0.0551828
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555513, upper bound: 0.0555513
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0174846, 0.0184832, -0.0193350, 0.0216215, -0.0391061, 0.0378182
1: -0.0177545, 0.0346891, -0.0221682, 0.0413110, -0.0590656, 0.0568573
2: -0.0469266, 0.0241803, -0.0484735, 0.0279975, -0.0749241, 0.0726538
3: -0.0308810, 0.0437489, -0.0360067, 0.0530757, -0.0839567, 0.0797556
4: -0.0604094, 0.0290925, -0.0654610, 0.0313325, -0.0917419, 0.0945535

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0169136, 0.0178243, -0.0193350, 0.0216215, -0.0385351, 0.0371592
1: -0.0172633, 0.0331967, -0.0221682, 0.0413110, -0.0585744, 0.0553650
2: -0.0456765, 0.0224354, -0.0484735, 0.0279975, -0.0736740, 0.0709089
3: -0.0304100, 0.0416375, -0.0360067, 0.0530757, -0.0834857, 0.0776442
4: -0.0582723, 0.0274961, -0.0654610, 0.0313325, -0.0896048, 0.0929571

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0174846, 0.0184832, -0.0249527, 0.0289441, -0.0464288, 0.0434359
1: -0.0177545, 0.0346891, -0.0302124, 0.0647534, -0.0825080, 0.0649014
2: -0.0469266, 0.0241803, -0.0640738, 0.0455330, -0.0924596, 0.0882542
3: -0.0308810, 0.0437489, -0.0429836, 0.0844141, -0.1152951, 0.0867325
4: -0.0604094, 0.0290925, -0.0914269, 0.0490248, -0.1094342, 0.1205194

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551293, upper bound: 0.0553731
time: 0.19 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551293, upper bound: 0.0556196
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0169136, 0.0178243, -0.0249527, 0.0289441, -0.0458578, 0.0427769
1: -0.0172633, 0.0331967, -0.0302124, 0.0647534, -0.0820168, 0.0634091
2: -0.0456765, 0.0224354, -0.0640738, 0.0455330, -0.0912095, 0.0865092
3: -0.0304100, 0.0416375, -0.0429836, 0.0844141, -0.1148241, 0.0846211
4: -0.0582723, 0.0274961, -0.0914269, 0.0490248, -0.1072971, 0.1189231

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555149, upper bound: 0.0553731
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555149, upper bound: 0.0556196
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0280088, 0.0324474, -0.0193350, 0.0216215, -0.0496303, 0.0517824
1: -0.0325964, 0.0775581, -0.0221682, 0.0413110, -0.0739075, 0.0997264
2: -0.0725141, 0.0540359, -0.0484735, 0.0279975, -0.1005116, 0.1025094
3: -0.0464773, 0.1009725, -0.0360067, 0.0530757, -0.0995530, 0.1369792
4: -0.1008744, 0.0594039, -0.0654610, 0.0313325, -0.1322069, 0.1248649

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0273379, 0.0311306, -0.0193350, 0.0216215, -0.0489594, 0.0504656
1: -0.0321509, 0.0732026, -0.0221682, 0.0413110, -0.0734620, 0.0953708
2: -0.0714296, 0.0520608, -0.0484735, 0.0279975, -0.0994272, 0.1005343
3: -0.0465936, 0.0955957, -0.0360067, 0.0530757, -0.0996693, 0.1316024
4: -0.0984768, 0.0579753, -0.0654610, 0.0313325, -0.1298094, 0.1234363

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0280088, 0.0324474, -0.0249527, 0.0289441, -0.0569530, 0.0574001
1: -0.0325964, 0.0775581, -0.0302124, 0.0647534, -0.0973499, 0.1077705
2: -0.0725141, 0.0540359, -0.0640738, 0.0455330, -0.1180471, 0.1181097
3: -0.0464773, 0.1009725, -0.0429836, 0.0844141, -0.1308914, 0.1439561
4: -0.1008744, 0.0594039, -0.0914269, 0.0490248, -0.1498992, 0.1508308

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551274, upper bound: 0.0550725
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551274, upper bound: 0.0555917
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0273379, 0.0311306, -0.0249527, 0.0289441, -0.0562820, 0.0560833
1: -0.0321509, 0.0732026, -0.0302124, 0.0647534, -0.0969044, 0.1034149
2: -0.0714296, 0.0520608, -0.0640738, 0.0455330, -0.1169626, 0.1161346
3: -0.0465936, 0.0955957, -0.0429836, 0.0844141, -0.1310077, 0.1385793
4: -0.0984768, 0.0579753, -0.0914269, 0.0490248, -0.1475016, 0.1494022

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551913, upper bound: 0.0550725
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551913, upper bound: 0.0555917
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0257626, 0.0305466, -0.0174403, 0.0183881, -0.0441507, 0.0479868
1: -0.0317584, 0.0698348, -0.0179365, 0.0346897, -0.0664481, 0.0877713
2: -0.0655043, 0.0478268, -0.0468226, 0.0235523, -0.0890566, 0.0946494
3: -0.0441914, 0.0912354, -0.0312331, 0.0437208, -0.0879122, 0.1224686
4: -0.0945988, 0.0507568, -0.0597628, 0.0287614, -0.1233602, 0.1105196

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553731, upper bound: 0.0551293
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553731, upper bound: 0.0555149
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0244811, 0.0283961, -0.0174403, 0.0183881, -0.0428692, 0.0458364
1: -0.0294868, 0.0633977, -0.0179365, 0.0346897, -0.0641765, 0.0813342
2: -0.0631345, 0.0446851, -0.0468226, 0.0235523, -0.0866868, 0.0915077
3: -0.0421317, 0.0824423, -0.0312331, 0.0437208, -0.0858525, 0.1136754
4: -0.0901240, 0.0480672, -0.0597628, 0.0287614, -0.1188854, 0.1078300

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556196, upper bound: 0.0551293
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556196, upper bound: 0.0555149
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0257626, 0.0305466, -0.0277988, 0.0316155, -0.0573782, 0.0583453
1: -0.0317584, 0.0698348, -0.0329395, 0.0745442, -0.1063026, 0.1027744
2: -0.0655043, 0.0478268, -0.0723793, 0.0529296, -0.1184339, 0.1202061
3: -0.0441914, 0.0912354, -0.0475005, 0.0975785, -0.1417699, 0.1387360
4: -0.0945988, 0.0507568, -0.0997654, 0.0589750, -0.1535738, 0.1505222

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553749, upper bound: 0.0551293
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553749, upper bound: 0.0554980
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0244811, 0.0283961, -0.0277988, 0.0316155, -0.0560967, 0.0561949
1: -0.0294868, 0.0633977, -0.0329395, 0.0745442, -0.1040310, 0.0963372
2: -0.0631345, 0.0446851, -0.0723793, 0.0529296, -0.1160641, 0.1170644
3: -0.0421317, 0.0824423, -0.0475005, 0.0975785, -0.1397102, 0.1299428
4: -0.0901240, 0.0480672, -0.0997654, 0.0589750, -0.1490990, 0.1478326

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556235, upper bound: 0.0551293
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556235, upper bound: 0.0554980
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0257626, 0.0305466, -0.0193350, 0.0216215, -0.0473841, 0.0498815
1: -0.0317584, 0.0698348, -0.0221682, 0.0413110, -0.0730694, 0.0920031
2: -0.0655043, 0.0478268, -0.0484735, 0.0279975, -0.0935018, 0.0963003
3: -0.0441914, 0.0912354, -0.0360067, 0.0530757, -0.0972671, 0.1272421
4: -0.0945988, 0.0507568, -0.0654610, 0.0313325, -0.1259313, 0.1162178

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0244811, 0.0283961, -0.0193350, 0.0216215, -0.0461026, 0.0477310
1: -0.0294868, 0.0633977, -0.0221682, 0.0413110, -0.0707978, 0.0855659
2: -0.0631345, 0.0446851, -0.0484735, 0.0279975, -0.0911320, 0.0931586
3: -0.0421317, 0.0824423, -0.0360067, 0.0530757, -0.0952074, 0.1184490
4: -0.0901240, 0.0480672, -0.0654610, 0.0313325, -0.1214565, 0.1135281

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0257626, 0.0305466, -0.0249527, 0.0289441, -0.0547068, 0.0554992
1: -0.0317584, 0.0698348, -0.0302124, 0.0647534, -0.0965118, 0.1000472
2: -0.0655043, 0.0478268, -0.0640738, 0.0455330, -0.1110373, 0.1119006
3: -0.0441914, 0.0912354, -0.0429836, 0.0844141, -0.1286055, 0.1342191
4: -0.0945988, 0.0507568, -0.0914269, 0.0490248, -0.1436236, 0.1421837

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553478, upper bound: 0.0551142
time: 0.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553478, upper bound: 0.0555037
time: 0.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0244811, 0.0283961, -0.0249527, 0.0289441, -0.0534253, 0.0533487
1: -0.0294868, 0.0633977, -0.0302124, 0.0647534, -0.0942402, 0.0936100
2: -0.0631345, 0.0446851, -0.0640738, 0.0455330, -0.1086674, 0.1087589
3: -0.0421317, 0.0824423, -0.0429836, 0.0844141, -0.1265458, 0.1254259
4: -0.0901240, 0.0480672, -0.0914269, 0.0490248, -0.1391487, 0.1394941

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556088, upper bound: 0.0551142
time: 0.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556088, upper bound: 0.0555037
time: 0.22 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.14 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0552061
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0555822
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0555822, upper bound: 0.0552061
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0555822, upper bound: 0.0555822
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0552061
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0555621
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0555681, upper bound: 0.0552061
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0555681, upper bound: 0.0555621
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0551828
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0552061, upper bound: 0.0555681
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0555621, upper bound: 0.0551828
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0555621, upper bound: 0.0555681
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0551828
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0551828, upper bound: 0.0555513
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0555513, upper bound: 0.0551828
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0555513, upper bound: 0.0555513
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0551293, upper bound: 0.0553731
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0551293, upper bound: 0.0556196
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0555149, upper bound: 0.0553731
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0555149, upper bound: 0.0556196
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0551274, upper bound: 0.0550725
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0551274, upper bound: 0.0555917
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0551913, upper bound: 0.0550725
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0551913, upper bound: 0.0555917
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0553731, upper bound: 0.0551293
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0553731, upper bound: 0.0555149
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0556196, upper bound: 0.0551293
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0556196, upper bound: 0.0555149
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0553749, upper bound: 0.0551293
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0553749, upper bound: 0.0554980
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0556235, upper bound: 0.0551293
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0556235, upper bound: 0.0554980
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0553478, upper bound: 0.0551142
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0553478, upper bound: 0.0555037
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0556088, upper bound: 0.0551142
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.14
Output dim: 0, lower bound: -0.0556088, upper bound: 0.0555037

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0174846, 0.0184832, -0.0174846, 0.0184832, -0.0359678, 0.0359678
1: -0.0177545, 0.0346891, -0.0177545, 0.0346891, -0.0524436, 0.0524436
2: -0.0469266, 0.0241803, -0.0469266, 0.0241803, -0.0711069, 0.0711069
3: -0.0308810, 0.0437489, -0.0308810, 0.0437489, -0.0746299, 0.0746299
4: -0.0604094, 0.0290925, -0.0604094, 0.0290925, -0.0895019, 0.0895019

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0553629
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
time: 0.18 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0174846, 0.0184832, -0.0169136, 0.0178243, -0.0353089, 0.0353969
1: -0.0177545, 0.0346891, -0.0172633, 0.0331967, -0.0509512, 0.0519524
2: -0.0469266, 0.0241803, -0.0456765, 0.0224354, -0.0693620, 0.0698568
3: -0.0308810, 0.0437489, -0.0304100, 0.0416375, -0.0725185, 0.0741589
4: -0.0604094, 0.0290925, -0.0582723, 0.0274961, -0.0879055, 0.0873647

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0553629
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0549071
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0169136, 0.0178243, -0.0174846, 0.0184832, -0.0353969, 0.0353089
1: -0.0172633, 0.0331967, -0.0177545, 0.0346891, -0.0519524, 0.0509512
2: -0.0456765, 0.0224354, -0.0469266, 0.0241803, -0.0698568, 0.0693620
3: -0.0304100, 0.0416375, -0.0308810, 0.0437489, -0.0741589, 0.0725185
4: -0.0582723, 0.0274961, -0.0604094, 0.0290925, -0.0873647, 0.0879055

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551574, upper bound: 0.0530275
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555655, upper bound: 0.0551896
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0169136, 0.0178243, -0.0169136, 0.0178243, -0.0347379, 0.0347379
1: -0.0172633, 0.0331967, -0.0172633, 0.0331967, -0.0504600, 0.0504600
2: -0.0456765, 0.0224354, -0.0456765, 0.0224354, -0.0681119, 0.0681119
3: -0.0304100, 0.0416375, -0.0304100, 0.0416375, -0.0720475, 0.0720475
4: -0.0582723, 0.0274961, -0.0582723, 0.0274961, -0.0857684, 0.0857684

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551574, upper bound: 0.0535517
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555655, upper bound: 0.0552465
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0174846, 0.0184832, -0.0280088, 0.0324474, -0.0499320, 0.0464920
1: -0.0177545, 0.0346891, -0.0325964, 0.0775581, -0.0953127, 0.0672855
2: -0.0469266, 0.0241803, -0.0725141, 0.0540359, -0.1009625, 0.0966944
3: -0.0308810, 0.0437489, -0.0464773, 0.1009725, -0.1318535, 0.0902262
4: -0.0604094, 0.0290925, -0.1008744, 0.0594039, -0.1198133, 0.1299669

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548878, upper bound: 0.0553892
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0544692
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0174846, 0.0184832, -0.0273379, 0.0311306, -0.0486153, 0.0458211
1: -0.0177545, 0.0346891, -0.0321509, 0.0732026, -0.0909571, 0.0668400
2: -0.0469266, 0.0241803, -0.0714296, 0.0520608, -0.0989874, 0.0956100
3: -0.0308810, 0.0437489, -0.0465936, 0.0955957, -0.1264767, 0.0903425
4: -0.0604094, 0.0290925, -0.0984768, 0.0579753, -0.1183847, 0.1275693

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548878, upper bound: 0.0556068
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0548454
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0169136, 0.0178243, -0.0280088, 0.0324474, -0.0493610, 0.0458331
1: -0.0172633, 0.0331967, -0.0325964, 0.0775581, -0.0948215, 0.0657931
2: -0.0456765, 0.0224354, -0.0725141, 0.0540359, -0.0997124, 0.0949495
3: -0.0304100, 0.0416375, -0.0464773, 0.1009725, -0.1313825, 0.0881148
4: -0.0582723, 0.0274961, -0.1008744, 0.0594039, -0.1176762, 0.1283706

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551429, upper bound: 0.0530275
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555478, upper bound: 0.0551896
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0169136, 0.0178243, -0.0273379, 0.0311306, -0.0480443, 0.0451621
1: -0.0172633, 0.0331967, -0.0321509, 0.0732026, -0.0904659, 0.0653476
2: -0.0456765, 0.0224354, -0.0714296, 0.0520608, -0.0977373, 0.0938650
3: -0.0304100, 0.0416375, -0.0465936, 0.0955957, -0.1260057, 0.0882311
4: -0.0582723, 0.0274961, -0.0984768, 0.0579753, -0.1162476, 0.1259730

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551429, upper bound: 0.0535107
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555478, upper bound: 0.0552465
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0280088, 0.0324474, -0.0174846, 0.0184832, -0.0464920, 0.0499320
1: -0.0325964, 0.0775581, -0.0177545, 0.0346891, -0.0672855, 0.0953127
2: -0.0725141, 0.0540359, -0.0469266, 0.0241803, -0.0966944, 0.1009625
3: -0.0464773, 0.1009725, -0.0308810, 0.0437489, -0.0902262, 0.1318535
4: -0.1008744, 0.0594039, -0.0604094, 0.0290925, -0.1299669, 0.1198133

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549520, upper bound: 0.0529986
time: 0.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551896, upper bound: 0.0553602
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0280088, 0.0324474, -0.0169136, 0.0178243, -0.0458331, 0.0493610
1: -0.0325964, 0.0775581, -0.0172633, 0.0331967, -0.0657931, 0.0948215
2: -0.0725141, 0.0540359, -0.0456765, 0.0224354, -0.0949495, 0.0997124
3: -0.0464773, 0.1009725, -0.0304100, 0.0416375, -0.0881148, 0.1313825
4: -0.1008744, 0.0594039, -0.0582723, 0.0274961, -0.1283706, 0.1176762

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549520, upper bound: 0.0534766
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551896, upper bound: 0.0555906
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0273379, 0.0311306, -0.0174846, 0.0184832, -0.0458211, 0.0486153
1: -0.0321509, 0.0732026, -0.0177545, 0.0346891, -0.0668400, 0.0909571
2: -0.0714296, 0.0520608, -0.0469266, 0.0241803, -0.0956100, 0.0989874
3: -0.0465936, 0.0955957, -0.0308810, 0.0437489, -0.0903425, 0.1264767
4: -0.0984768, 0.0579753, -0.0604094, 0.0290925, -0.1275693, 0.1183847

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551826, upper bound: 0.0531071
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555362, upper bound: 0.0551583
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0273379, 0.0311306, -0.0169136, 0.0178243, -0.0451621, 0.0480443
1: -0.0321509, 0.0732026, -0.0172633, 0.0331967, -0.0653476, 0.0904659
2: -0.0714296, 0.0520608, -0.0456765, 0.0224354, -0.0938650, 0.0977373
3: -0.0465936, 0.0955957, -0.0304100, 0.0416375, -0.0882311, 0.1260057
4: -0.0984768, 0.0579753, -0.0582723, 0.0274961, -0.1259730, 0.1162476

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551826, upper bound: 0.0535648
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555362, upper bound: 0.0551583
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0280088, 0.0324474, -0.0280088, 0.0324474, -0.0604562, 0.0604562
1: -0.0325964, 0.0775581, -0.0325964, 0.0775581, -0.1101546, 0.1101546
2: -0.0725141, 0.0540359, -0.0725141, 0.0540359, -0.1265500, 0.1265500
3: -0.0464773, 0.1009725, -0.0464773, 0.1009725, -0.1474498, 0.1474498
4: -0.1008744, 0.0594039, -0.1008744, 0.0594039, -0.1602783, 0.1602783

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549217, upper bound: 0.0529986
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551583, upper bound: 0.0553602
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0280088, 0.0324474, -0.0273379, 0.0311306, -0.0591394, 0.0597853
1: -0.0325964, 0.0775581, -0.0321509, 0.0732026, -0.1057990, 0.1097091
2: -0.0725141, 0.0540359, -0.0714296, 0.0520608, -0.1245749, 0.1254655
3: -0.0464773, 0.1009725, -0.0465936, 0.0955957, -0.1420730, 0.1475661
4: -0.1008744, 0.0594039, -0.0984768, 0.0579753, -0.1588497, 0.1578807

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549217, upper bound: 0.0533617
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551583, upper bound: 0.0555906
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0273379, 0.0311306, -0.0280088, 0.0324474, -0.0597853, 0.0591394
1: -0.0321509, 0.0732026, -0.0325964, 0.0775581, -0.1097091, 0.1057990
2: -0.0714296, 0.0520608, -0.0725141, 0.0540359, -0.1254655, 0.1245749
3: -0.0465936, 0.0955957, -0.0464773, 0.1009725, -0.1475661, 0.1420730
4: -0.0984768, 0.0579753, -0.1008744, 0.0594039, -0.1578807, 0.1588497

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551826, upper bound: 0.0531071
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555212, upper bound: 0.0551583
time: 0.24 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0273379, 0.0311306, -0.0273379, 0.0311306, -0.0584685, 0.0584685
1: -0.0321509, 0.0732026, -0.0321509, 0.0732026, -0.1053535, 0.1053535
2: -0.0714296, 0.0520608, -0.0714296, 0.0520608, -0.1234904, 0.1234904
3: -0.0465936, 0.0955957, -0.0465936, 0.0955957, -0.1421893, 0.1421893
4: -0.0984768, 0.0579753, -0.0984768, 0.0579753, -0.1564521, 0.1564521

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551826, upper bound: 0.0535312
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555212, upper bound: 0.0551583
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0174846, 0.0184832, -0.0257626, 0.0305466, -0.0480312, 0.0442459
1: -0.0177545, 0.0346891, -0.0317584, 0.0698348, -0.0875894, 0.0664474
2: -0.0469266, 0.0241803, -0.0655043, 0.0478268, -0.0947534, 0.0896846
3: -0.0308810, 0.0437489, -0.0441914, 0.0912354, -0.1221164, 0.0879403
4: -0.0604094, 0.0290925, -0.0945988, 0.0507568, -0.1111662, 0.1236912

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0554649
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0547033
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0174846, 0.0184832, -0.0244811, 0.0283961, -0.0458807, 0.0429644
1: -0.0177545, 0.0346891, -0.0294868, 0.0633977, -0.0811522, 0.0641759
2: -0.0469266, 0.0241803, -0.0631345, 0.0446851, -0.0916117, 0.0873148
3: -0.0308810, 0.0437489, -0.0421317, 0.0824423, -0.1133233, 0.0858805
4: -0.0604094, 0.0290925, -0.0901240, 0.0480672, -0.1084765, 0.1192164

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0554649
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0547033
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0169136, 0.0178243, -0.0257626, 0.0305466, -0.0474602, 0.0435869
1: -0.0172633, 0.0331967, -0.0317584, 0.0698348, -0.0870982, 0.0649551
2: -0.0456765, 0.0224354, -0.0655043, 0.0478268, -0.0935033, 0.0879397
3: -0.0304100, 0.0416375, -0.0441914, 0.0912354, -0.1216454, 0.0858289
4: -0.0582723, 0.0274961, -0.0945988, 0.0507568, -0.1090291, 0.1220949

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550814, upper bound: 0.0532216
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554868, upper bound: 0.0553560
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0169136, 0.0178243, -0.0244811, 0.0283961, -0.0453097, 0.0423054
1: -0.0172633, 0.0331967, -0.0294868, 0.0633977, -0.0806610, 0.0626835
2: -0.0456765, 0.0224354, -0.0631345, 0.0446851, -0.0903616, 0.0855699
3: -0.0304100, 0.0416375, -0.0421317, 0.0824423, -0.1128523, 0.0837692
4: -0.0582723, 0.0274961, -0.0901240, 0.0480672, -0.1063394, 0.1176201

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550814, upper bound: 0.0536820
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554868, upper bound: 0.0553614
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0280088, 0.0324474, -0.0244811, 0.0283961, -0.0564049, 0.0569285
1: -0.0325964, 0.0775581, -0.0294868, 0.0633977, -0.0959941, 0.1070449
2: -0.0725141, 0.0540359, -0.0631345, 0.0446851, -0.1171992, 0.1171703
3: -0.0464773, 0.1009725, -0.0421317, 0.0824423, -0.1289196, 0.1431042
4: -0.1008744, 0.0594039, -0.0901240, 0.0480672, -0.1489415, 0.1495278

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548612, upper bound: 0.0535417
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550957, upper bound: 0.0556074
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0273379, 0.0311306, -0.0257626, 0.0305466, -0.0578844, 0.0568933
1: -0.0321509, 0.0732026, -0.0317584, 0.0698348, -0.1019858, 0.1049610
2: -0.0714296, 0.0520608, -0.0655043, 0.0478268, -0.1192564, 0.1175651
3: -0.0465936, 0.0955957, -0.0441914, 0.0912354, -0.1378290, 0.1397871
4: -0.0984768, 0.0579753, -0.0945988, 0.0507568, -0.1492336, 0.1525741

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551025, upper bound: 0.0530775
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551410, upper bound: 0.0550480
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0273379, 0.0311306, -0.0244811, 0.0283961, -0.0557340, 0.0556118
1: -0.0321509, 0.0732026, -0.0294868, 0.0633977, -0.0955486, 0.1026894
2: -0.0714296, 0.0520608, -0.0631345, 0.0446851, -0.1161148, 0.1151953
3: -0.0465936, 0.0955957, -0.0421317, 0.0824423, -0.1290358, 0.1377274
4: -0.0984768, 0.0579753, -0.0901240, 0.0480672, -0.1465440, 0.1480993

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551025, upper bound: 0.0535919
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551410, upper bound: 0.0550480
time: 0.25 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0257626, 0.0305466, -0.0174846, 0.0184832, -0.0442459, 0.0480312
1: -0.0317584, 0.0698348, -0.0177545, 0.0346891, -0.0664474, 0.0875894
2: -0.0655043, 0.0478268, -0.0469266, 0.0241803, -0.0896846, 0.0947534
3: -0.0441914, 0.0912354, -0.0308810, 0.0437489, -0.0879403, 0.1221164
4: -0.0945988, 0.0507568, -0.0604094, 0.0290925, -0.1236912, 0.1111662

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551496, upper bound: 0.0532008
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553560, upper bound: 0.0553135
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0257626, 0.0305466, -0.0169136, 0.0178243, -0.0435869, 0.0474602
1: -0.0317584, 0.0698348, -0.0172633, 0.0331967, -0.0649551, 0.0870982
2: -0.0655043, 0.0478268, -0.0456765, 0.0224354, -0.0879397, 0.0935033
3: -0.0441914, 0.0912354, -0.0304100, 0.0416375, -0.0858289, 0.1216454
4: -0.0945988, 0.0507568, -0.0582723, 0.0274961, -0.1220949, 0.1090291

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551496, upper bound: 0.0532008
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553560, upper bound: 0.0555439
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0244811, 0.0283961, -0.0174846, 0.0184832, -0.0429644, 0.0458807
1: -0.0294868, 0.0633977, -0.0177545, 0.0346891, -0.0641759, 0.0811522
2: -0.0631345, 0.0446851, -0.0469266, 0.0241803, -0.0873148, 0.0916117
3: -0.0421317, 0.0824423, -0.0308810, 0.0437489, -0.0858805, 0.1133233
4: -0.0901240, 0.0480672, -0.0604094, 0.0290925, -0.1192164, 0.1084765

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551903, upper bound: 0.0532008
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555997, upper bound: 0.0550977
time: 0.25 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0244811, 0.0283961, -0.0169136, 0.0178243, -0.0423054, 0.0453097
1: -0.0294868, 0.0633977, -0.0172633, 0.0331967, -0.0626835, 0.0806610
2: -0.0631345, 0.0446851, -0.0456765, 0.0224354, -0.0855699, 0.0903616
3: -0.0421317, 0.0824423, -0.0304100, 0.0416375, -0.0837692, 0.1128523
4: -0.0901240, 0.0480672, -0.0582723, 0.0274961, -0.1176201, 0.1063394

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551903, upper bound: 0.0532008
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555997, upper bound: 0.0551017
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0257626, 0.0305466, -0.0280088, 0.0324474, -0.0582100, 0.0585554
1: -0.0317584, 0.0698348, -0.0325964, 0.0775581, -0.1093165, 0.1024313
2: -0.0655043, 0.0478268, -0.0725141, 0.0540359, -0.1195401, 0.1203409
3: -0.0441914, 0.0912354, -0.0464773, 0.1009725, -0.1451639, 0.1377127
4: -0.0945988, 0.0507568, -0.1008744, 0.0594039, -0.1540027, 0.1516312

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551623, upper bound: 0.0532371
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553561, upper bound: 0.0553135
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0257626, 0.0305466, -0.0273379, 0.0311306, -0.0568933, 0.0578844
1: -0.0317584, 0.0698348, -0.0321509, 0.0732026, -0.1049610, 0.1019858
2: -0.0655043, 0.0478268, -0.0714296, 0.0520608, -0.1175651, 0.1192564
3: -0.0441914, 0.0912354, -0.0465936, 0.0955957, -0.1397871, 0.1378290
4: -0.0945988, 0.0507568, -0.0984768, 0.0579753, -0.1525741, 0.1492336

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551623, upper bound: 0.0536380
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553561, upper bound: 0.0555439
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0244811, 0.0283961, -0.0280088, 0.0324474, -0.0569285, 0.0564049
1: -0.0294868, 0.0633977, -0.0325964, 0.0775581, -0.1070449, 0.0959941
2: -0.0631345, 0.0446851, -0.0725141, 0.0540359, -0.1171703, 0.1171992
3: -0.0421317, 0.0824423, -0.0464773, 0.1009725, -0.1431042, 0.1289196
4: -0.0901240, 0.0480672, -0.1008744, 0.0594039, -0.1495278, 0.1489415

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552168, upper bound: 0.0532371
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555997, upper bound: 0.0550977
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0244811, 0.0283961, -0.0273379, 0.0311306, -0.0556118, 0.0557340
1: -0.0294868, 0.0633977, -0.0321509, 0.0732026, -0.1026894, 0.0955486
2: -0.0631345, 0.0446851, -0.0714296, 0.0520608, -0.1151953, 0.1161148
3: -0.0421317, 0.0824423, -0.0465936, 0.0955957, -0.1377274, 0.1290358
4: -0.0901240, 0.0480672, -0.0984768, 0.0579753, -0.1480993, 0.1465440

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552168, upper bound: 0.0535743
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555997, upper bound: 0.0551017
time: 0.22 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0257626, 0.0305466, -0.0257626, 0.0305466, -0.0563092, 0.0563092
1: -0.0317584, 0.0698348, -0.0317584, 0.0698348, -0.1015932, 0.1015932
2: -0.0655043, 0.0478268, -0.0655043, 0.0478268, -0.1133310, 0.1133310
3: -0.0441914, 0.0912354, -0.0441914, 0.0912354, -0.1354268, 0.1354268
4: -0.0945988, 0.0507568, -0.0945988, 0.0507568, -0.1453556, 0.1453556

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551617, upper bound: 0.0535113
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553233, upper bound: 0.0553110
time: 0.22 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0257626, 0.0305466, -0.0244811, 0.0283961, -0.0541587, 0.0550277
1: -0.0317584, 0.0698348, -0.0294868, 0.0633977, -0.0951561, 0.0993216
2: -0.0655043, 0.0478268, -0.0631345, 0.0446851, -0.1101894, 0.1109612
3: -0.0441914, 0.0912354, -0.0421317, 0.0824423, -0.1266336, 0.1333671
4: -0.0945988, 0.0507568, -0.0901240, 0.0480672, -0.1426659, 0.1408808

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551617, upper bound: 0.0538177
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553233, upper bound: 0.0555375
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0244811, 0.0283961, -0.0257626, 0.0305466, -0.0550277, 0.0541587
1: -0.0294868, 0.0633977, -0.0317584, 0.0698348, -0.0993216, 0.0951561
2: -0.0631345, 0.0446851, -0.0655043, 0.0478268, -0.1109612, 0.1101894
3: -0.0421317, 0.0824423, -0.0441914, 0.0912354, -0.1333671, 0.1266336
4: -0.0901240, 0.0480672, -0.0945988, 0.0507568, -0.1408808, 0.1426659

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552168, upper bound: 0.0534960
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555769, upper bound: 0.0550850
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0244811, 0.0283961, -0.0244811, 0.0283961, -0.0528772, 0.0528772
1: -0.0294868, 0.0633977, -0.0294868, 0.0633977, -0.0928845, 0.0928845
2: -0.0631345, 0.0446851, -0.0631345, 0.0446851, -0.1078196, 0.1078196
3: -0.0421317, 0.0824423, -0.0421317, 0.0824423, -0.1245739, 0.1245739
4: -0.0901240, 0.0480672, -0.0901240, 0.0480672, -0.1381911, 0.1381911

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552168, upper bound: 0.0536437
time: 0.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555769, upper bound: 0.0550850
time: 0.23 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 1.20 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0553629
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0553629
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0549071
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551574, upper bound: 0.0530275
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0555655, upper bound: 0.0551896
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551574, upper bound: 0.0535517
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0555655, upper bound: 0.0552465
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0548878, upper bound: 0.0553892
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0544692
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0548878, upper bound: 0.0556068
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0548454
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551429, upper bound: 0.0530275
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0555478, upper bound: 0.0551896
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551429, upper bound: 0.0535107
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0555478, upper bound: 0.0552465
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0549520, upper bound: 0.0529986
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551896, upper bound: 0.0553602
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0549520, upper bound: 0.0534766
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551896, upper bound: 0.0555906
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551826, upper bound: 0.0531071
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0555362, upper bound: 0.0551583
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551826, upper bound: 0.0535648
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0555362, upper bound: 0.0551583
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0549217, upper bound: 0.0529986
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551583, upper bound: 0.0553602
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0549217, upper bound: 0.0533617
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551583, upper bound: 0.0555906
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551826, upper bound: 0.0531071
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0555212, upper bound: 0.0551583
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551826, upper bound: 0.0535312
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0555212, upper bound: 0.0551583
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0554649
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0547033
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0549722, upper bound: 0.0554649
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0547033
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0550814, upper bound: 0.0532216
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0554868, upper bound: 0.0553560
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0550814, upper bound: 0.0536820
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0554868, upper bound: 0.0553614
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0548612, upper bound: 0.0535417
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0550957, upper bound: 0.0556074
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551025, upper bound: 0.0530775
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551410, upper bound: 0.0550480
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551025, upper bound: 0.0535919
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551410, upper bound: 0.0550480
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551496, upper bound: 0.0532008
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0553560, upper bound: 0.0553135
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551496, upper bound: 0.0532008
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0553560, upper bound: 0.0555439
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551903, upper bound: 0.0532008
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0555997, upper bound: 0.0550977
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551903, upper bound: 0.0532008
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0555997, upper bound: 0.0551017
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551623, upper bound: 0.0532371
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0553561, upper bound: 0.0553135
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551623, upper bound: 0.0536380
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0553561, upper bound: 0.0555439
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0552168, upper bound: 0.0532371
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0555997, upper bound: 0.0550977
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0552168, upper bound: 0.0535743
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0555997, upper bound: 0.0551017
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551617, upper bound: 0.0535113
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0553233, upper bound: 0.0553110
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0551617, upper bound: 0.0538177
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0553233, upper bound: 0.0555375
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0552168, upper bound: 0.0534960
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0555769, upper bound: 0.0550850
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0552168, upper bound: 0.0536437
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.20
Output dim: 0, lower bound: -0.0555769, upper bound: 0.0550850

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0188328, 0.0191179, -0.0174846, 0.0184832, -0.0373160, 0.0366025
1: -0.0185345, 0.0336902, -0.0177545, 0.0346891, -0.0532236, 0.0514447
2: -0.0465600, 0.0245035, -0.0469266, 0.0241803, -0.0707403, 0.0714301
3: -0.0314490, 0.0414548, -0.0308810, 0.0437489, -0.0751979, 0.0723358
4: -0.0564701, 0.0289175, -0.0604094, 0.0290925, -0.0855626, 0.0893269

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
time: 0.19 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0188328, 0.0191179, -0.0169136, 0.0178243, -0.0366571, 0.0360315
1: -0.0185345, 0.0336902, -0.0172633, 0.0331967, -0.0517312, 0.0509535
2: -0.0465600, 0.0245035, -0.0456765, 0.0224354, -0.0689954, 0.0701800
3: -0.0314490, 0.0414548, -0.0304100, 0.0416375, -0.0730866, 0.0718648
4: -0.0564701, 0.0289175, -0.0582723, 0.0274961, -0.0839663, 0.0871897

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549208, upper bound: 0.0548114
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549208, upper bound: 0.0549071
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0174846, 0.0184832, -0.0347834, 0.0345879
1: -0.0163531, 0.0312095, -0.0177545, 0.0346891, -0.0510422, 0.0489641
2: -0.0443946, 0.0211404, -0.0469266, 0.0241803, -0.0685749, 0.0680670
3: -0.0292436, 0.0388854, -0.0308810, 0.0437489, -0.0729924, 0.0697664
4: -0.0566717, 0.0260474, -0.0604094, 0.0290925, -0.0857642, 0.0864568

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555965, upper bound: 0.0549181
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547940, upper bound: 0.0548758
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0169136, 0.0178243, -0.0341244, 0.0340169
1: -0.0163531, 0.0312095, -0.0172633, 0.0331967, -0.0495499, 0.0484729
2: -0.0443946, 0.0211404, -0.0456765, 0.0224354, -0.0668300, 0.0668169
3: -0.0292436, 0.0388854, -0.0304100, 0.0416375, -0.0708811, 0.0692954
4: -0.0566717, 0.0260474, -0.0582723, 0.0274961, -0.0841679, 0.0843197

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536317, upper bound: 0.0549928
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0536317, upper bound: 0.0552465
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0188328, 0.0191179, -0.0280088, 0.0324474, -0.0512802, 0.0471267
1: -0.0185345, 0.0336902, -0.0325964, 0.0775581, -0.0960927, 0.0662866
2: -0.0465600, 0.0245035, -0.0725141, 0.0540359, -0.1005959, 0.0970176
3: -0.0314490, 0.0414548, -0.0464773, 0.1009725, -0.1324215, 0.0879321
4: -0.0564701, 0.0289175, -0.1008744, 0.0594039, -0.1158740, 0.1297919

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0544692
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0544692
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0188328, 0.0191179, -0.0273379, 0.0311306, -0.0499635, 0.0464557
1: -0.0185345, 0.0336902, -0.0321509, 0.0732026, -0.0917371, 0.0658411
2: -0.0465600, 0.0245035, -0.0714296, 0.0520608, -0.0986208, 0.0959332
3: -0.0314490, 0.0414548, -0.0465936, 0.0955957, -0.1270447, 0.0880484
4: -0.0564701, 0.0289175, -0.0984768, 0.0579753, -0.1144455, 0.1273943

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549551, upper bound: 0.0545802
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549551, upper bound: 0.0548454
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0280088, 0.0324474, -0.0487475, 0.0451121
1: -0.0163531, 0.0312095, -0.0325964, 0.0775581, -0.0939113, 0.0638060
2: -0.0443946, 0.0211404, -0.0725141, 0.0540359, -0.0984305, 0.0936545
3: -0.0292436, 0.0388854, -0.0464773, 0.1009725, -0.1302161, 0.0853627
4: -0.0566717, 0.0260474, -0.1008744, 0.0594039, -0.1160756, 0.1269218

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534766, upper bound: 0.0549520
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0534766, upper bound: 0.0551896
time: 0.24 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0273379, 0.0311306, -0.0474308, 0.0444412
1: -0.0163531, 0.0312095, -0.0321509, 0.0732026, -0.0895557, 0.0633605
2: -0.0443946, 0.0211404, -0.0714296, 0.0520608, -0.0964554, 0.0925701
3: -0.0292436, 0.0388854, -0.0465936, 0.0955957, -0.1248393, 0.0854789
4: -0.0566717, 0.0260474, -0.0984768, 0.0579753, -0.1146471, 0.1245242

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536523, upper bound: 0.0550097
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0536523, upper bound: 0.0552465
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0275401, 0.0318847, -0.0174846, 0.0184832, -0.0460233, 0.0493693
1: -0.0315660, 0.0759994, -0.0177545, 0.0346891, -0.0662550, 0.0937540
2: -0.0714210, 0.0530449, -0.0469266, 0.0241803, -0.0956013, 0.0999715
3: -0.0452617, 0.0986971, -0.0308810, 0.0437489, -0.0890106, 0.1295781
4: -0.0993945, 0.0582935, -0.0604094, 0.0290925, -0.1284870, 0.1187029

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553710, upper bound: 0.0548459
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543515, upper bound: 0.0548036
time: 0.20 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0275401, 0.0318847, -0.0169136, 0.0178243, -0.0453644, 0.0487983
1: -0.0315660, 0.0759994, -0.0172633, 0.0331967, -0.0647627, 0.0932628
2: -0.0714210, 0.0530449, -0.0456765, 0.0224354, -0.0938564, 0.0987214
3: -0.0452617, 0.0986971, -0.0304100, 0.0416375, -0.0868993, 0.1291071
4: -0.0993945, 0.0582935, -0.0582723, 0.0274961, -0.1268907, 0.1165658

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0530275, upper bound: 0.0551429
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0530275, upper bound: 0.0555906
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0194715, 0.0185953, -0.0174846, 0.0184832, -0.0379547, 0.0360800
1: -0.0236661, 0.0364224, -0.0177545, 0.0346891, -0.0583552, 0.0541769
2: -0.0416546, 0.0217706, -0.0469266, 0.0241803, -0.0658349, 0.0686972
3: -0.0353903, 0.0438848, -0.0308810, 0.0437489, -0.0791392, 0.0747658
4: -0.0484835, 0.0241966, -0.0604094, 0.0290925, -0.0775760, 0.0846060

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551584, upper bound: 0.0530663
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544608, upper bound: 0.0530587
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0268415, 0.0305504, -0.0174846, 0.0184832, -0.0453247, 0.0480350
1: -0.0310880, 0.0714282, -0.0177545, 0.0346891, -0.0657771, 0.0891827
2: -0.0703224, 0.0510674, -0.0469266, 0.0241803, -0.0945028, 0.0979940
3: -0.0453250, 0.0927791, -0.0308810, 0.0437489, -0.0890739, 0.1236601
4: -0.0969780, 0.0568669, -0.0604094, 0.0290925, -0.1260705, 0.1172763

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555965, upper bound: 0.0549521
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547199, upper bound: 0.0549097
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0194715, 0.0185953, -0.0169136, 0.0178243, -0.0372958, 0.0355090
1: -0.0236661, 0.0364224, -0.0172633, 0.0331967, -0.0568628, 0.0536857
2: -0.0416546, 0.0217706, -0.0456765, 0.0224354, -0.0640900, 0.0674471
3: -0.0353903, 0.0438848, -0.0304100, 0.0416375, -0.0770278, 0.0742948
4: -0.0484835, 0.0241966, -0.0582723, 0.0274961, -0.0759797, 0.0824689

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533084, upper bound: 0.0533208
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533084, upper bound: 0.0535648
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0268415, 0.0305504, -0.0169136, 0.0178243, -0.0446657, 0.0474641
1: -0.0310880, 0.0714282, -0.0172633, 0.0331967, -0.0642847, 0.0886915
2: -0.0703224, 0.0510674, -0.0456765, 0.0224354, -0.0927578, 0.0967439
3: -0.0453250, 0.0927791, -0.0304100, 0.0416375, -0.0869625, 0.1231891
4: -0.0969780, 0.0568669, -0.0582723, 0.0274961, -0.1244742, 0.1151392

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0548581
time: 0.25 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0548581
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0275401, 0.0318847, -0.0280088, 0.0324474, -0.0599875, 0.0598935
1: -0.0315660, 0.0759994, -0.0325964, 0.0775581, -0.1091241, 0.1085959
2: -0.0714210, 0.0530449, -0.0725141, 0.0540359, -0.1254568, 0.1255590
3: -0.0452617, 0.0986971, -0.0464773, 0.1009725, -0.1462342, 0.1451744
4: -0.0993945, 0.0582935, -0.1008744, 0.0594039, -0.1587984, 0.1591679

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529209, upper bound: 0.0550748
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0529209, upper bound: 0.0553602
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0275401, 0.0318847, -0.0273379, 0.0311306, -0.0586707, 0.0592225
1: -0.0315660, 0.0759994, -0.0321509, 0.0732026, -0.1047685, 0.1081504
2: -0.0714210, 0.0530449, -0.0714296, 0.0520608, -0.1234818, 0.1244745
3: -0.0452617, 0.0986971, -0.0465936, 0.0955957, -0.1408574, 0.1452906
4: -0.0993945, 0.0582935, -0.0984768, 0.0579753, -0.1573699, 0.1567703

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0531053, upper bound: 0.0551463
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0531053, upper bound: 0.0555906
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0194715, 0.0185953, -0.0280088, 0.0324474, -0.0519189, 0.0466041
1: -0.0236661, 0.0364224, -0.0325964, 0.0775581, -0.1012243, 0.0690188
2: -0.0416546, 0.0217706, -0.0725141, 0.0540359, -0.0956904, 0.0942847
3: -0.0353903, 0.0438848, -0.0464773, 0.1009725, -0.1363628, 0.0903621
4: -0.0484835, 0.0241966, -0.1008744, 0.0594039, -0.1078874, 0.1250710

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529050, upper bound: 0.0527752
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529050, upper bound: 0.0531071
time: 0.24 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0268415, 0.0305504, -0.0280088, 0.0324474, -0.0592889, 0.0585592
1: -0.0310880, 0.0714282, -0.0325964, 0.0775581, -0.1086462, 0.1040246
2: -0.0703224, 0.0510674, -0.0725141, 0.0540359, -0.1243583, 0.1235815
3: -0.0453250, 0.0927791, -0.0464773, 0.1009725, -0.1462975, 0.1392564
4: -0.0969780, 0.0568669, -0.1008744, 0.0594039, -0.1563819, 0.1577413

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529209, upper bound: 0.0539374
time: 0.25 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529364, upper bound: 0.0551583
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0194715, 0.0185953, -0.0273379, 0.0311306, -0.0506021, 0.0459332
1: -0.0236661, 0.0364224, -0.0321509, 0.0732026, -0.0968687, 0.0685733
2: -0.0416546, 0.0217706, -0.0714296, 0.0520608, -0.0937154, 0.0932002
3: -0.0353903, 0.0438848, -0.0465936, 0.0955957, -0.1309860, 0.0904784
4: -0.0484835, 0.0241966, -0.0984768, 0.0579753, -0.1064588, 0.1226734

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533204, upper bound: 0.0533208
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533204, upper bound: 0.0535312
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0268415, 0.0305504, -0.0273379, 0.0311306, -0.0579721, 0.0578883
1: -0.0310880, 0.0714282, -0.0321509, 0.0732026, -0.1042906, 0.1035791
2: -0.0703224, 0.0510674, -0.0714296, 0.0520608, -0.1223832, 0.1224970
3: -0.0453250, 0.0927791, -0.0465936, 0.0955957, -0.1409207, 0.1393727
4: -0.0969780, 0.0568669, -0.0984768, 0.0579753, -0.1549534, 0.1553438

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0548593
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0551583
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0188328, 0.0191179, -0.0257626, 0.0305466, -0.0493794, 0.0448805
1: -0.0185345, 0.0336902, -0.0317584, 0.0698348, -0.0883694, 0.0654485
2: -0.0465600, 0.0245035, -0.0655043, 0.0478268, -0.0943867, 0.0900078
3: -0.0314490, 0.0414548, -0.0441914, 0.0912354, -0.1226845, 0.0856462
4: -0.0564701, 0.0289175, -0.0945988, 0.0507568, -0.1072270, 0.1235162

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0544686
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0547033
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0188328, 0.0191179, -0.0244811, 0.0283961, -0.0472289, 0.0435990
1: -0.0185345, 0.0336902, -0.0294868, 0.0633977, -0.0819322, 0.0631769
2: -0.0465600, 0.0245035, -0.0631345, 0.0446851, -0.0912451, 0.0876380
3: -0.0314490, 0.0414548, -0.0421317, 0.0824423, -0.1138913, 0.0835865
4: -0.0564701, 0.0289175, -0.0901240, 0.0480672, -0.1045373, 0.1190414

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0540290, upper bound: 0.0548070
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549663, upper bound: 0.0555851
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0257626, 0.0305466, -0.0468467, 0.0428659
1: -0.0163531, 0.0312095, -0.0317584, 0.0698348, -0.0861880, 0.0629679
2: -0.0443946, 0.0211404, -0.0655043, 0.0478268, -0.0922213, 0.0866447
3: -0.0292436, 0.0388854, -0.0441914, 0.0912354, -0.1204790, 0.0830768
4: -0.0566717, 0.0260474, -0.0945988, 0.0507568, -0.1074286, 0.1206462

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537091, upper bound: 0.0551498
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0537091, upper bound: 0.0553560
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0244811, 0.0283961, -0.0446962, 0.0415844
1: -0.0163531, 0.0312095, -0.0294868, 0.0633977, -0.0797508, 0.0606963
2: -0.0443946, 0.0211404, -0.0631345, 0.0446851, -0.0890797, 0.0842749
3: -0.0292436, 0.0388854, -0.0421317, 0.0824423, -0.1116858, 0.0810170
4: -0.0566717, 0.0260474, -0.0901240, 0.0480672, -0.1047389, 0.1161713

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0537518, upper bound: 0.0551730
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0537518, upper bound: 0.0553614
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0275401, 0.0318847, -0.0244811, 0.0283961, -0.0559362, 0.0563658
1: -0.0315660, 0.0759994, -0.0294868, 0.0633977, -0.0949637, 0.1054862
2: -0.0714210, 0.0530449, -0.0631345, 0.0446851, -0.1161061, 0.1161793
3: -0.0452617, 0.0986971, -0.0421317, 0.0824423, -0.1277040, 0.1408287
4: -0.0993945, 0.0582935, -0.0901240, 0.0480672, -0.1474617, 0.1484174

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0531259, upper bound: 0.0551390
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0531259, upper bound: 0.0556074
time: 0.21 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0251495, 0.0299018, -0.0174846, 0.0184832, -0.0436327, 0.0473865
1: -0.0308103, 0.0680934, -0.0177545, 0.0346891, -0.0654994, 0.0858479
2: -0.0643693, 0.0467766, -0.0469266, 0.0241803, -0.0885496, 0.0937032
3: -0.0430202, 0.0885118, -0.0308810, 0.0437489, -0.0867691, 0.1193927
4: -0.0929886, 0.0495827, -0.0604094, 0.0290925, -0.1220811, 0.1099921

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554526, upper bound: 0.0549449
time: 0.25 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545959, upper bound: 0.0549026
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0251495, 0.0299018, -0.0169136, 0.0178243, -0.0429737, 0.0468155
1: -0.0308103, 0.0680934, -0.0172633, 0.0331967, -0.0640071, 0.0853567
2: -0.0643693, 0.0467766, -0.0456765, 0.0224354, -0.0868047, 0.0924531
3: -0.0430202, 0.0885118, -0.0304100, 0.0416375, -0.0846577, 0.1189218
4: -0.0929886, 0.0495827, -0.0582723, 0.0274961, -0.1204848, 0.1078550

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0532216, upper bound: 0.0550814
time: 0.24 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0532216, upper bound: 0.0555439
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0216623, 0.0203123, -0.0174846, 0.0184832, -0.0401455, 0.0377969
1: -0.0303493, 0.0405147, -0.0177545, 0.0346891, -0.0650384, 0.0582693
2: -0.0427012, 0.0220349, -0.0469266, 0.0241803, -0.0668816, 0.0689615
3: -0.0446583, 0.0515581, -0.0308810, 0.0437489, -0.0884071, 0.0824391
4: -0.0529711, 0.0229706, -0.0604094, 0.0290925, -0.0820636, 0.0833800

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551433, upper bound: 0.0531656
time: 0.24 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545161, upper bound: 0.0531652
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0238435, 0.0277011, -0.0174846, 0.0184832, -0.0423268, 0.0451857
1: -0.0284588, 0.0615482, -0.0177545, 0.0346891, -0.0631479, 0.0793027
2: -0.0619252, 0.0435389, -0.0469266, 0.0241803, -0.0861056, 0.0904655
3: -0.0407595, 0.0795807, -0.0308810, 0.0437489, -0.0845083, 0.1104617
4: -0.0884067, 0.0467996, -0.0604094, 0.0290925, -0.1174992, 0.1072090

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556054, upper bound: 0.0549521
time: 0.24 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548619, upper bound: 0.0549097
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0216623, 0.0203123, -0.0169136, 0.0178243, -0.0394866, 0.0372259
1: -0.0303493, 0.0405147, -0.0172633, 0.0331967, -0.0635460, 0.0577781
2: -0.0427012, 0.0220349, -0.0456765, 0.0224354, -0.0651366, 0.0677114
3: -0.0446583, 0.0515581, -0.0304100, 0.0416375, -0.0862958, 0.0819681
4: -0.0529711, 0.0229706, -0.0582723, 0.0274961, -0.0804673, 0.0812429

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533081, upper bound: 0.0533309
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533081, upper bound: 0.0535778
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0238435, 0.0277011, -0.0169136, 0.0178243, -0.0416678, 0.0446147
1: -0.0284588, 0.0615482, -0.0172633, 0.0331967, -0.0616555, 0.0788115
2: -0.0619252, 0.0435389, -0.0456765, 0.0224354, -0.0843606, 0.0892154
3: -0.0407595, 0.0795807, -0.0304100, 0.0416375, -0.0823970, 0.1099907
4: -0.0884067, 0.0467996, -0.0582723, 0.0274961, -0.1159028, 0.1050719

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536906, upper bound: 0.0548305
time: 0.24 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536906, upper bound: 0.0551017
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0235327, 0.0224063, -0.0280088, 0.0324474, -0.0559801, 0.0504151
1: -0.0340902, 0.0474932, -0.0325964, 0.0775581, -0.1116484, 0.0800896
2: -0.0453555, 0.0251679, -0.0725141, 0.0540359, -0.0993913, 0.0976820
3: -0.0497409, 0.0613024, -0.0464773, 0.1009725, -0.1507134, 0.1077797
4: -0.0578012, 0.0258172, -0.1008744, 0.0594039, -0.1172051, 0.1266916

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0531202, upper bound: 0.0530382
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0531202, upper bound: 0.0532371
time: 0.22 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0251495, 0.0299018, -0.0280088, 0.0324474, -0.0575969, 0.0579106
1: -0.0308103, 0.0680934, -0.0325964, 0.0775581, -0.1083685, 0.1006898
2: -0.0643693, 0.0467766, -0.0725141, 0.0540359, -0.1184051, 0.1192907
3: -0.0430202, 0.0885118, -0.0464773, 0.1009725, -0.1439927, 0.1349890
4: -0.0929886, 0.0495827, -0.1008744, 0.0594039, -0.1523925, 0.1504571

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0532526, upper bound: 0.0550737
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0532526, upper bound: 0.0553135
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0235327, 0.0224063, -0.0273379, 0.0311306, -0.0546634, 0.0497441
1: -0.0340902, 0.0474932, -0.0321509, 0.0732026, -0.1072928, 0.0796441
2: -0.0453555, 0.0251679, -0.0714296, 0.0520608, -0.0974163, 0.0965975
3: -0.0497409, 0.0613024, -0.0465936, 0.0955957, -0.1453366, 0.1078959
4: -0.0578012, 0.0258172, -0.0984768, 0.0579753, -0.1157765, 0.1242940

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0532564, upper bound: 0.0533752
time: 0.24 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0532564, upper bound: 0.0536380
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0251495, 0.0299018, -0.0273379, 0.0311306, -0.0562801, 0.0572397
1: -0.0308103, 0.0680934, -0.0321509, 0.0732026, -0.1040129, 0.1002443
2: -0.0643693, 0.0467766, -0.0714296, 0.0520608, -0.1164301, 0.1182063
3: -0.0430202, 0.0885118, -0.0465936, 0.0955957, -0.1386159, 0.1351053
4: -0.0929886, 0.0495827, -0.0984768, 0.0579753, -0.1509640, 0.1480595

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533468, upper bound: 0.0551351
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0533468, upper bound: 0.0555439
time: 0.26 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0216623, 0.0203123, -0.0280088, 0.0324474, -0.0541097, 0.0483211
1: -0.0303493, 0.0405147, -0.0325964, 0.0775581, -0.1079075, 0.0731112
2: -0.0427012, 0.0220349, -0.0725141, 0.0540359, -0.0967371, 0.0945490
3: -0.0446583, 0.0515581, -0.0464773, 0.1009725, -0.1456308, 0.0980354
4: -0.0529711, 0.0229706, -0.1008744, 0.0594039, -0.1123750, 0.1238450

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0532387, upper bound: 0.0530382
time: 0.24 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0532387, upper bound: 0.0532371
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0238435, 0.0277011, -0.0280088, 0.0324474, -0.0562909, 0.0557099
1: -0.0284588, 0.0615482, -0.0325964, 0.0775581, -0.1060170, 0.0941446
2: -0.0619252, 0.0435389, -0.0725141, 0.0540359, -0.1159611, 0.1160530
3: -0.0407595, 0.0795807, -0.0464773, 0.1009725, -0.1417320, 0.1260580
4: -0.0884067, 0.0467996, -0.1008744, 0.0594039, -0.1478106, 0.1476740

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535415, upper bound: 0.0547894
time: 0.24 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535415, upper bound: 0.0550977
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0216623, 0.0203123, -0.0273379, 0.0311306, -0.0527930, 0.0476502
1: -0.0303493, 0.0405147, -0.0321509, 0.0732026, -0.1035519, 0.0726657
2: -0.0427012, 0.0220349, -0.0714296, 0.0520608, -0.0947620, 0.0934645
3: -0.0446583, 0.0515581, -0.0465936, 0.0955957, -0.1402540, 0.0981517
4: -0.0529711, 0.0229706, -0.0984768, 0.0579753, -0.1109465, 0.1214474

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533350, upper bound: 0.0533309
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533350, upper bound: 0.0535743
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0238435, 0.0277011, -0.0273379, 0.0311306, -0.0549742, 0.0550389
1: -0.0284588, 0.0615482, -0.0321509, 0.0732026, -0.1016614, 0.0936991
2: -0.0619252, 0.0435389, -0.0714296, 0.0520608, -0.1139860, 0.1149686
3: -0.0407595, 0.0795807, -0.0465936, 0.0955957, -0.1363552, 0.1261743
4: -0.0884067, 0.0467996, -0.0984768, 0.0579753, -0.1463820, 0.1452764

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537111, upper bound: 0.0548559
time: 0.25 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537111, upper bound: 0.0551017
time: 0.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0235327, 0.0224063, -0.0257626, 0.0305466, -0.0540793, 0.0481689
1: -0.0340902, 0.0474932, -0.0317584, 0.0698348, -0.1039251, 0.0792516
2: -0.0453555, 0.0251679, -0.0655043, 0.0478268, -0.0931822, 0.0906722
3: -0.0497409, 0.0613024, -0.0441914, 0.0912354, -0.1409764, 0.1054938
4: -0.0578012, 0.0258172, -0.0945988, 0.0507568, -0.1085580, 0.1204160

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533946, upper bound: 0.0533964
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533946, upper bound: 0.0535113
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0251495, 0.0299018, -0.0257626, 0.0305466, -0.0556960, 0.0556645
1: -0.0308103, 0.0680934, -0.0317584, 0.0698348, -0.1006452, 0.0998518
2: -0.0643693, 0.0467766, -0.0655043, 0.0478268, -0.1121960, 0.1122809
3: -0.0430202, 0.0885118, -0.0441914, 0.0912354, -0.1342556, 0.1327031
4: -0.0929886, 0.0495827, -0.0945988, 0.0507568, -0.1437455, 0.1441815

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0550673
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0553110
time: 0.22 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0235327, 0.0224063, -0.0244811, 0.0283961, -0.0519288, 0.0468874
1: -0.0340902, 0.0474932, -0.0294868, 0.0633977, -0.0974879, 0.0769800
2: -0.0453555, 0.0251679, -0.0631345, 0.0446851, -0.0900406, 0.0883024
3: -0.0497409, 0.0613024, -0.0421317, 0.0824423, -0.1321832, 0.1034340
4: -0.0578012, 0.0258172, -0.0901240, 0.0480672, -0.1058684, 0.1159412

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533961, upper bound: 0.0535150
time: 0.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533961, upper bound: 0.0538177
time: 0.22 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0251495, 0.0299018, -0.0244811, 0.0283961, -0.0535456, 0.0543830
1: -0.0308103, 0.0680934, -0.0294868, 0.0633977, -0.0942080, 0.0975802
2: -0.0643693, 0.0467766, -0.0631345, 0.0446851, -0.1090544, 0.1099111
3: -0.0430202, 0.0885118, -0.0421317, 0.0824423, -0.1254625, 0.1306434
4: -0.0929886, 0.0495827, -0.0901240, 0.0480672, -0.1410558, 0.1397066

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0551351
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0555375
time: 0.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0216623, 0.0203123, -0.0257626, 0.0305466, -0.0522089, 0.0460749
1: -0.0303493, 0.0405147, -0.0317584, 0.0698348, -0.1001841, 0.0722731
2: -0.0427012, 0.0220349, -0.0655043, 0.0478268, -0.0905280, 0.0875391
3: -0.0446583, 0.0515581, -0.0441914, 0.0912354, -0.1358937, 0.0957495
4: -0.0529711, 0.0229706, -0.0945988, 0.0507568, -0.1037280, 0.1175694

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534448, upper bound: 0.0533964
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534448, upper bound: 0.0534960
time: 0.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0238435, 0.0277011, -0.0257626, 0.0305466, -0.0543901, 0.0534637
1: -0.0284588, 0.0615482, -0.0317584, 0.0698348, -0.0982937, 0.0933066
2: -0.0619252, 0.0435389, -0.0655043, 0.0478268, -0.1097520, 0.1090432
3: -0.0407595, 0.0795807, -0.0441914, 0.0912354, -0.1319949, 0.1237721
4: -0.0884067, 0.0467996, -0.0945988, 0.0507568, -0.1391635, 0.1413983

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536540, upper bound: 0.0547787
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536540, upper bound: 0.0550850
time: 0.23 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0216623, 0.0203123, -0.0244811, 0.0283961, -0.0500584, 0.0447934
1: -0.0303493, 0.0405147, -0.0294868, 0.0633977, -0.0937470, 0.0700015
2: -0.0427012, 0.0220349, -0.0631345, 0.0446851, -0.0873864, 0.0851693
3: -0.0446583, 0.0515581, -0.0421317, 0.0824423, -0.1271005, 0.0936898
4: -0.0529711, 0.0229706, -0.0901240, 0.0480672, -0.1010383, 0.1130946

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535029, upper bound: 0.0535047
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535029, upper bound: 0.0536437
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0238435, 0.0277011, -0.0244811, 0.0283961, -0.0522396, 0.0521822
1: -0.0284588, 0.0615482, -0.0294868, 0.0633977, -0.0918565, 0.0910350
2: -0.0619252, 0.0435389, -0.0631345, 0.0446851, -0.1066104, 0.1066734
3: -0.0407595, 0.0795807, -0.0421317, 0.0824423, -0.1232017, 0.1217124
4: -0.0884067, 0.0467996, -0.0901240, 0.0480672, -0.1364738, 0.1369235

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537749, upper bound: 0.0547965
time: 0.29 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537749, upper bound: 0.0550850
time: 0.24 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 1.36 seconds
NS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
NS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0544429, upper bound: 0.0544429
NS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0549208, upper bound: 0.0548114
NS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0549208, upper bound: 0.0549071
NS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0555965, upper bound: 0.0549181
NS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0547940, upper bound: 0.0548758
NS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0536317, upper bound: 0.0549928
NS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0536317, upper bound: 0.0552465
NS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0544692
NS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0544692
NS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0549551, upper bound: 0.0545802
NS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0549551, upper bound: 0.0548454
NS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0534766, upper bound: 0.0549520
NS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0534766, upper bound: 0.0551896
NS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0536523, upper bound: 0.0550097
NS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0536523, upper bound: 0.0552465
NS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0553710, upper bound: 0.0548459
NS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0543515, upper bound: 0.0548036
NS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0530275, upper bound: 0.0551429
NS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0530275, upper bound: 0.0555906
NS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0551584, upper bound: 0.0530663
NS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0544608, upper bound: 0.0530587
NS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0555965, upper bound: 0.0549521
NS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0547199, upper bound: 0.0549097
NS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0533084, upper bound: 0.0533208
NS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0533084, upper bound: 0.0535648
NS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0548581
NS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0548581
NS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0529209, upper bound: 0.0550748
NS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0529209, upper bound: 0.0553602
NS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0531053, upper bound: 0.0551463
NS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0531053, upper bound: 0.0555906
NS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0529050, upper bound: 0.0527752
NS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0529050, upper bound: 0.0531071
NS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0529209, upper bound: 0.0539374
NS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0529364, upper bound: 0.0551583
NS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0533204, upper bound: 0.0533208
NS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0533204, upper bound: 0.0535312
NS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0548593
NS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0534829, upper bound: 0.0551583
NS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0544686
NS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0549439, upper bound: 0.0547033
NS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0540290, upper bound: 0.0548070
NS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0549663, upper bound: 0.0555851
NS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0537091, upper bound: 0.0551498
NS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0537091, upper bound: 0.0553560
NS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0537518, upper bound: 0.0551730
NS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0537518, upper bound: 0.0553614
NS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0531259, upper bound: 0.0551390
NS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0531259, upper bound: 0.0556074
NS_A2_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0554526, upper bound: 0.0549449
NS_A2_B1_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0545959, upper bound: 0.0549026
NS_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0532216, upper bound: 0.0550814
NS_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0532216, upper bound: 0.0555439
NS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0551433, upper bound: 0.0531656
NS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0545161, upper bound: 0.0531652
NS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0556054, upper bound: 0.0549521
NS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0548619, upper bound: 0.0549097
NS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0533081, upper bound: 0.0533309
NS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0533081, upper bound: 0.0535778
NS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0536906, upper bound: 0.0548305
NS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0536906, upper bound: 0.0551017
NS_A2_B1_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0531202, upper bound: 0.0530382
NS_A2_B1_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0531202, upper bound: 0.0532371
NS_A2_B1_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0532526, upper bound: 0.0550737
NS_A2_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0532526, upper bound: 0.0553135
NS_A2_B1_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0532564, upper bound: 0.0533752
NS_A2_B1_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0532564, upper bound: 0.0536380
NS_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0533468, upper bound: 0.0551351
NS_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0533468, upper bound: 0.0555439
NS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0532387, upper bound: 0.0530382
NS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0532387, upper bound: 0.0532371
NS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0535415, upper bound: 0.0547894
NS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0535415, upper bound: 0.0550977
NS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0533350, upper bound: 0.0533309
NS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0533350, upper bound: 0.0535743
NS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0537111, upper bound: 0.0548559
NS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0537111, upper bound: 0.0551017
NS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0533946, upper bound: 0.0533964
NS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0533946, upper bound: 0.0535113
NS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0550673
NS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0553110
NS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0533961, upper bound: 0.0535150
NS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0533961, upper bound: 0.0538177
NS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0551351
NS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0535289, upper bound: 0.0555375
NS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0534448, upper bound: 0.0533964
NS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0534448, upper bound: 0.0534960
NS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0536540, upper bound: 0.0547787
NS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0536540, upper bound: 0.0550850
NS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0535029, upper bound: 0.0535047
NS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0535029, upper bound: 0.0536437
NS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0537749, upper bound: 0.0547965
NS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 1.36
Output dim: 0, lower bound: -0.0537749, upper bound: 0.0550850

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0188328, 0.0191179, -0.0354180, 0.0359361
1: -0.0163531, 0.0312095, -0.0185345, 0.0336902, -0.0500433, 0.0497441
2: -0.0443946, 0.0211404, -0.0465600, 0.0245035, -0.0688981, 0.0677004
3: -0.0292436, 0.0388854, -0.0314490, 0.0414548, -0.0706984, 0.0703344
4: -0.0566717, 0.0260474, -0.0564701, 0.0289175, -0.0855892, 0.0825175

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547040, upper bound: 0.0548758
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547040, upper bound: 0.0548758
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0163001, 0.0171033, -0.0334034, 0.0334034
1: -0.0163531, 0.0312095, -0.0163531, 0.0312095, -0.0475627, 0.0475627
2: -0.0443946, 0.0211404, -0.0443946, 0.0211404, -0.0655350, 0.0655350
3: -0.0292436, 0.0388854, -0.0292436, 0.0388854, -0.0681289, 0.0681289
4: -0.0566717, 0.0260474, -0.0566717, 0.0260474, -0.0827191, 0.0827191

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0535063, upper bound: 0.0552465
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536153, upper bound: 0.0551117
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0275401, 0.0318847, -0.0481848, 0.0446434
1: -0.0163531, 0.0312095, -0.0315660, 0.0759994, -0.0923526, 0.0627755
2: -0.0443946, 0.0211404, -0.0714210, 0.0530449, -0.0974395, 0.0925614
3: -0.0292436, 0.0388854, -0.0452617, 0.0986971, -0.1279406, 0.0841471
4: -0.0566717, 0.0260474, -0.0993945, 0.0582935, -0.1149652, 0.1254419

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0533088, upper bound: 0.0551844
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0534154, upper bound: 0.0549130
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0268415, 0.0305504, -0.0468506, 0.0439448
1: -0.0163531, 0.0312095, -0.0310880, 0.0714282, -0.0877813, 0.0622975
2: -0.0443946, 0.0211404, -0.0703224, 0.0510674, -0.0954619, 0.0914629
3: -0.0292436, 0.0388854, -0.0453250, 0.0927791, -0.1220227, 0.0842104
4: -0.0566717, 0.0260474, -0.0969780, 0.0568669, -0.1135387, 0.1230254

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0535228, upper bound: 0.0552465
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536319, upper bound: 0.0551117
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0275401, 0.0318847, -0.0188328, 0.0191179, -0.0466579, 0.0507175
1: -0.0315660, 0.0759994, -0.0185345, 0.0336902, -0.0652561, 0.0945340
2: -0.0714210, 0.0530449, -0.0465600, 0.0245035, -0.0959245, 0.0996048
3: -0.0452617, 0.0986971, -0.0314490, 0.0414548, -0.0867165, 0.1301461
4: -0.0993945, 0.0582935, -0.0564701, 0.0289175, -0.1283120, 0.1147636

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543515, upper bound: 0.0548036
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543515, upper bound: 0.0548036
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0275401, 0.0318847, -0.0163001, 0.0171033, -0.0446434, 0.0481848
1: -0.0315660, 0.0759994, -0.0163531, 0.0312095, -0.0627755, 0.0923526
2: -0.0714210, 0.0530449, -0.0443946, 0.0211404, -0.0925614, 0.0974395
3: -0.0452617, 0.0986971, -0.0292436, 0.0388854, -0.0841471, 0.1279406
4: -0.0993945, 0.0582935, -0.0566717, 0.0260474, -0.1254419, 0.1149652

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529876, upper bound: 0.0550998
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0529876, upper bound: 0.0552509
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0268415, 0.0305504, -0.0188328, 0.0191179, -0.0459593, 0.0493832
1: -0.0310880, 0.0714282, -0.0185345, 0.0336902, -0.0647782, 0.0899627
2: -0.0703224, 0.0510674, -0.0465600, 0.0245035, -0.0948260, 0.0976273
3: -0.0453250, 0.0927791, -0.0314490, 0.0414548, -0.0867798, 0.1242281
4: -0.0969780, 0.0568669, -0.0564701, 0.0289175, -0.1258955, 0.1133371

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544728, upper bound: 0.0549097
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544728, upper bound: 0.0549097
time: 0.24 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0275401, 0.0318847, -0.0275401, 0.0318847, -0.0594247, 0.0594247
1: -0.0315660, 0.0759994, -0.0315660, 0.0759994, -0.1075654, 0.1075654
2: -0.0714210, 0.0530449, -0.0714210, 0.0530449, -0.1244658, 0.1244658
3: -0.0452617, 0.0986971, -0.0452617, 0.0986971, -0.1439588, 0.1439588
4: -0.0993945, 0.0582935, -0.0993945, 0.0582935, -0.1576880, 0.1576880

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0528588, upper bound: 0.0553265
time: 0.25 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0528586, upper bound: 0.0548381
time: 0.25 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0275401, 0.0318847, -0.0268415, 0.0305504, -0.0580905, 0.0587261
1: -0.0315660, 0.0759994, -0.0310880, 0.0714282, -0.1029941, 0.1070874
2: -0.0714210, 0.0530449, -0.0703224, 0.0510674, -0.1224883, 0.1233673
3: -0.0452617, 0.0986971, -0.0453250, 0.0927791, -0.1380408, 0.1440221
4: -0.0993945, 0.0582935, -0.0969780, 0.0568669, -0.1562615, 0.1552715

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0530645, upper bound: 0.0555821
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0530645, upper bound: 0.0552091
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0188328, 0.0191179, -0.0241868, 0.0279991, -0.0468319, 0.0433047
1: -0.0185345, 0.0336902, -0.0289310, 0.0624349, -0.0809695, 0.0626212
2: -0.0465600, 0.0245035, -0.0625475, 0.0440468, -0.0906067, 0.0870510
3: -0.0314490, 0.0414548, -0.0414982, 0.0810555, -0.1125045, 0.0829530
4: -0.0564701, 0.0289175, -0.0892359, 0.0473967, -0.1038668, 0.1181533

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0251495, 0.0299018, -0.0462020, 0.0422528
1: -0.0163531, 0.0312095, -0.0308103, 0.0680934, -0.0844465, 0.0620199
2: -0.0443946, 0.0211404, -0.0643693, 0.0467766, -0.0911712, 0.0855097
3: -0.0292436, 0.0388854, -0.0430202, 0.0885118, -0.1177553, 0.0819056
4: -0.0566717, 0.0260474, -0.0929886, 0.0495827, -0.1062544, 0.1190360

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0535956, upper bound: 0.0553343
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0536922, upper bound: 0.0551663
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0216623, 0.0203123, -0.0366124, 0.0387656
1: -0.0163531, 0.0312095, -0.0303493, 0.0405147, -0.0568679, 0.0615588
2: -0.0443946, 0.0211404, -0.0427012, 0.0220349, -0.0664295, 0.0638417
3: -0.0292436, 0.0388854, -0.0446583, 0.0515581, -0.0808017, 0.0835436
4: -0.0566717, 0.0260474, -0.0529711, 0.0229706, -0.0796423, 0.0790185

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536236, upper bound: 0.0551095
time: 0.27 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537327, upper bound: 0.0551142
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0238435, 0.0277011, -0.0440012, 0.0409468
1: -0.0163531, 0.0312095, -0.0284588, 0.0615482, -0.0779013, 0.0596684
2: -0.0443946, 0.0211404, -0.0619252, 0.0435389, -0.0879335, 0.0830657
3: -0.0292436, 0.0388854, -0.0407595, 0.0795807, -0.1088243, 0.0796448
4: -0.0566717, 0.0260474, -0.0884067, 0.0467996, -0.1034713, 0.1144541

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0536236, upper bound: 0.0553353
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0537327, upper bound: 0.0552655
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0275401, 0.0318847, -0.0238435, 0.0277011, -0.0552411, 0.0557282
1: -0.0315660, 0.0759994, -0.0284588, 0.0615482, -0.0931142, 0.1044583
2: -0.0714210, 0.0530449, -0.0619252, 0.0435389, -0.1149599, 0.1149701
3: -0.0452617, 0.0986971, -0.0407595, 0.0795807, -0.1248424, 0.1394565
4: -0.0993945, 0.0582935, -0.0884067, 0.0467996, -0.1461941, 0.1467002

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0530880, upper bound: 0.0555815
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0530894, upper bound: 0.0553265
time: 0.25 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0251495, 0.0299018, -0.0188328, 0.0191179, -0.0442673, 0.0487347
1: -0.0308103, 0.0680934, -0.0185345, 0.0336902, -0.0645005, 0.0866279
2: -0.0643693, 0.0467766, -0.0465600, 0.0245035, -0.0888728, 0.0933366
3: -0.0430202, 0.0885118, -0.0314490, 0.0414548, -0.0844750, 0.1199608
4: -0.0929886, 0.0495827, -0.0564701, 0.0289175, -0.1219061, 0.1060528

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543606, upper bound: 0.0549026
time: 0.22 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543606, upper bound: 0.0549026
time: 0.25 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0251495, 0.0299018, -0.0163001, 0.0171033, -0.0422528, 0.0462020
1: -0.0308103, 0.0680934, -0.0163531, 0.0312095, -0.0620199, 0.0844465
2: -0.0643693, 0.0467766, -0.0443946, 0.0211404, -0.0855097, 0.0911712
3: -0.0430202, 0.0885118, -0.0292436, 0.0388854, -0.0819056, 0.1177553
4: -0.0929886, 0.0495827, -0.0566717, 0.0260474, -0.1190360, 0.1062544

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529155, upper bound: 0.0545084
time: 0.24 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0528469, upper bound: 0.0549594
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0238435, 0.0277011, -0.0188328, 0.0191179, -0.0429614, 0.0465339
1: -0.0284588, 0.0615482, -0.0185345, 0.0336902, -0.0621490, 0.0800827
2: -0.0619252, 0.0435389, -0.0465600, 0.0245035, -0.0864288, 0.0900989
3: -0.0407595, 0.0795807, -0.0314490, 0.0414548, -0.0822143, 0.1110297
4: -0.0884067, 0.0467996, -0.0564701, 0.0289175, -0.1173242, 0.1032697

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547141, upper bound: 0.0539635
time: 0.24 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555702, upper bound: 0.0549320
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0251495, 0.0299018, -0.0275401, 0.0318847, -0.0570341, 0.0574419
1: -0.0308103, 0.0680934, -0.0315660, 0.0759994, -0.1068098, 0.0996594
2: -0.0643693, 0.0467766, -0.0714210, 0.0530449, -0.1174141, 0.1181976
3: -0.0430202, 0.0885118, -0.0452617, 0.0986971, -0.1417173, 0.1337735
4: -0.0929886, 0.0495827, -0.0993945, 0.0582935, -0.1512821, 0.1489772

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0529226, upper bound: 0.0543195
time: 0.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0528281, upper bound: 0.0551743
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0251495, 0.0299018, -0.0268415, 0.0305504, -0.0556999, 0.0567433
1: -0.0308103, 0.0680934, -0.0310880, 0.0714282, -0.1022385, 0.0991814
2: -0.0643693, 0.0467766, -0.0703224, 0.0510674, -0.1154366, 0.1170991
3: -0.0430202, 0.0885118, -0.0453250, 0.0927791, -0.1357993, 0.1338367
4: -0.0929886, 0.0495827, -0.0969780, 0.0568669, -0.1498556, 0.1465607

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0530003, upper bound: 0.0545096
time: 0.27 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0529174, upper bound: 0.0554006
time: 0.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0251495, 0.0299018, -0.0251495, 0.0299018, -0.0550513, 0.0550513
1: -0.0308103, 0.0680934, -0.0308103, 0.0680934, -0.0989038, 0.0989038
2: -0.0643693, 0.0467766, -0.0643693, 0.0467766, -0.1111459, 0.1111459
3: -0.0430202, 0.0885118, -0.0430202, 0.0885118, -0.1315320, 0.1315320
4: -0.0929886, 0.0495827, -0.0929886, 0.0495827, -0.1425713, 0.1425713

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0530165, upper bound: 0.0544885
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0529220, upper bound: 0.0551660
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0251495, 0.0299018, -0.0238435, 0.0277011, -0.0528505, 0.0537454
1: -0.0308103, 0.0680934, -0.0284588, 0.0615482, -0.0923586, 0.0965522
2: -0.0643693, 0.0467766, -0.0619252, 0.0435389, -0.1079082, 0.1087019
3: -0.0430202, 0.0885118, -0.0407595, 0.0795807, -0.1226009, 0.1292712
4: -0.0929886, 0.0495827, -0.0884067, 0.0467996, -0.1397882, 0.1379894

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0530863, upper bound: 0.0547273
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0529938, upper bound: 0.0553920
time: 0.22 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 1.31 seconds
NS_A1_B1_A1_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0547040, upper bound: 0.0548758
NS_A1_B1_A1_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0547040, upper bound: 0.0548758
NS_A1_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0535063, upper bound: 0.0552465
NS_A1_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0536153, upper bound: 0.0551117
NS_A1_B1_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0533088, upper bound: 0.0551844
NS_A1_B1_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0534154, upper bound: 0.0549130
NS_A1_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0535228, upper bound: 0.0552465
NS_A1_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0536319, upper bound: 0.0551117
NS_A1_B1_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0543515, upper bound: 0.0548036
NS_A1_B1_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0543515, upper bound: 0.0548036
NS_A1_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0529876, upper bound: 0.0550998
NS_A1_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0529876, upper bound: 0.0552509
NS_A1_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0544728, upper bound: 0.0549097
NS_A1_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0544728, upper bound: 0.0549097
NS_A1_B1_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0528588, upper bound: 0.0553265
NS_A1_B1_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0528586, upper bound: 0.0548381
NS_A1_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0530645, upper bound: 0.0555821
NS_A1_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0530645, upper bound: 0.0552091
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0535956, upper bound: 0.0553343
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0536922, upper bound: 0.0551663
NS_A1_B2_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0536236, upper bound: 0.0551095
NS_A1_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0537327, upper bound: 0.0551142
NS_A1_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0536236, upper bound: 0.0553353
NS_A1_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0537327, upper bound: 0.0552655
NS_A1_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0530880, upper bound: 0.0555815
NS_A1_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0530894, upper bound: 0.0553265
NS_A2_B1_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0543606, upper bound: 0.0549026
NS_A2_B1_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0543606, upper bound: 0.0549026
NS_A2_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0529155, upper bound: 0.0545084
NS_A2_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0528469, upper bound: 0.0549594
NS_A2_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0547141, upper bound: 0.0539635
NS_A2_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0555702, upper bound: 0.0549320
NS_A2_B1_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0529226, upper bound: 0.0543195
NS_A2_B1_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0528281, upper bound: 0.0551743
NS_A2_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0530003, upper bound: 0.0545096
NS_A2_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0529174, upper bound: 0.0554006
NS_A2_B2_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0530165, upper bound: 0.0544885
NS_A2_B2_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0529220, upper bound: 0.0551660
NS_A2_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0530863, upper bound: 0.0547273
NS_A2_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.31
Output dim: 0, lower bound: -0.0529938, upper bound: 0.0553920

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0170747, 0.0169581, -0.0163001, 0.0171033, -0.0341780, 0.0332582
1: -0.0162292, 0.0278331, -0.0163531, 0.0312095, -0.0474387, 0.0441862
2: -0.0424308, 0.0201539, -0.0443946, 0.0211404, -0.0635713, 0.0645484
3: -0.0286224, 0.0335460, -0.0292436, 0.0388854, -0.0675077, 0.0627896
4: -0.0510356, 0.0244890, -0.0566717, 0.0260474, -0.0770830, 0.0811607

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552716, upper bound: 0.0550665
time: 0.25 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552716, upper bound: 0.0551117
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0170747, 0.0169581, -0.0275401, 0.0318847, -0.0489594, 0.0444982
1: -0.0162292, 0.0278331, -0.0315660, 0.0759994, -0.0922286, 0.0593990
2: -0.0424308, 0.0201539, -0.0714210, 0.0530449, -0.0954757, 0.0915748
3: -0.0286224, 0.0335460, -0.0452617, 0.0986971, -0.1273194, 0.0788078
4: -0.0510356, 0.0244890, -0.0993945, 0.0582935, -0.1093291, 0.1238835

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551750, upper bound: 0.0549130
time: 0.26 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551750, upper bound: 0.0549130
time: 0.26 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0170747, 0.0169581, -0.0268415, 0.0305504, -0.0476251, 0.0437996
1: -0.0162292, 0.0278331, -0.0310880, 0.0714282, -0.0876574, 0.0589211
2: -0.0424308, 0.0201539, -0.0703224, 0.0510674, -0.0934982, 0.0904763
3: -0.0286224, 0.0335460, -0.0453250, 0.0927791, -0.1214015, 0.0788710
4: -0.0510356, 0.0244890, -0.0969780, 0.0568669, -0.1079025, 0.1214670

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552904, upper bound: 0.0550343
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552904, upper bound: 0.0551117
time: 0.26 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0236056, 0.0276349, -0.0163001, 0.0171033, -0.0407089, 0.0439350
1: -0.0246522, 0.0609175, -0.0163531, 0.0312095, -0.0558617, 0.0772707
2: -0.0631518, 0.0459309, -0.0443946, 0.0211404, -0.0842922, 0.0903255
3: -0.0380466, 0.0772544, -0.0292436, 0.0388854, -0.0769320, 0.1064979
4: -0.0868571, 0.0502552, -0.0566717, 0.0260474, -0.1129045, 0.1069269

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547760, upper bound: 0.0550983
time: 0.26 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547760, upper bound: 0.0552509
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0272616, 0.0315264, -0.0275401, 0.0318847, -0.0591463, 0.0590665
1: -0.0318267, 0.0731649, -0.0315660, 0.0759994, -0.1078262, 0.1047309
2: -0.0690642, 0.0515455, -0.0714210, 0.0530449, -0.1221090, 0.1229665
3: -0.0456918, 0.0943532, -0.0452617, 0.0986971, -0.1443888, 0.1396149
4: -0.0947821, 0.0565235, -0.0993945, 0.0582935, -0.1530755, 0.1559181

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548364, upper bound: 0.0548381
time: 0.25 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548364, upper bound: 0.0548381
time: 0.24 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0272616, 0.0315264, -0.0268415, 0.0305504, -0.0578120, 0.0583679
1: -0.0318267, 0.0731649, -0.0310880, 0.0714282, -0.1032549, 0.1042529
2: -0.0690642, 0.0515455, -0.0703224, 0.0510674, -0.1201315, 0.1218680
3: -0.0456918, 0.0943532, -0.0453250, 0.0927791, -0.1384709, 0.1396782
4: -0.0947821, 0.0565235, -0.0969780, 0.0568669, -0.1516490, 0.1535016

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548591, upper bound: 0.0549220
time: 0.25 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548591, upper bound: 0.0552091
time: 0.25 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0236056, 0.0276349, -0.0268415, 0.0305504, -0.0541560, 0.0544763
1: -0.0246522, 0.0609175, -0.0310880, 0.0714282, -0.0960803, 0.0920056
2: -0.0631518, 0.0459309, -0.0703224, 0.0510674, -0.1142191, 0.1162534
3: -0.0380466, 0.0772544, -0.0453250, 0.0927791, -0.1308257, 0.1225794
4: -0.0868571, 0.0502552, -0.0969780, 0.0568669, -0.1437240, 0.1472332

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548591, upper bound: 0.0549220
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548591, upper bound: 0.0552091
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0170747, 0.0169581, -0.0251495, 0.0299018, -0.0469765, 0.0421076
1: -0.0162292, 0.0278331, -0.0308103, 0.0680934, -0.0843226, 0.0586434
2: -0.0424308, 0.0201539, -0.0643693, 0.0467766, -0.0892075, 0.0845231
3: -0.0286224, 0.0335460, -0.0430202, 0.0885118, -0.1171341, 0.0765662
4: -0.0510356, 0.0244890, -0.0929886, 0.0495827, -0.1006183, 0.1174776

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552600, upper bound: 0.0549221
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552600, upper bound: 0.0551663
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0138432, 0.0135909, -0.0251495, 0.0299018, -0.0437450, 0.0387404
1: -0.0128219, 0.0240455, -0.0308103, 0.0680934, -0.0809153, 0.0548558
2: -0.0376750, 0.0155482, -0.0643693, 0.0467766, -0.0844516, 0.0799175
3: -0.0249458, 0.0289503, -0.0430202, 0.0885118, -0.1134576, 0.0719705
4: -0.0467689, 0.0194372, -0.0929886, 0.0495827, -0.0963516, 0.1124259

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553115, upper bound: 0.0549221
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553115, upper bound: 0.0551663
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0170747, 0.0169581, -0.0238435, 0.0277011, -0.0447758, 0.0408016
1: -0.0162292, 0.0278331, -0.0284588, 0.0615482, -0.0777774, 0.0562919
2: -0.0424308, 0.0201539, -0.0619252, 0.0435389, -0.0859697, 0.0820791
3: -0.0286224, 0.0335460, -0.0407595, 0.0795807, -0.1082031, 0.0743055
4: -0.0510356, 0.0244890, -0.0884067, 0.0467996, -0.0978352, 0.1128957

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543176, upper bound: 0.0551405
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552703, upper bound: 0.0552970
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0138432, 0.0135909, -0.0238435, 0.0277011, -0.0415442, 0.0374345
1: -0.0128219, 0.0240455, -0.0284588, 0.0615482, -0.0743701, 0.0525043
2: -0.0376750, 0.0155482, -0.0619252, 0.0435389, -0.0812139, 0.0774734
3: -0.0249458, 0.0289503, -0.0407595, 0.0795807, -0.1045265, 0.0697097
4: -0.0467689, 0.0194372, -0.0884067, 0.0467996, -0.0935685, 0.1078439

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543790, upper bound: 0.0551378
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553107, upper bound: 0.0552265
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0272616, 0.0315264, -0.0238435, 0.0277011, -0.0549626, 0.0553700
1: -0.0318267, 0.0731649, -0.0284588, 0.0615482, -0.0933749, 0.1016237
2: -0.0690642, 0.0515455, -0.0619252, 0.0435389, -0.1126031, 0.1134708
3: -0.0456918, 0.0943532, -0.0407595, 0.0795807, -0.1252725, 0.1351126
4: -0.0947821, 0.0565235, -0.0884067, 0.0467996, -0.1415816, 0.1449302

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0541587
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0553265
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0236056, 0.0276349, -0.0238435, 0.0277011, -0.0513067, 0.0514784
1: -0.0246522, 0.0609175, -0.0284588, 0.0615482, -0.0862004, 0.0893764
2: -0.0631518, 0.0459309, -0.0619252, 0.0435389, -0.1066907, 0.1078562
3: -0.0380466, 0.0772544, -0.0407595, 0.0795807, -0.1176273, 0.1180138
4: -0.0868571, 0.0502552, -0.0884067, 0.0467996, -0.1336567, 0.1386619

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0541577
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0553265
time: 0.25 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0235434, 0.0272988, -0.0188328, 0.0191179, -0.0426613, 0.0461316
1: -0.0278962, 0.0605775, -0.0185345, 0.0336902, -0.0615864, 0.0791120
2: -0.0613289, 0.0428859, -0.0465600, 0.0245035, -0.0858324, 0.0894459
3: -0.0400995, 0.0782054, -0.0314490, 0.0414548, -0.0815543, 0.1096544
4: -0.0875081, 0.0461136, -0.0564701, 0.0289175, -0.1164255, 0.1025837

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0245268, 0.0292438, -0.0275401, 0.0318847, -0.0564114, 0.0567838
1: -0.0296864, 0.0662994, -0.0315660, 0.0759994, -0.1056858, 0.0978654
2: -0.0634363, 0.0458538, -0.0714210, 0.0530449, -0.1164811, 0.1172748
3: -0.0417656, 0.0857486, -0.0452617, 0.0986971, -0.1404626, 0.1310104
4: -0.0916911, 0.0485822, -0.0993945, 0.0582935, -0.1499846, 0.1479768

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550454, upper bound: 0.0547826
time: 0.26 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547498, upper bound: 0.0547742
time: 0.25 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0245268, 0.0292438, -0.0268415, 0.0305504, -0.0550772, 0.0560852
1: -0.0296864, 0.0662994, -0.0310880, 0.0714282, -0.1011146, 0.0973874
2: -0.0634363, 0.0458538, -0.0703224, 0.0510674, -0.1145036, 0.1161762
3: -0.0417656, 0.0857486, -0.0453250, 0.0927791, -0.1345447, 0.1310736
4: -0.0916911, 0.0485822, -0.0969780, 0.0568669, -0.1485580, 0.1455603

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548632, upper bound: 0.0548974
time: 0.25 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547802, upper bound: 0.0551658
time: 0.26 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0245268, 0.0292438, -0.0251495, 0.0299018, -0.0544286, 0.0543932
1: -0.0296864, 0.0662994, -0.0308103, 0.0680934, -0.0977798, 0.0971098
2: -0.0634363, 0.0458538, -0.0643693, 0.0467766, -0.1102129, 0.1102231
3: -0.0417656, 0.0857486, -0.0430202, 0.0885118, -0.1302773, 0.1287689
4: -0.0916911, 0.0485822, -0.0929886, 0.0495827, -0.1412738, 0.1415709

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542286, upper bound: 0.0545577
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0542286, upper bound: 0.0551660
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0245268, 0.0292438, -0.0238435, 0.0277011, -0.0522278, 0.0530873
1: -0.0296864, 0.0662994, -0.0284588, 0.0615482, -0.0912346, 0.0947582
2: -0.0634363, 0.0458538, -0.0619252, 0.0435389, -0.1069752, 0.1077790
3: -0.0417656, 0.0857486, -0.0407595, 0.0795807, -0.1213463, 0.1265081
4: -0.0916911, 0.0485822, -0.0884067, 0.0467996, -0.1384906, 0.1369889

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0542026, upper bound: 0.0548689
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549196, upper bound: 0.0553897
time: 0.25 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 1.42 seconds
NS_A1_B1_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0552716, upper bound: 0.0550665
NS_A1_B1_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0552716, upper bound: 0.0551117
NS_A1_B1_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0551750, upper bound: 0.0549130
NS_A1_B1_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0551750, upper bound: 0.0549130
NS_A1_B1_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0552904, upper bound: 0.0550343
NS_A1_B1_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0552904, upper bound: 0.0551117
NS_A1_B1_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0547760, upper bound: 0.0550983
NS_A1_B1_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0547760, upper bound: 0.0552509
NS_A1_B1_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0548364, upper bound: 0.0548381
NS_A1_B1_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0548364, upper bound: 0.0548381
NS_A1_B1_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0548591, upper bound: 0.0549220
NS_A1_B1_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0548591, upper bound: 0.0552091
NS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0548591, upper bound: 0.0549220
NS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0548591, upper bound: 0.0552091
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0552600, upper bound: 0.0549221
NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0552600, upper bound: 0.0551663
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0553115, upper bound: 0.0549221
NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0553115, upper bound: 0.0551663
NS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0543176, upper bound: 0.0551405
NS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0552703, upper bound: 0.0552970
NS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0543790, upper bound: 0.0551378
NS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0553107, upper bound: 0.0552265
NS_A1_B2_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0541587
NS_A1_B2_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0553265
NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0541577
NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0548595, upper bound: 0.0553265
NS_A2_B1_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0550454, upper bound: 0.0547826
NS_A2_B1_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0547498, upper bound: 0.0547742
NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0548632, upper bound: 0.0548974
NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0547802, upper bound: 0.0551658
NS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0542286, upper bound: 0.0545577
NS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0542286, upper bound: 0.0551660
NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0542026, upper bound: 0.0548689
NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.42
Output dim: 0, lower bound: -0.0549196, upper bound: 0.0553897

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0170747, 0.0169581, -0.0170747, 0.0169581, -0.0340328, 0.0340328
1: -0.0162292, 0.0278331, -0.0162292, 0.0278331, -0.0440623, 0.0440623
2: -0.0424308, 0.0201539, -0.0424308, 0.0201539, -0.0625847, 0.0625847
3: -0.0286224, 0.0335460, -0.0286224, 0.0335460, -0.0621684, 0.0621684
4: -0.0510356, 0.0244890, -0.0510356, 0.0244890, -0.0755246, 0.0755246

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0170747, 0.0169581, -0.0138432, 0.0135909, -0.0306657, 0.0308013
1: -0.0162292, 0.0278331, -0.0128219, 0.0240455, -0.0402747, 0.0406550
2: -0.0424308, 0.0201539, -0.0376750, 0.0155482, -0.0579790, 0.0578288
3: -0.0286224, 0.0335460, -0.0249458, 0.0289503, -0.0575726, 0.0584918
4: -0.0510356, 0.0244890, -0.0467689, 0.0194372, -0.0704728, 0.0712579

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0170747, 0.0169581, -0.0272616, 0.0315264, -0.0486012, 0.0442197
1: -0.0162292, 0.0278331, -0.0318267, 0.0731649, -0.0893941, 0.0596598
2: -0.0424308, 0.0201539, -0.0690642, 0.0515455, -0.0939763, 0.0892180
3: -0.0286224, 0.0335460, -0.0456918, 0.0943532, -0.1229755, 0.0792378
4: -0.0510356, 0.0244890, -0.0947821, 0.0565235, -0.1075591, 0.1192710

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0170747, 0.0169581, -0.0236056, 0.0276349, -0.0447096, 0.0405637
1: -0.0162292, 0.0278331, -0.0246522, 0.0609175, -0.0771468, 0.0524852
2: -0.0424308, 0.0201539, -0.0631518, 0.0459309, -0.0883618, 0.0833056
3: -0.0286224, 0.0335460, -0.0380466, 0.0772544, -0.1058767, 0.0715926
4: -0.0510356, 0.0244890, -0.0868571, 0.0502552, -0.1012908, 0.1113461

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0170747, 0.0169581, -0.0266347, 0.0300013, -0.0470760, 0.0435928
1: -0.0162292, 0.0278331, -0.0309553, 0.0678936, -0.0841229, 0.0587884
2: -0.0424308, 0.0201539, -0.0675894, 0.0492085, -0.0916393, 0.0877432
3: -0.0286224, 0.0335460, -0.0454101, 0.0878100, -0.1164324, 0.0789561
4: -0.0510356, 0.0244890, -0.0916399, 0.0547648, -0.1058004, 0.1161289

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0170747, 0.0169581, -0.0230536, 0.0262401, -0.0433149, 0.0400117
1: -0.0162292, 0.0278331, -0.0242822, 0.0563962, -0.0726254, 0.0521153
2: -0.0424308, 0.0201539, -0.0619228, 0.0438159, -0.0862467, 0.0820767
3: -0.0286224, 0.0335460, -0.0380059, 0.0721582, -0.1007806, 0.0715519
4: -0.0510356, 0.0244890, -0.0842291, 0.0486820, -0.0997176, 0.1087181

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0236056, 0.0276349, -0.0138432, 0.0135909, -0.0371965, 0.0414780
1: -0.0246522, 0.0609175, -0.0128219, 0.0240455, -0.0486976, 0.0737395
2: -0.0631518, 0.0459309, -0.0376750, 0.0155482, -0.0787000, 0.0836059
3: -0.0380466, 0.0772544, -0.0249458, 0.0289503, -0.0669969, 0.1022002
4: -0.0868571, 0.0502552, -0.0467689, 0.0194372, -0.1062943, 0.0970241

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0272616, 0.0315264, -0.0230536, 0.0262401, -0.0535017, 0.0545800
1: -0.0318267, 0.0731649, -0.0242822, 0.0563962, -0.0882230, 0.0974471
2: -0.0690642, 0.0515455, -0.0619228, 0.0438159, -0.1128800, 0.1134683
3: -0.0456918, 0.0943532, -0.0380059, 0.0721582, -0.1178500, 0.1323591
4: -0.0947821, 0.0565235, -0.0842291, 0.0486820, -0.1434640, 0.1407526

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0236056, 0.0276349, -0.0230536, 0.0262401, -0.0498457, 0.0506884
1: -0.0246522, 0.0609175, -0.0242822, 0.0563962, -0.0810484, 0.0851998
2: -0.0631518, 0.0459309, -0.0619228, 0.0438159, -0.1069676, 0.1078537
3: -0.0380466, 0.0772544, -0.0380059, 0.0721582, -0.1102048, 0.1152603
4: -0.0868571, 0.0502552, -0.0842291, 0.0486820, -0.1355391, 0.1344843

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0170747, 0.0169581, -0.0256376, 0.0295575, -0.0466322, 0.0425957
1: -0.0162292, 0.0278331, -0.0305225, 0.0648765, -0.0811058, 0.0583556
2: -0.0424308, 0.0201539, -0.0628560, 0.0447159, -0.0871467, 0.0830099
3: -0.0286224, 0.0335460, -0.0431723, 0.0831687, -0.1117911, 0.0767183
4: -0.0510356, 0.0244890, -0.0872203, 0.0479456, -0.0989812, 0.1117093

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0170747, 0.0169581, -0.0204236, 0.0249054, -0.0419802, 0.0373817
1: -0.0162292, 0.0278331, -0.0217593, 0.0511472, -0.0673764, 0.0495924
2: -0.0424308, 0.0201539, -0.0544312, 0.0388515, -0.0812824, 0.0745850
3: -0.0286224, 0.0335460, -0.0334944, 0.0643775, -0.0929999, 0.0670404
4: -0.0510356, 0.0244890, -0.0778459, 0.0405995, -0.0916351, 0.1023349

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0138432, 0.0135909, -0.0256376, 0.0295575, -0.0434006, 0.0392286
1: -0.0128219, 0.0240455, -0.0305225, 0.0648765, -0.0776985, 0.0545680
2: -0.0376750, 0.0155482, -0.0628560, 0.0447159, -0.0823908, 0.0784042
3: -0.0249458, 0.0289503, -0.0431723, 0.0831687, -0.1081145, 0.0721226
4: -0.0467689, 0.0194372, -0.0872203, 0.0479456, -0.0947145, 0.1066575

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0138432, 0.0135909, -0.0204236, 0.0249054, -0.0387486, 0.0340145
1: -0.0128219, 0.0240455, -0.0217593, 0.0511472, -0.0639692, 0.0458048
2: -0.0376750, 0.0155482, -0.0544312, 0.0388515, -0.0765265, 0.0699794
3: -0.0249458, 0.0289503, -0.0334944, 0.0643775, -0.0893233, 0.0624447
4: -0.0467689, 0.0194372, -0.0778459, 0.0405995, -0.0873684, 0.0972831

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0170747, 0.0169581, -0.0235434, 0.0272988, -0.0443735, 0.0405015
1: -0.0162292, 0.0278331, -0.0278962, 0.0605775, -0.0768067, 0.0557293
2: -0.0424308, 0.0201539, -0.0613289, 0.0428859, -0.0853167, 0.0814828
3: -0.0286224, 0.0335460, -0.0400995, 0.0782054, -0.1068277, 0.0736455
4: -0.0510356, 0.0244890, -0.0875081, 0.0461136, -0.0971492, 0.1119970

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0138432, 0.0135909, -0.0235434, 0.0272988, -0.0411420, 0.0371344
1: -0.0128219, 0.0240455, -0.0278962, 0.0605775, -0.0733994, 0.0519417
2: -0.0376750, 0.0155482, -0.0613289, 0.0428859, -0.0805608, 0.0768771
3: -0.0249458, 0.0289503, -0.0400995, 0.0782054, -0.1031512, 0.0690498
4: -0.0467689, 0.0194372, -0.0875081, 0.0461136, -0.0928825, 0.1069453

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0272616, 0.0315264, -0.0193763, 0.0228807, -0.0501423, 0.0509028
1: -0.0318267, 0.0731649, -0.0198272, 0.0449618, -0.0767885, 0.0929921
2: -0.0690642, 0.0515455, -0.0521190, 0.0357868, -0.1048509, 0.1036645
3: -0.0456918, 0.0943532, -0.0315510, 0.0568579, -0.1025496, 0.1259042
4: -0.0947821, 0.0565235, -0.0734975, 0.0379837, -0.1327658, 0.1300210

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0236056, 0.0276349, -0.0193763, 0.0228807, -0.0464863, 0.0470112
1: -0.0246522, 0.0609175, -0.0198272, 0.0449618, -0.0696139, 0.0807447
2: -0.0631518, 0.0459309, -0.0521190, 0.0357868, -0.0989386, 0.0980499
3: -0.0380466, 0.0772544, -0.0315510, 0.0568579, -0.0949045, 0.1088054
4: -0.0868571, 0.0502552, -0.0734975, 0.0379837, -0.1248408, 0.1237527

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0245268, 0.0292438, -0.0230536, 0.0262401, -0.0507669, 0.0522974
1: -0.0296864, 0.0662994, -0.0242822, 0.0563962, -0.0860826, 0.0905816
2: -0.0634363, 0.0458538, -0.0619228, 0.0438159, -0.1072521, 0.1077766
3: -0.0417656, 0.0857486, -0.0380059, 0.0721582, -0.1139238, 0.1237546
4: -0.0916911, 0.0485822, -0.0842291, 0.0486820, -0.1403730, 0.1328113

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547787, upper bound: 0.0551658
time: 0.25 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547787, upper bound: 0.0551658
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0245268, 0.0292438, -0.0245268, 0.0292438, -0.0537705, 0.0537705
1: -0.0296864, 0.0662994, -0.0296864, 0.0662994, -0.0959858, 0.0959858
2: -0.0634363, 0.0458538, -0.0634363, 0.0458538, -0.1092901, 0.1092901
3: -0.0417656, 0.0857486, -0.0417656, 0.0857486, -0.1275142, 0.1275142
4: -0.0916911, 0.0485822, -0.0916911, 0.0485822, -0.1402733, 0.1402733

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0245268, 0.0292438, -0.0233569, 0.0272538, -0.0517806, 0.0526006
1: -0.0296864, 0.0662994, -0.0273691, 0.0601274, -0.0898138, 0.0936686
2: -0.0634363, 0.0458538, -0.0611299, 0.0428821, -0.1063184, 0.1069837
3: -0.0417656, 0.0857486, -0.0394889, 0.0775464, -0.1193119, 0.1252375
4: -0.0916911, 0.0485822, -0.0872208, 0.0460661, -0.1377571, 0.1358030

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547787, upper bound: 0.0553371
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547802, upper bound: 0.0552177
time: 0.25 seconds

## Summary of splitting at layer (split count: 10)
- Time for NS candidates: 2.00 seconds
NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.00
Output dim: 0, lower bound: -0.0547787, upper bound: 0.0551658
NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.00
Output dim: 0, lower bound: -0.0547787, upper bound: 0.0551658
NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 2.00
Output dim: 0, lower bound: -0.0547787, upper bound: 0.0553371
NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 2.00
Output dim: 0, lower bound: -0.0547802, upper bound: 0.0552177

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0245639, 0.0287946, -0.0230536, 0.0262401, -0.0508040, 0.0518482
1: -0.0288738, 0.0628582, -0.0242822, 0.0563962, -0.0852700, 0.0871404
2: -0.0610065, 0.0436774, -0.0619228, 0.0438159, -0.1048224, 0.1056002
3: -0.0413026, 0.0801518, -0.0380059, 0.0721582, -0.1134608, 0.1181577
4: -0.0858290, 0.0466287, -0.0842291, 0.0486820, -0.1345110, 0.1308578

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0199507, 0.0243634, -0.0230536, 0.0262401, -0.0461909, 0.0474170
1: -0.0207362, 0.0497175, -0.0242822, 0.0563962, -0.0771325, 0.0739997
2: -0.0536080, 0.0381038, -0.0619228, 0.0438159, -0.0974239, 0.1000266
3: -0.0322277, 0.0623925, -0.0380059, 0.0721582, -0.1043860, 0.1003984
4: -0.0766708, 0.0397601, -0.0842291, 0.0486820, -0.1253528, 0.1239891

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0245639, 0.0287946, -0.0233569, 0.0272538, -0.0518177, 0.0521515
1: -0.0288738, 0.0628582, -0.0273691, 0.0601274, -0.0890012, 0.0902273
2: -0.0610065, 0.0436774, -0.0611299, 0.0428821, -0.1038886, 0.1048073
3: -0.0413026, 0.0801518, -0.0394889, 0.0775464, -0.1188490, 0.1196407
4: -0.0858290, 0.0466287, -0.0872208, 0.0460661, -0.1318951, 0.1338494

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547787, upper bound: 0.0548050
time: 0.27 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547787, upper bound: 0.0552177
time: 0.28 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0199507, 0.0243634, -0.0233569, 0.0272538, -0.0472046, 0.0477203
1: -0.0207362, 0.0497175, -0.0273691, 0.0601274, -0.0808637, 0.0770866
2: -0.0536080, 0.0381038, -0.0611299, 0.0428821, -0.0964901, 0.0992337
3: -0.0322277, 0.0623925, -0.0394889, 0.0775464, -0.1097741, 0.1018814
4: -0.0766708, 0.0397601, -0.0872208, 0.0460661, -0.1227369, 0.1269808

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0541682, upper bound: 0.0546468
time: 0.27 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0541682, upper bound: 0.0552177
time: 0.30 seconds

## Summary of splitting at layer (split count: 11)
- Time for NS candidates: 1.46 seconds
NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 12, time: 1.46
Output dim: 0, lower bound: -0.0547787, upper bound: 0.0548050
NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 12, time: 1.46
Output dim: 0, lower bound: -0.0547787, upper bound: 0.0552177
NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 1.46
Output dim: 0, lower bound: -0.0541682, upper bound: 0.0546468
NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 1.46
Output dim: 0, lower bound: -0.0541682, upper bound: 0.0552177

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0245639, 0.0287946, -0.0189508, 0.0225059, -0.0470698, 0.0477454
1: -0.0288738, 0.0628582, -0.0189328, 0.0435142, -0.0723880, 0.0817910
2: -0.0610065, 0.0436774, -0.0513224, 0.0352242, -0.0962307, 0.0949998
3: -0.0413026, 0.0801518, -0.0304608, 0.0547802, -0.0960828, 0.1106126
4: -0.0858290, 0.0466287, -0.0724334, 0.0373175, -0.1231465, 0.1190621

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0199507, 0.0243634, -0.0226454, 0.0265212, -0.0464719, 0.0470088
1: -0.0207362, 0.0497175, -0.0259927, 0.0580920, -0.0788283, 0.0757101
2: -0.0536080, 0.0381038, -0.0600343, 0.0417470, -0.0953550, 0.0981381
3: -0.0322277, 0.0623925, -0.0378310, 0.0746846, -0.1069124, 0.1002235
4: -0.0766708, 0.0397601, -0.0856158, 0.0448427, -0.1215136, 0.1253758

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.39 + 262.20 = 263.59 seconds
