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
execution time: IAR + RelationalAnalysis = 0.78 + 0.77 = 1.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0559989, upper bound: 0.0559989

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556559
time: 0.20 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0556391
time: 0.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.52 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.52
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556559
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.52
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0556391

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0206600, 0.0216224, -0.0278593, 0.0309887, -0.0516487, 0.0494817
1: -0.0226827, 0.0442280, -0.0350513, 0.0705397, -0.0932224, 0.0792793
2: -0.0535189, 0.0294451, -0.0677722, 0.0423128, -0.0958317, 0.0972173
3: -0.0368305, 0.0571968, -0.0527389, 0.0981292, -0.1349597, 0.1099357
4: -0.0685483, 0.0351860, -0.0944105, 0.0499190, -0.1184673, 0.1295964

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547472, upper bound: 0.0554908
time: 0.21 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556559
time: 0.21 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0213640, 0.0238672, -0.0278593, 0.0309887, -0.0523526, 0.0517265
1: -0.0256137, 0.0480500, -0.0350513, 0.0705397, -0.0961535, 0.0831014
2: -0.0530172, 0.0317309, -0.0677722, 0.0423128, -0.0953299, 0.0995031
3: -0.0398649, 0.0629934, -0.0527389, 0.0981292, -0.1379941, 0.1157324
4: -0.0719933, 0.0355104, -0.0944105, 0.0499190, -0.1219123, 0.1299209

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554908, upper bound: 0.0548863
time: 0.21 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0556391
time: 0.24 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.29 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.29
Output dim: 0, lower bound: -0.0547472, upper bound: 0.0554908
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.29
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556559
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 1.29
Output dim: 0, lower bound: -0.0554908, upper bound: 0.0548863
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 1.29
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0556391

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0206600, 0.0216224, -0.0277201, 0.0323688, -0.0530288, 0.0493425
1: -0.0226827, 0.0442280, -0.0308648, 0.0743695, -0.0970522, 0.0750928
2: -0.0535189, 0.0294451, -0.0791640, 0.0529521, -0.1064711, 0.1086091
3: -0.0368305, 0.0571968, -0.0498447, 0.1098390, -0.1466695, 0.1070414
4: -0.0685483, 0.0351860, -0.1205781, 0.0659159, -0.1344641, 0.1557640

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546173, upper bound: 0.0554184
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547103, upper bound: 0.0554292
time: 0.22 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0206600, 0.0216224, -0.0253249, 0.0272606, -0.0479206, 0.0469473
1: -0.0226827, 0.0442280, -0.0293032, 0.0589747, -0.0816574, 0.0735311
2: -0.0535189, 0.0294451, -0.0625771, 0.0371501, -0.0906690, 0.0920222
3: -0.0368305, 0.0571968, -0.0449141, 0.0797573, -0.1165878, 0.1021109
4: -0.0685483, 0.0351860, -0.0838840, 0.0437622, -0.1123105, 0.1190700

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556485
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556545
time: 0.21 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.0200828, 0.0232392, -0.0278593, 0.0309887, -0.0510714, 0.0510985
1: -0.0189451, 0.0503686, -0.0350513, 0.0705397, -0.0894848, 0.0854199
2: -0.0627839, 0.0392658, -0.0677722, 0.0423128, -0.1050967, 0.1070380
3: -0.0322327, 0.0689562, -0.0527389, 0.0981292, -0.1303619, 0.1216951
4: -0.0872019, 0.0471414, -0.0944105, 0.0499190, -0.1371209, 0.1415519

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548836, upper bound: 0.0548837
time: 0.21 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548836, upper bound: 0.0548863
time: 0.21 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.0195409, 0.0220395, -0.0278593, 0.0309887, -0.0505296, 0.0498988
1: -0.0217663, 0.0421363, -0.0350513, 0.0705397, -0.0923060, 0.0771876
2: -0.0493362, 0.0288256, -0.0677722, 0.0423128, -0.0916490, 0.0965979
3: -0.0353122, 0.0542807, -0.0527389, 0.0981292, -0.1334414, 0.1070196
4: -0.0671461, 0.0321655, -0.0944105, 0.0499190, -0.1170651, 0.1265759

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555548, upper bound: 0.0552530
time: 0.23 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0556391
time: 0.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.23 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.23
Output dim: 0, lower bound: -0.0546173, upper bound: 0.0554184
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.23
Output dim: 0, lower bound: -0.0547103, upper bound: 0.0554292
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.23
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556485
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.23
Output dim: 0, lower bound: -0.0555625, upper bound: 0.0556545
NS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 1.23
Output dim: 0, lower bound: -0.0548836, upper bound: 0.0548837
NS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 1.23
Output dim: 0, lower bound: -0.0548836, upper bound: 0.0548863
NS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 1.23
Output dim: 0, lower bound: -0.0555548, upper bound: 0.0552530
NS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 1.23
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0556391

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0220630, 0.0224854, -0.0277201, 0.0323688, -0.0544318, 0.0502055
1: -0.0235168, 0.0437383, -0.0308648, 0.0743695, -0.0978863, 0.0746032
2: -0.0539281, 0.0298981, -0.0791640, 0.0529521, -0.1068802, 0.1090621
3: -0.0377854, 0.0553166, -0.0498447, 0.1098390, -0.1476244, 0.1051613
4: -0.0649626, 0.0354342, -0.1205781, 0.0659159, -0.1308785, 0.1560123

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546107, upper bound: 0.0553041
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546173, upper bound: 0.0554183
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0173346, 0.0176596, -0.0277201, 0.0323688, -0.0497034, 0.0453797
1: -0.0178321, 0.0313102, -0.0308648, 0.0743695, -0.0922015, 0.0621751
2: -0.0455050, 0.0226024, -0.0791640, 0.0529521, -0.0984572, 0.1017665
3: -0.0306701, 0.0403801, -0.0498447, 0.1098390, -0.1405091, 0.0902248
4: -0.0569695, 0.0274540, -0.1205781, 0.0659159, -0.1228854, 0.1480321

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546851, upper bound: 0.0551832
time: 0.19 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547103, upper bound: 0.0554292
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0174403, 0.0183881, -0.0253249, 0.0272606, -0.0447009, 0.0437129
1: -0.0179365, 0.0346897, -0.0293032, 0.0589747, -0.0769111, 0.0639929
2: -0.0468226, 0.0235523, -0.0625771, 0.0371501, -0.0839727, 0.0861294
3: -0.0312331, 0.0437208, -0.0449141, 0.0797573, -0.1109904, 0.0886349
4: -0.0597628, 0.0287614, -0.0838840, 0.0437622, -0.1035250, 0.1126454

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545644, upper bound: 0.0552696
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554129, upper bound: 0.0555597
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0277988, 0.0316155, -0.0253249, 0.0272606, -0.0550594, 0.0569404
1: -0.0329395, 0.0745442, -0.0293032, 0.0589747, -0.0919142, 0.1038474
2: -0.0723793, 0.0529296, -0.0625771, 0.0371501, -0.1095294, 0.1155067
3: -0.0475005, 0.0975785, -0.0449141, 0.0797573, -0.1272578, 0.1424926
4: -0.0997654, 0.0589750, -0.0838840, 0.0437622, -0.1435277, 0.1428590

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556167
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556545
time: 0.21 seconds

## BFS NS instance: NS_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0278593, 0.0309887, -0.0487415, 0.0478833
1: -0.0189958, 0.0363287, -0.0350513, 0.0705397, -0.0895356, 0.0713800
2: -0.0453282, 0.0254624, -0.0677722, 0.0423128, -0.0876410, 0.0932346
3: -0.0320662, 0.0460069, -0.0527389, 0.0981292, -0.1301954, 0.0987459
4: -0.0614046, 0.0284768, -0.0944105, 0.0499190, -0.1113236, 0.1228873

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0552506
time: 0.21 seconds

## Relational analysis of NS_A2_A2_A1_B2

### Relational analysis result of NS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0552530
time: 0.22 seconds

## BFS NS instance: NS_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0244671, 0.0285040, -0.0278593, 0.0309887, -0.0554557, 0.0563633
1: -0.0291369, 0.0633202, -0.0350513, 0.0705397, -0.0996767, 0.0983715
2: -0.0632677, 0.0448866, -0.0677722, 0.0423128, -0.1055805, 0.1126589
3: -0.0417759, 0.0822587, -0.0527389, 0.0981292, -0.1399051, 0.1349977
4: -0.0902513, 0.0482991, -0.0944105, 0.0499190, -0.1401703, 0.1427096

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_A2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0555568
time: 0.24 seconds

## Relational analysis of NS_A2_A2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0555568
time: 0.23 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.27 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.27
Output dim: 0, lower bound: -0.0546107, upper bound: 0.0553041
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.27
Output dim: 0, lower bound: -0.0546173, upper bound: 0.0554183
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.27
Output dim: 0, lower bound: -0.0546851, upper bound: 0.0551832
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.27
Output dim: 0, lower bound: -0.0547103, upper bound: 0.0554292
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.27
Output dim: 0, lower bound: -0.0545644, upper bound: 0.0552696
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.27
Output dim: 0, lower bound: -0.0554129, upper bound: 0.0555597
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.27
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556167
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.27
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556545
NS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.27
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0552506
NS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.27
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0552530
NS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.27
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0555568
NS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.27
Output dim: 0, lower bound: -0.0556391, upper bound: 0.0555568

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0220630, 0.0224854, -0.0264689, 0.0305887, -0.0526517, 0.0489543
1: -0.0235168, 0.0437383, -0.0270966, 0.0694186, -0.0929354, 0.0708349
2: -0.0539281, 0.0298981, -0.0762584, 0.0489375, -0.1028656, 0.1061565
3: -0.0377854, 0.0553166, -0.0429505, 0.1018456, -0.1396310, 0.0982671
4: -0.0649626, 0.0354342, -0.1151821, 0.0590568, -0.1240194, 0.1506162

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0535956, upper bound: 0.0550675
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546074, upper bound: 0.0552825
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0220630, 0.0224854, -0.0272482, 0.0315258, -0.0535888, 0.0497336
1: -0.0235168, 0.0437383, -0.0296192, 0.0724507, -0.0959676, 0.0733575
2: -0.0539281, 0.0298981, -0.0780765, 0.0513000, -0.1052281, 0.1079746
3: -0.0377854, 0.0553166, -0.0476144, 0.1069658, -0.1447512, 0.1029310
4: -0.0649626, 0.0354342, -0.1186049, 0.0633985, -0.1283610, 0.1540391

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536236, upper bound: 0.0551141
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546114, upper bound: 0.0553942
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0173346, 0.0176596, -0.0264689, 0.0305887, -0.0479233, 0.0441285
1: -0.0178321, 0.0313102, -0.0270966, 0.0694186, -0.0872506, 0.0584068
2: -0.0455050, 0.0226024, -0.0762584, 0.0489375, -0.0944426, 0.0988608
3: -0.0306701, 0.0403801, -0.0429505, 0.1018456, -0.1325157, 0.0833306
4: -0.0569695, 0.0274540, -0.1151821, 0.0590568, -0.1160264, 0.1426361

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536922, upper bound: 0.0550306
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546804, upper bound: 0.0551591
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0173346, 0.0176596, -0.0272482, 0.0315258, -0.0488603, 0.0449078
1: -0.0178321, 0.0313102, -0.0296192, 0.0724507, -0.0902828, 0.0609294
2: -0.0455050, 0.0226024, -0.0780765, 0.0513000, -0.0968051, 0.1006789
3: -0.0306701, 0.0403801, -0.0476144, 0.1069658, -0.1376359, 0.0879945
4: -0.0569695, 0.0274540, -0.1186049, 0.0633985, -0.1203680, 0.1460589

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537327, upper bound: 0.0551335
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546919, upper bound: 0.0554038
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0173575, 0.0182809, -0.0119187, 0.0089635, -0.0263210, 0.0301996
1: -0.0178190, 0.0343548, -0.0112788, 0.0104788, -0.0282978, 0.0456336
2: -0.0465940, 0.0233131, -0.0260187, 0.0060609, -0.0526549, 0.0493319
3: -0.0310810, 0.0432619, -0.0227056, 0.0075995, -0.0386805, 0.0659675
4: -0.0594004, 0.0285027, -0.0251642, 0.0067536, -0.0661540, 0.0536669

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544417, upper bound: 0.0552160
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545095, upper bound: 0.0551352
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0174403, 0.0183881, -0.0230995, 0.0234014, -0.0408417, 0.0414876
1: -0.0179365, 0.0346897, -0.0252745, 0.0471038, -0.0650403, 0.0599642
2: -0.0468226, 0.0235523, -0.0575486, 0.0316391, -0.0784617, 0.0811009
3: -0.0312331, 0.0437208, -0.0398286, 0.0625571, -0.0937902, 0.0835494
4: -0.0597628, 0.0287614, -0.0735941, 0.0380771, -0.0978399, 0.1023556

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550691, upper bound: 0.0555526
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550691, upper bound: 0.0555597
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0277988, 0.0316155, -0.0224162, 0.0235390, -0.0513378, 0.0540317
1: -0.0329395, 0.0745442, -0.0244602, 0.0471920, -0.0801315, 0.0990044
2: -0.0723793, 0.0529296, -0.0564170, 0.0319501, -0.1043294, 0.1093466
3: -0.0475005, 0.0975785, -0.0391877, 0.0615597, -0.1090603, 0.1367662
4: -0.0997654, 0.0589750, -0.0724862, 0.0380577, -0.1378231, 0.1314612

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556167
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556167
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0277988, 0.0316155, -0.0421270, 0.0516748, -0.0794736, 0.0737425
1: -0.0329395, 0.0745442, -0.0601539, 0.1331329, -0.1660725, 0.1346981
2: -0.0723793, 0.0529296, -0.0991491, 0.0719263, -0.1443056, 0.1520787
3: -0.0475005, 0.0975785, -0.0832208, 0.1978890, -0.2453895, 0.1807994
4: -0.0997654, 0.0589750, -0.1639526, 0.0807114, -0.1804768, 0.2229276

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556173
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556248
time: 0.22 seconds

## BFS NS instance: NS_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0240651, 0.0255380, -0.0432908, 0.0440892
1: -0.0189958, 0.0363287, -0.0277024, 0.0537890, -0.0727848, 0.0640311
2: -0.0453282, 0.0254624, -0.0598821, 0.0348018, -0.0801300, 0.0853444
3: -0.0320662, 0.0460069, -0.0434552, 0.0715716, -0.1036378, 0.0894621
4: -0.0614046, 0.0284768, -0.0781505, 0.0412263, -0.1026309, 0.1066273

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_A2_A1_B1_B1

### Relational analysis result of NS_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
time: 0.22 seconds

## Relational analysis of NS_A2_A2_A1_B1_B2

### Relational analysis result of NS_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
time: 0.21 seconds

## BFS NS instance: NS_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0442673, 0.0576540, -0.0754068, 0.0642913
1: -0.0189958, 0.0363287, -0.0810332, 0.1428985, -0.1618943, 0.1173618
2: -0.0453282, 0.0254624, -0.1032308, 0.0975138, -0.1428420, 0.1286932
3: -0.0320662, 0.0460069, -0.1345689, 0.2135442, -0.2456104, 0.1805758
4: -0.0614046, 0.0284768, -0.1719761, 0.1324061, -0.1938107, 0.2004529

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_A2_A1_B2_B1

### Relational analysis result of NS_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
time: 0.20 seconds

## Relational analysis of NS_A2_A2_A1_B2_B2

### Relational analysis result of NS_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
time: 0.21 seconds

## BFS NS instance: NS_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0244671, 0.0285040, -0.0206600, 0.0216224, -0.0460895, 0.0491640
1: -0.0291369, 0.0633202, -0.0226827, 0.0442280, -0.0733649, 0.0860029
2: -0.0632677, 0.0448866, -0.0535189, 0.0294451, -0.0927128, 0.0984056
3: -0.0417759, 0.0822587, -0.0368305, 0.0571968, -0.0989727, 0.1190892
4: -0.0902513, 0.0482991, -0.0685483, 0.0351860, -0.1254373, 0.1168474

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555548
time: 0.23 seconds

## Relational analysis of NS_A2_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555568
time: 0.25 seconds

## BFS NS instance: NS_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0244671, 0.0285040, -0.0213640, 0.0238672, -0.0483343, 0.0498680
1: -0.0291369, 0.0633202, -0.0256137, 0.0480500, -0.0771870, 0.0889339
2: -0.0632677, 0.0448866, -0.0530172, 0.0317309, -0.0949986, 0.0979038
3: -0.0417759, 0.0822587, -0.0398649, 0.0629934, -0.1047693, 0.1221236
4: -0.0902513, 0.0482991, -0.0719933, 0.0355104, -0.1257617, 0.1202924

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555548
time: 0.23 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555568
time: 0.22 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.34 seconds
NS_A1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0535956, upper bound: 0.0550675
NS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0546074, upper bound: 0.0552825
NS_A1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0536236, upper bound: 0.0551141
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0546114, upper bound: 0.0553942
NS_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0536922, upper bound: 0.0550306
NS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0546804, upper bound: 0.0551591
NS_A1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0537327, upper bound: 0.0551335
NS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0546919, upper bound: 0.0554038
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0544417, upper bound: 0.0552160
NS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0545095, upper bound: 0.0551352
NS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0550691, upper bound: 0.0555526
NS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0550691, upper bound: 0.0555597
NS_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556167
NS_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556167
NS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556173
NS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0550900, upper bound: 0.0556248
NS_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
NS_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
NS_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
NS_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0552506, upper bound: 0.0550900
NS_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555548
NS_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555568
NS_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555548
NS_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 1.34
Output dim: 0, lower bound: -0.0552530, upper bound: 0.0555568

## BFS NS instance: NS_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0220630, 0.0224854, -0.0259937, 0.0298868, -0.0519498, 0.0484791
1: -0.0235168, 0.0437383, -0.0261583, 0.0675495, -0.0910664, 0.0698966
2: -0.0539281, 0.0298981, -0.0751536, 0.0479900, -0.1019180, 0.1050517
3: -0.0377854, 0.0553166, -0.0417194, 0.0987438, -0.1365292, 0.0970360
4: -0.0649626, 0.0354342, -0.1131465, 0.0579570, -0.1229195, 0.1485806

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0543371, upper bound: 0.0552825
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0543371, upper bound: 0.0552825
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0220630, 0.0224854, -0.0267285, 0.0305231, -0.0525861, 0.0492139
1: -0.0235168, 0.0437383, -0.0281868, 0.0701722, -0.0936890, 0.0719251
2: -0.0539281, 0.0298981, -0.0768717, 0.0496046, -0.1035327, 0.1067698
3: -0.0377854, 0.0553166, -0.0452320, 0.1035235, -0.1413089, 0.1005487
4: -0.0649626, 0.0354342, -0.1163596, 0.0609572, -0.1259197, 0.1517938

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0543013, upper bound: 0.0553799
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0543013, upper bound: 0.0553001
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0173346, 0.0176596, -0.0259937, 0.0298868, -0.0472214, 0.0436533
1: -0.0178321, 0.0313102, -0.0261583, 0.0675495, -0.0853816, 0.0574685
2: -0.0455050, 0.0226024, -0.0751536, 0.0479900, -0.0934950, 0.0977561
3: -0.0306701, 0.0403801, -0.0417194, 0.0987438, -0.1294139, 0.0820995
4: -0.0569695, 0.0274540, -0.1131465, 0.0579570, -0.1149265, 0.1406004

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543155, upper bound: 0.0532552
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0543155, upper bound: 0.0551591
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0173346, 0.0176596, -0.0267285, 0.0305231, -0.0478577, 0.0443881
1: -0.0178321, 0.0313102, -0.0281868, 0.0701722, -0.0880042, 0.0594970
2: -0.0455050, 0.0226024, -0.0768717, 0.0496046, -0.0951096, 0.0994741
3: -0.0306701, 0.0403801, -0.0452320, 0.1035235, -0.1341936, 0.0856121
4: -0.0569695, 0.0274540, -0.1163596, 0.0609572, -0.1179267, 0.1438136

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543572, upper bound: 0.0536569
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0543572, upper bound: 0.0554038
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0187671, 0.0190559, -0.0119187, 0.0089635, -0.0277306, 0.0309746
1: -0.0186150, 0.0335206, -0.0112788, 0.0104788, -0.0290939, 0.0447994
2: -0.0464860, 0.0238866, -0.0260187, 0.0060609, -0.0525469, 0.0499054
3: -0.0316599, 0.0410530, -0.0227056, 0.0075995, -0.0392594, 0.0637586
4: -0.0554192, 0.0285984, -0.0251642, 0.0067536, -0.0621728, 0.0537626

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544414, upper bound: 0.0551680
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544397, upper bound: 0.0552041
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0174403, 0.0183881, -0.0205188, 0.0206299, -0.0380701, 0.0389069
1: -0.0179365, 0.0346897, -0.0210706, 0.0388912, -0.0568277, 0.0557603
2: -0.0468226, 0.0235523, -0.0522289, 0.0274310, -0.0742536, 0.0757812
3: -0.0312331, 0.0437208, -0.0347407, 0.0506525, -0.0818856, 0.0784616
4: -0.0597628, 0.0287614, -0.0663188, 0.0334479, -0.0932107, 0.0950802

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550691, upper bound: 0.0555526
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550691, upper bound: 0.0555526
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0174403, 0.0183881, -0.0405656, 0.0413909, -0.0588312, 0.0589536
1: -0.0179365, 0.0346897, -0.0567703, 0.1031073, -0.1210438, 0.0914600
2: -0.0468226, 0.0235523, -0.0949313, 0.0645219, -0.1113445, 0.1184836
3: -0.0312331, 0.0437208, -0.0783715, 0.1450140, -0.1762472, 0.1220923
4: -0.0597628, 0.0287614, -0.1297568, 0.0740883, -0.1338511, 0.1585183

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547091, upper bound: 0.0534031
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550329, upper bound: 0.0555436
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0277988, 0.0316155, -0.0168666, 0.0177314, -0.0455302, 0.0484821
1: -0.0329395, 0.0745442, -0.0166844, 0.0324948, -0.0654343, 0.0912286
2: -0.0723793, 0.0529296, -0.0457053, 0.0225558, -0.0949352, 0.0986349
3: -0.0475005, 0.0975785, -0.0294895, 0.0407429, -0.0882434, 0.1270680
4: -0.0997654, 0.0589750, -0.0583189, 0.0276646, -0.1274301, 0.1172940

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547768, upper bound: 0.0555901
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550420, upper bound: 0.0554034
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0277988, 0.0316155, -0.0177528, 0.0200240, -0.0478228, 0.0493684
1: -0.0329395, 0.0745442, -0.0189958, 0.0363287, -0.0692682, 0.0935401
2: -0.0723793, 0.0529296, -0.0453282, 0.0254624, -0.0978417, 0.0982578
3: -0.0475005, 0.0975785, -0.0320662, 0.0460069, -0.0935075, 0.1296447
4: -0.0997654, 0.0589750, -0.0614046, 0.0284768, -0.1282422, 0.1203796

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547768, upper bound: 0.0555901
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550420, upper bound: 0.0554455
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0277988, 0.0316155, -0.0397151, 0.0488666, -0.0766654, 0.0713306
1: -0.0329395, 0.0745442, -0.0569106, 0.1248108, -0.1577503, 0.1314548
2: -0.0723793, 0.0529296, -0.0939133, 0.0682135, -0.1405928, 0.1468429
3: -0.0475005, 0.0975785, -0.0793627, 0.1845545, -0.2320550, 0.1769412
4: -0.0997654, 0.0589750, -0.1542809, 0.0764101, -0.1761755, 0.2132559

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551255, upper bound: 0.0535317
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551592, upper bound: 0.0556013
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0277988, 0.0316155, -0.0232206, 0.0272127, -0.0550115, 0.0548362
1: -0.0329395, 0.0745442, -0.0260053, 0.0594450, -0.0923846, 0.1005495
2: -0.0723793, 0.0529296, -0.0611328, 0.0432778, -0.1156571, 0.1140624
3: -0.0475005, 0.0975785, -0.0381173, 0.0764618, -0.1239623, 0.1356958
4: -0.0997654, 0.0589750, -0.0869171, 0.0466825, -0.1464479, 0.1458921

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551255, upper bound: 0.0537097
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551592, upper bound: 0.0556074
time: 0.24 seconds

## BFS NS instance: NS_A2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0174403, 0.0183881, -0.0361409, 0.0374643
1: -0.0189958, 0.0363287, -0.0179365, 0.0346897, -0.0536855, 0.0542652
2: -0.0453282, 0.0254624, -0.0468226, 0.0235523, -0.0688805, 0.0722850
3: -0.0320662, 0.0460069, -0.0312331, 0.0437208, -0.0757870, 0.0772400
4: -0.0614046, 0.0284768, -0.0597628, 0.0287614, -0.0901660, 0.0882396

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0193350, 0.0216215, -0.0393743, 0.0393590
1: -0.0189958, 0.0363287, -0.0221682, 0.0413110, -0.0603069, 0.0584969
2: -0.0453282, 0.0254624, -0.0484735, 0.0279975, -0.0733257, 0.0739359
3: -0.0320662, 0.0460069, -0.0360067, 0.0530757, -0.0851419, 0.0820136
4: -0.0614046, 0.0284768, -0.0654610, 0.0313325, -0.0927371, 0.0939378

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_A2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0414968, 0.0538694, -0.0716222, 0.0615209
1: -0.0189958, 0.0363287, -0.0741096, 0.1330418, -0.1520376, 0.1104383
2: -0.0453282, 0.0254624, -0.0972371, 0.0890556, -0.1343838, 0.1226994
3: -0.0320662, 0.0460069, -0.1205643, 0.1977571, -0.2298233, 0.1665712
4: -0.0614046, 0.0284768, -0.1608542, 0.1174676, -0.1788722, 0.1893310

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A1_B2_B1_B1

### Relational analysis result of NS_A2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555005, upper bound: 0.0547001
time: 0.22 seconds

## Relational analysis of NS_A2_A2_A1_B2_B1_B2

### Relational analysis result of NS_A2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554390, upper bound: 0.0550420
time: 0.22 seconds

## BFS NS instance: NS_A2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0237000, 0.0276457, -0.0453985, 0.0437240
1: -0.0189958, 0.0363287, -0.0270187, 0.0608168, -0.0798126, 0.0633474
2: -0.0453282, 0.0254624, -0.0619147, 0.0439293, -0.0892575, 0.0873770
3: -0.0320662, 0.0460069, -0.0392698, 0.0784101, -0.1104763, 0.0852767
4: -0.0614046, 0.0284768, -0.0880449, 0.0474173, -0.1088219, 0.1165217

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A1_B2_B2_B1

### Relational analysis result of NS_A2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555005, upper bound: 0.0547001
time: 0.23 seconds

## Relational analysis of NS_A2_A2_A1_B2_B2_B2

### Relational analysis result of NS_A2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554390, upper bound: 0.0550420
time: 0.21 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0244671, 0.0285040, -0.0174403, 0.0183881, -0.0428552, 0.0459443
1: -0.0291369, 0.0633202, -0.0179365, 0.0346897, -0.0638266, 0.0812567
2: -0.0632677, 0.0448866, -0.0468226, 0.0235523, -0.0868200, 0.0917093
3: -0.0417759, 0.0822587, -0.0312331, 0.0437208, -0.0854968, 0.1134919
4: -0.0902513, 0.0482991, -0.0597628, 0.0287614, -0.1190127, 0.1080619

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0555164
time: 0.23 seconds

## Relational analysis of NS_A2_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554712, upper bound: 0.0553575
time: 0.22 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0244671, 0.0285040, -0.0277988, 0.0316155, -0.0560826, 0.0563028
1: -0.0291369, 0.0633202, -0.0329395, 0.0745442, -0.1036811, 0.0962597
2: -0.0632677, 0.0448866, -0.0723793, 0.0529296, -0.1161973, 0.1172660
3: -0.0417759, 0.0822587, -0.0475005, 0.0975785, -0.1393544, 0.1297593
4: -0.0902513, 0.0482991, -0.0997654, 0.0589750, -0.1492263, 0.1480646

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B1_B2_B1

### Relational analysis result of NS_A2_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0536906, upper bound: 0.0551355
time: 0.24 seconds

## Relational analysis of NS_A2_A2_A2_B1_B2_B2

### Relational analysis result of NS_A2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556323, upper bound: 0.0555439
time: 0.25 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0244671, 0.0285040, -0.0193350, 0.0216215, -0.0460886, 0.0478390
1: -0.0291369, 0.0633202, -0.0221682, 0.0413110, -0.0704480, 0.0854884
2: -0.0632677, 0.0448866, -0.0484735, 0.0279975, -0.0912652, 0.0933601
3: -0.0417759, 0.0822587, -0.0360067, 0.0530757, -0.0948516, 0.1182654
4: -0.0902513, 0.0482991, -0.0654610, 0.0313325, -0.1215838, 0.1137601

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547001, upper bound: 0.0555005
time: 0.24 seconds

## Relational analysis of NS_A2_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552050, upper bound: 0.0554084
time: 0.24 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0244671, 0.0285040, -0.0249527, 0.0289441, -0.0534112, 0.0534567
1: -0.0291369, 0.0633202, -0.0302124, 0.0647534, -0.0938904, 0.0935325
2: -0.0632677, 0.0448866, -0.0640738, 0.0455330, -0.1088007, 0.1089605
3: -0.0417759, 0.0822587, -0.0429836, 0.0844141, -0.1261900, 0.1252424
4: -0.0902513, 0.0482991, -0.0914269, 0.0490248, -0.1392761, 0.1397261

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549152, upper bound: 0.0538035
time: 0.24 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552177, upper bound: 0.0555375
time: 0.24 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.31 seconds
NS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0543371, upper bound: 0.0552825
NS_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0543371, upper bound: 0.0552825
NS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0543013, upper bound: 0.0553799
NS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0543013, upper bound: 0.0553001
NS_A1_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0543155, upper bound: 0.0532552
NS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0543155, upper bound: 0.0551591
NS_A1_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0543572, upper bound: 0.0536569
NS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0543572, upper bound: 0.0554038
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0544414, upper bound: 0.0551680
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0544397, upper bound: 0.0552041
NS_A1_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0550691, upper bound: 0.0555526
NS_A1_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0550691, upper bound: 0.0555526
NS_A1_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0547091, upper bound: 0.0534031
NS_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0550329, upper bound: 0.0555436
NS_A1_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0547768, upper bound: 0.0555901
NS_A1_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0550420, upper bound: 0.0554034
NS_A1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0547768, upper bound: 0.0555901
NS_A1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0550420, upper bound: 0.0554455
NS_A1_B2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0551255, upper bound: 0.0535317
NS_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0551592, upper bound: 0.0556013
NS_A1_B2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0551255, upper bound: 0.0537097
NS_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0551592, upper bound: 0.0556074
NS_A2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0555005, upper bound: 0.0547001
NS_A2_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0554390, upper bound: 0.0550420
NS_A2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0555005, upper bound: 0.0547001
NS_A2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0554390, upper bound: 0.0550420
NS_A2_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0555164
NS_A2_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0554712, upper bound: 0.0553575
NS_A2_A2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0536906, upper bound: 0.0551355
NS_A2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0556323, upper bound: 0.0555439
NS_A2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0547001, upper bound: 0.0555005
NS_A2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0552050, upper bound: 0.0554084
NS_A2_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0549152, upper bound: 0.0538035
NS_A2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.31
Output dim: 0, lower bound: -0.0552177, upper bound: 0.0555375

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0219276, 0.0224061, -0.0259937, 0.0298868, -0.0518144, 0.0483998
1: -0.0232139, 0.0435903, -0.0261583, 0.0675495, -0.0907635, 0.0697486
2: -0.0535728, 0.0301348, -0.0751536, 0.0479900, -0.1015628, 0.1052884
3: -0.0373126, 0.0552721, -0.0417194, 0.0987438, -0.1360564, 0.0969915
4: -0.0653377, 0.0353545, -0.1131465, 0.0579570, -0.1232947, 0.1485010

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0213500, 0.0216952, -0.0259937, 0.0298868, -0.0512368, 0.0476889
1: -0.0225845, 0.0414368, -0.0261583, 0.0675495, -0.0901341, 0.0675952
2: -0.0523226, 0.0283392, -0.0751536, 0.0479900, -0.1003126, 0.1034928
3: -0.0366326, 0.0521505, -0.0417194, 0.0987438, -0.1353764, 0.0938699
4: -0.0626164, 0.0337324, -0.1131465, 0.0579570, -0.1205734, 0.1468789

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0219276, 0.0224061, -0.0267285, 0.0305231, -0.0524507, 0.0491346
1: -0.0232139, 0.0435903, -0.0281868, 0.0701722, -0.0933861, 0.0717771
2: -0.0535728, 0.0301348, -0.0768717, 0.0496046, -0.1031775, 0.1070064
3: -0.0373126, 0.0552721, -0.0452320, 0.1035235, -0.1408362, 0.1005042
4: -0.0653377, 0.0353545, -0.1163596, 0.0609572, -0.1262949, 0.1517141

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0213500, 0.0216952, -0.0267285, 0.0305231, -0.0518731, 0.0484237
1: -0.0225845, 0.0414368, -0.0281868, 0.0701722, -0.0927567, 0.0696237
2: -0.0523226, 0.0283392, -0.0768717, 0.0496046, -0.1019273, 0.1052109
3: -0.0366326, 0.0521505, -0.0452320, 0.1035235, -0.1401561, 0.0973826
4: -0.0626164, 0.0337324, -0.1163596, 0.0609572, -0.1235736, 0.1500920

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0165583, 0.0167663, -0.0259937, 0.0298868, -0.0464450, 0.0427600
1: -0.0165932, 0.0292139, -0.0261583, 0.0675495, -0.0841427, 0.0553723
2: -0.0437887, 0.0207630, -0.0751536, 0.0479900, -0.0917787, 0.0959167
3: -0.0290741, 0.0371772, -0.0417194, 0.0987438, -0.1278179, 0.0788966
4: -0.0545131, 0.0254438, -0.1131465, 0.0579570, -0.1124700, 0.1385903

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0530213, upper bound: 0.0550523
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0530213, upper bound: 0.0551591
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0165583, 0.0167663, -0.0267285, 0.0305231, -0.0470814, 0.0434948
1: -0.0165932, 0.0292139, -0.0281868, 0.0701722, -0.0867654, 0.0574008
2: -0.0437887, 0.0207630, -0.0768717, 0.0496046, -0.0933933, 0.0976347
3: -0.0290741, 0.0371772, -0.0452320, 0.1035235, -0.1325976, 0.0824092
4: -0.0545131, 0.0254438, -0.1163596, 0.0609572, -0.1154702, 0.1418034

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0530213, upper bound: 0.0552629
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0530213, upper bound: 0.0552655
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0187671, 0.0190559, -0.0104673, 0.0080033, -0.0267705, 0.0295233
1: -0.0186150, 0.0335206, -0.0101913, 0.0072679, -0.0258829, 0.0437119
2: -0.0464860, 0.0238866, -0.0221414, 0.0055417, -0.0520277, 0.0460280
3: -0.0316599, 0.0410530, -0.0217378, 0.0019941, -0.0336541, 0.0627908
4: -0.0554192, 0.0285984, -0.0208521, 0.0052320, -0.0606512, 0.0494505

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0541265, upper bound: 0.0551680
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0541265, upper bound: 0.0551680
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0187671, 0.0190559, -0.0115633, 0.0087682, -0.0275353, 0.0306192
1: -0.0186150, 0.0335206, -0.0109297, 0.0096126, -0.0282276, 0.0444504
2: -0.0464860, 0.0238866, -0.0248758, 0.0059768, -0.0524629, 0.0487624
3: -0.0316599, 0.0410530, -0.0225022, 0.0062176, -0.0378775, 0.0635552
4: -0.0554192, 0.0285984, -0.0238930, 0.0064142, -0.0618335, 0.0524914

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0541265, upper bound: 0.0552019
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0541265, upper bound: 0.0552041
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0174403, 0.0183881, -0.0161676, 0.0166442, -0.0340845, 0.0345557
1: -0.0179365, 0.0346897, -0.0154069, 0.0299797, -0.0479161, 0.0500966
2: -0.0468226, 0.0235523, -0.0443634, 0.0210207, -0.0678433, 0.0679157
3: -0.0312331, 0.0437208, -0.0278074, 0.0378903, -0.0691235, 0.0715283
4: -0.0597628, 0.0287614, -0.0565481, 0.0260608, -0.0858236, 0.0853096

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549088, upper bound: 0.0555211
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550226, upper bound: 0.0552799
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0174403, 0.0183881, -0.0157226, 0.0173771, -0.0348174, 0.0341107
1: -0.0179365, 0.0346897, -0.0152636, 0.0291742, -0.0471107, 0.0499533
2: -0.0468226, 0.0235523, -0.0407109, 0.0206817, -0.0675044, 0.0642632
3: -0.0312331, 0.0437208, -0.0271661, 0.0362221, -0.0674553, 0.0708870
4: -0.0597628, 0.0287614, -0.0546506, 0.0232683, -0.0830311, 0.0834120

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547091, upper bound: 0.0534011
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550329, upper bound: 0.0555350
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0167507, 0.0175735, -0.0405656, 0.0413909, -0.0581416, 0.0581391
1: -0.0169593, 0.0324132, -0.0567703, 0.1031073, -0.1200666, 0.0891835
2: -0.0453734, 0.0220414, -0.0949313, 0.0645219, -0.1098952, 0.1169726
3: -0.0299786, 0.0405239, -0.0783715, 0.1450140, -0.1749926, 0.1188954
4: -0.0578682, 0.0270928, -0.1297568, 0.0740883, -0.1319565, 0.1568496

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553983, upper bound: 0.0555436
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553983, upper bound: 0.0555436
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0278452, 0.0310977, -0.0168666, 0.0177314, -0.0455766, 0.0479643
1: -0.0329159, 0.0712517, -0.0166844, 0.0324948, -0.0654107, 0.0879361
2: -0.0701464, 0.0509891, -0.0457053, 0.0225558, -0.0927022, 0.0966944
3: -0.0480456, 0.0928499, -0.0294895, 0.0407429, -0.0887885, 0.1223394
4: -0.0942536, 0.0571052, -0.0583189, 0.0276646, -0.1219182, 0.1154242

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545802, upper bound: 0.0551114
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0555258
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0240019, 0.0272771, -0.0168666, 0.0177314, -0.0417332, 0.0441437
1: -0.0260409, 0.0594983, -0.0166844, 0.0324948, -0.0585357, 0.0761827
2: -0.0639335, 0.0455851, -0.0457053, 0.0225558, -0.0864893, 0.0912904
3: -0.0402265, 0.0769224, -0.0294895, 0.0407429, -0.0809694, 0.1064119
4: -0.0868411, 0.0507081, -0.0583189, 0.0276646, -0.1145057, 0.1090271

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550350, upper bound: 0.0536319
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0553742
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0278452, 0.0310977, -0.0177528, 0.0200240, -0.0478692, 0.0488505
1: -0.0329159, 0.0712517, -0.0189958, 0.0363287, -0.0692446, 0.0902475
2: -0.0701464, 0.0509891, -0.0453282, 0.0254624, -0.0956087, 0.0963173
3: -0.0480456, 0.0928499, -0.0320662, 0.0460069, -0.0940525, 0.1249161
4: -0.0942536, 0.0571052, -0.0614046, 0.0284768, -0.1227304, 0.1185098

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546678, upper bound: 0.0555879
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547768, upper bound: 0.0555454
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0240019, 0.0272771, -0.0177528, 0.0200240, -0.0440259, 0.0450300
1: -0.0260409, 0.0594983, -0.0189958, 0.0363287, -0.0623696, 0.0784941
2: -0.0639335, 0.0455851, -0.0453282, 0.0254624, -0.0893959, 0.0909133
3: -0.0402265, 0.0769224, -0.0320662, 0.0460069, -0.0862335, 0.1089887
4: -0.0868411, 0.0507081, -0.0614046, 0.0284768, -0.1153179, 0.1121127

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547900, upper bound: 0.0536888
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549929, upper bound: 0.0554125
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0273027, 0.0310323, -0.0397151, 0.0488666, -0.0761693, 0.0707474
1: -0.0318687, 0.0727345, -0.0569106, 0.1248108, -0.1566795, 0.1296450
2: -0.0712754, 0.0519350, -0.0939133, 0.0682135, -0.1394889, 0.1458483
3: -0.0462107, 0.0947695, -0.0793627, 0.1845545, -0.2307652, 0.1741322
4: -0.0982611, 0.0578686, -0.1542809, 0.0764101, -0.1746711, 0.2121494

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0555970
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0553158
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0273027, 0.0310323, -0.0232206, 0.0272127, -0.0545154, 0.0542529
1: -0.0318687, 0.0727345, -0.0260053, 0.0594450, -0.0913137, 0.0987398
2: -0.0712754, 0.0519350, -0.0611328, 0.0432778, -0.1145532, 0.1130678
3: -0.0462107, 0.0947695, -0.0381173, 0.0764618, -0.1226725, 0.1328868
4: -0.0982611, 0.0578686, -0.0869171, 0.0466825, -0.1449436, 0.1447857

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549915, upper bound: 0.0555815
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550773, upper bound: 0.0554409
time: 0.21 seconds

## BFS NS instance: NS_A2_A2_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0393321, 0.0495266, -0.0672794, 0.0593561
1: -0.0189958, 0.0363287, -0.0687150, 0.1214286, -0.1404245, 0.1050437
2: -0.0453282, 0.0254624, -0.0912258, 0.0822822, -0.1276104, 0.1166882
3: -0.0320662, 0.0460069, -0.1112330, 0.1786075, -0.2106737, 0.1572399
4: -0.0614046, 0.0284768, -0.1461978, 0.1076009, -0.1690055, 0.1746745

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_A2_A2_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0362481, 0.0464668, -0.0642196, 0.0562721
1: -0.0189958, 0.0363287, -0.0602394, 0.1100473, -0.1290431, 0.0965680
2: -0.0453282, 0.0254624, -0.0875491, 0.0741147, -0.1194429, 0.1130114
3: -0.0320662, 0.0460069, -0.0969803, 0.1631809, -0.1952471, 0.1429872
4: -0.0614046, 0.0284768, -0.1404332, 0.0949922, -0.1563968, 0.1689100

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_A2_A2_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0252150, 0.0284831, -0.0462359, 0.0452390
1: -0.0189958, 0.0363287, -0.0279653, 0.0610138, -0.0800096, 0.0642940
2: -0.0453282, 0.0254624, -0.0635368, 0.0439139, -0.0892421, 0.0889991
3: -0.0320662, 0.0460069, -0.0413184, 0.0779307, -0.1099969, 0.0873253
4: -0.0614046, 0.0284768, -0.0858417, 0.0480413, -0.1094459, 0.1143185

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A1_B2_B2_B1_B1

### Relational analysis result of NS_A2_A2_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555005, upper bound: 0.0546650
time: 0.21 seconds

## Relational analysis of NS_A2_A2_A1_B2_B2_B1_B2

### Relational analysis result of NS_A2_A2_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554746, upper bound: 0.0547001
time: 0.19 seconds

## BFS NS instance: NS_A2_A2_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0198437, 0.0232866, -0.0410394, 0.0398677
1: -0.0189958, 0.0363287, -0.0199278, 0.0458330, -0.0648288, 0.0562564
2: -0.0453282, 0.0254624, -0.0528894, 0.0365279, -0.0818561, 0.0783518
3: -0.0320662, 0.0460069, -0.0318742, 0.0581797, -0.0902459, 0.0778811
4: -0.0614046, 0.0284768, -0.0742503, 0.0389823, -0.1003869, 0.1027271

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B2_B2_B2_B1

### Relational analysis result of NS_A2_A2_A1_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0537197, upper bound: 0.0547859
time: 0.23 seconds

## Relational analysis of NS_A2_A2_A1_B2_B2_B2_B2

### Relational analysis result of NS_A2_A2_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554089, upper bound: 0.0549929
time: 0.23 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0257471, 0.0290832, -0.0174403, 0.0183881, -0.0441351, 0.0465234
1: -0.0298784, 0.0626745, -0.0179365, 0.0346897, -0.0645681, 0.0806109
2: -0.0645730, 0.0442843, -0.0468226, 0.0235523, -0.0881253, 0.0911069
3: -0.0435096, 0.0804226, -0.0312331, 0.0437208, -0.0872304, 0.1116557
4: -0.0875694, 0.0483158, -0.0597628, 0.0287614, -0.1163309, 0.1080786

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_A2_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0545035, upper bound: 0.0550418
time: 0.22 seconds

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0554553
time: 0.25 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0199965, 0.0235820, -0.0174403, 0.0183881, -0.0383846, 0.0410223
1: -0.0205243, 0.0465149, -0.0179365, 0.0346897, -0.0552140, 0.0644514
2: -0.0532895, 0.0369669, -0.0468226, 0.0235523, -0.0768418, 0.0837895
3: -0.0326785, 0.0591032, -0.0312331, 0.0437208, -0.0763993, 0.0903363
4: -0.0750140, 0.0393477, -0.0597628, 0.0287614, -0.1037754, 0.0991105

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551352, upper bound: 0.0545095
time: 0.25 seconds

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553554, upper bound: 0.0552699
time: 0.23 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0244671, 0.0285040, -0.0273027, 0.0310323, -0.0554994, 0.0558067
1: -0.0291369, 0.0633202, -0.0318687, 0.0727345, -0.1018714, 0.0951889
2: -0.0632677, 0.0448866, -0.0712754, 0.0519350, -0.1152027, 0.1161620
3: -0.0417759, 0.0822587, -0.0462107, 0.0947695, -0.1365454, 0.1284695
4: -0.0902513, 0.0482991, -0.0982611, 0.0578686, -0.1481199, 0.1465602

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B1

### Relational analysis result of NS_A2_A2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0556139, upper bound: 0.0550682
time: 0.22 seconds

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B2

### Relational analysis result of NS_A2_A2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554704, upper bound: 0.0553158
time: 0.22 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0257471, 0.0290832, -0.0193350, 0.0216215, -0.0473686, 0.0484181
1: -0.0298784, 0.0626745, -0.0221682, 0.0413110, -0.0711894, 0.0848427
2: -0.0645730, 0.0442843, -0.0484735, 0.0279975, -0.0925706, 0.0927578
3: -0.0435096, 0.0804226, -0.0360067, 0.0530757, -0.0965853, 0.1164293
4: -0.0875694, 0.0483158, -0.0654610, 0.0313325, -0.1189020, 0.1137768

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_A2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546652, upper bound: 0.0555005
time: 0.21 seconds

## Relational analysis of NS_A2_A2_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_A2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547001, upper bound: 0.0554553
time: 0.21 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0199965, 0.0235820, -0.0193350, 0.0216215, -0.0416180, 0.0429170
1: -0.0205243, 0.0465149, -0.0221682, 0.0413110, -0.0618353, 0.0686831
2: -0.0532895, 0.0369669, -0.0484735, 0.0279975, -0.0812870, 0.0854404
3: -0.0326785, 0.0591032, -0.0360067, 0.0530757, -0.0857542, 0.0951099
4: -0.0750140, 0.0393477, -0.0654610, 0.0313325, -0.1063465, 0.1048087

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_A2_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548698, upper bound: 0.0537801
time: 0.24 seconds

## Relational analysis of NS_A2_A2_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551723, upper bound: 0.0553811
time: 0.23 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0238193, 0.0278017, -0.0249527, 0.0289441, -0.0527634, 0.0527544
1: -0.0280888, 0.0614425, -0.0302124, 0.0647534, -0.0928423, 0.0916549
2: -0.0620518, 0.0437358, -0.0640738, 0.0455330, -0.1075848, 0.1078096
3: -0.0403428, 0.0793698, -0.0429836, 0.0844141, -0.1247569, 0.1223534
4: -0.0885203, 0.0470289, -0.0914269, 0.0490248, -0.1375451, 0.1384558

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555769, upper bound: 0.0549915
time: 0.23 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554704, upper bound: 0.0553811
time: 0.22 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 1.37 seconds
NS_A1_B1_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0530213, upper bound: 0.0550523
NS_A1_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0530213, upper bound: 0.0551591
NS_A1_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0530213, upper bound: 0.0552629
NS_A1_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0530213, upper bound: 0.0552655
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0541265, upper bound: 0.0551680
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0541265, upper bound: 0.0551680
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0541265, upper bound: 0.0552019
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0541265, upper bound: 0.0552041
NS_A1_B2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0549088, upper bound: 0.0555211
NS_A1_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0550226, upper bound: 0.0552799
NS_A1_B2_A1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0547091, upper bound: 0.0534011
NS_A1_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0550329, upper bound: 0.0555350
NS_A1_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0553983, upper bound: 0.0555436
NS_A1_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0553983, upper bound: 0.0555436
NS_A1_B2_A2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0545802, upper bound: 0.0551114
NS_A1_B2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0555258
NS_A1_B2_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0550350, upper bound: 0.0536319
NS_A1_B2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0553742
NS_A1_B2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0546678, upper bound: 0.0555879
NS_A1_B2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0547768, upper bound: 0.0555454
NS_A1_B2_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0547900, upper bound: 0.0536888
NS_A1_B2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0549929, upper bound: 0.0554125
NS_A1_B2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0555970
NS_A1_B2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0553158
NS_A1_B2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0549915, upper bound: 0.0555815
NS_A1_B2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0550773, upper bound: 0.0554409
NS_A2_A2_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0555005, upper bound: 0.0546650
NS_A2_A2_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0554746, upper bound: 0.0547001
NS_A2_A2_A1_B2_B2_B2_B1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0537197, upper bound: 0.0547859
NS_A2_A2_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0554089, upper bound: 0.0549929
NS_A2_A2_A2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0545035, upper bound: 0.0550418
NS_A2_A2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0554553
NS_A2_A2_A2_B1_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0551352, upper bound: 0.0545095
NS_A2_A2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0553554, upper bound: 0.0552699
NS_A2_A2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0556139, upper bound: 0.0550682
NS_A2_A2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0554704, upper bound: 0.0553158
NS_A2_A2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0546652, upper bound: 0.0555005
NS_A2_A2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0547001, upper bound: 0.0554553
NS_A2_A2_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0548698, upper bound: 0.0537801
NS_A2_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0551723, upper bound: 0.0553811
NS_A2_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0555769, upper bound: 0.0549915
NS_A2_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.37
Output dim: 0, lower bound: -0.0554704, upper bound: 0.0553811

## BFS NS instance: NS_A1_B1_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0160824, 0.0162786, -0.0259937, 0.0298868, -0.0459692, 0.0422723
1: -0.0158827, 0.0281735, -0.0261583, 0.0675495, -0.0834322, 0.0543318
2: -0.0426353, 0.0196426, -0.0751536, 0.0479900, -0.0906252, 0.0947962
3: -0.0282152, 0.0356028, -0.0417194, 0.0987438, -0.1269590, 0.0773221
4: -0.0529485, 0.0241707, -0.1131465, 0.0579570, -0.1109054, 0.1373171

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

## BFS NS instance: NS_A1_B1_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0163773, 0.0165949, -0.0267285, 0.0305231, -0.0469004, 0.0433234
1: -0.0162556, 0.0290263, -0.0281868, 0.0701722, -0.0864278, 0.0572131
2: -0.0434588, 0.0206079, -0.0768717, 0.0496046, -0.0930634, 0.0974796
3: -0.0285381, 0.0368503, -0.0452320, 0.1035235, -0.1320616, 0.0820824
4: -0.0543467, 0.0250632, -0.1163596, 0.0609572, -0.1153039, 0.1414228

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

## BFS NS instance: NS_A1_B1_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0160824, 0.0162786, -0.0267285, 0.0305231, -0.0466056, 0.0430072
1: -0.0158827, 0.0281735, -0.0281868, 0.0701722, -0.0860549, 0.0563603
2: -0.0426353, 0.0196426, -0.0768717, 0.0496046, -0.0922399, 0.0965142
3: -0.0282152, 0.0356028, -0.0452320, 0.1035235, -0.1317387, 0.0808348
4: -0.0529485, 0.0241707, -0.1163596, 0.0609572, -0.1139056, 0.1405303

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0184445, 0.0186504, -0.0104673, 0.0080033, -0.0264479, 0.0291177
1: -0.0177631, 0.0323133, -0.0101913, 0.0072679, -0.0250310, 0.0425046
2: -0.0456871, 0.0236643, -0.0221414, 0.0055417, -0.0512288, 0.0458057
3: -0.0303963, 0.0395534, -0.0217378, 0.0019941, -0.0323904, 0.0612912
4: -0.0552228, 0.0280121, -0.0208521, 0.0052320, -0.0604547, 0.0488642

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0182772, 0.0185430, -0.0104673, 0.0080033, -0.0262805, 0.0290104
1: -0.0179854, 0.0321531, -0.0101913, 0.0072679, -0.0252533, 0.0423444
2: -0.0453839, 0.0229184, -0.0221414, 0.0055417, -0.0509256, 0.0450599
3: -0.0308918, 0.0392026, -0.0217378, 0.0019941, -0.0328859, 0.0609404
4: -0.0541408, 0.0274676, -0.0208521, 0.0052320, -0.0593728, 0.0483197

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0184445, 0.0186504, -0.0115633, 0.0087682, -0.0272127, 0.0302137
1: -0.0177631, 0.0323133, -0.0109297, 0.0096126, -0.0273757, 0.0432430
2: -0.0456871, 0.0236643, -0.0248758, 0.0059768, -0.0516639, 0.0485401
3: -0.0303963, 0.0395534, -0.0225022, 0.0062176, -0.0366139, 0.0620556
4: -0.0552228, 0.0280121, -0.0238930, 0.0064142, -0.0616370, 0.0519051

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0182772, 0.0185430, -0.0115633, 0.0087682, -0.0270454, 0.0301063
1: -0.0179854, 0.0321531, -0.0109297, 0.0096126, -0.0275980, 0.0430829
2: -0.0453839, 0.0229184, -0.0248758, 0.0059768, -0.0513608, 0.0477943
3: -0.0308918, 0.0392026, -0.0225022, 0.0062176, -0.0371094, 0.0617048
4: -0.0541408, 0.0274676, -0.0238930, 0.0064142, -0.0605551, 0.0513606

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0188422, 0.0191683, -0.0161676, 0.0166442, -0.0354864, 0.0353359
1: -0.0187277, 0.0338746, -0.0154069, 0.0299797, -0.0487073, 0.0492815
2: -0.0467068, 0.0241366, -0.0443634, 0.0210207, -0.0677276, 0.0685001
3: -0.0318076, 0.0415401, -0.0278074, 0.0378903, -0.0696979, 0.0693475
4: -0.0558105, 0.0288630, -0.0565481, 0.0260608, -0.0818712, 0.0854112

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551795, upper bound: 0.0551795
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551795, upper bound: 0.0552799
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0148684, 0.0150387, -0.0161676, 0.0166442, -0.0315127, 0.0312063
1: -0.0138386, 0.0261036, -0.0154069, 0.0299797, -0.0438182, 0.0415105
2: -0.0399446, 0.0177537, -0.0443634, 0.0210207, -0.0609653, 0.0621171
3: -0.0261081, 0.0320604, -0.0278074, 0.0378903, -0.0639984, 0.0598679
4: -0.0496877, 0.0220269, -0.0565481, 0.0260608, -0.0757484, 0.0785751

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552799, upper bound: 0.0551795
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552799, upper bound: 0.0552799
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0167507, 0.0175735, -0.0157226, 0.0173771, -0.0341278, 0.0332961
1: -0.0169593, 0.0324132, -0.0152636, 0.0291742, -0.0461335, 0.0476768
2: -0.0453734, 0.0220414, -0.0407109, 0.0206817, -0.0660551, 0.0627523
3: -0.0299786, 0.0405239, -0.0271661, 0.0362221, -0.0662007, 0.0676900
4: -0.0578682, 0.0270928, -0.0546506, 0.0232683, -0.0811366, 0.0817434

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545248, upper bound: 0.0555350
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550329, upper bound: 0.0555155
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0167507, 0.0175735, -0.0382298, 0.0395165, -0.0562672, 0.0558034
1: -0.0169593, 0.0324132, -0.0538139, 0.0975074, -0.1144666, 0.0862271
2: -0.0453734, 0.0220414, -0.0899229, 0.0613182, -0.1066916, 0.1119643
3: -0.0299786, 0.0405239, -0.0748341, 0.1366450, -0.1666235, 0.1153581
4: -0.0578682, 0.0270928, -0.1234836, 0.0702202, -0.1280884, 0.1505764

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549565, upper bound: 0.0555436
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553620, upper bound: 0.0554881
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0167507, 0.0175735, -0.0222534, 0.0252567, -0.0420074, 0.0398269
1: -0.0169593, 0.0324132, -0.0241738, 0.0545417, -0.0715010, 0.0565870
2: -0.0453734, 0.0220414, -0.0590635, 0.0398606, -0.0852340, 0.0811049
3: -0.0299786, 0.0405239, -0.0359105, 0.0701916, -0.1001701, 0.0764344
4: -0.0578682, 0.0270928, -0.0829497, 0.0435772, -0.1014455, 0.1100425

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549565, upper bound: 0.0555436
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553620, upper bound: 0.0555155
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0278452, 0.0310977, -0.0163317, 0.0171610, -0.0450062, 0.0474294
1: -0.0329159, 0.0712517, -0.0160104, 0.0310019, -0.0639178, 0.0872621
2: -0.0701464, 0.0509891, -0.0445348, 0.0214143, -0.0915607, 0.0955240
3: -0.0480456, 0.0928499, -0.0286694, 0.0387396, -0.0867852, 0.1215193
4: -0.0942536, 0.0571052, -0.0567923, 0.0263709, -0.1206245, 0.1138975

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0554483
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0555258
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0234873, 0.0266846, -0.0168666, 0.0177314, -0.0412187, 0.0435512
1: -0.0250003, 0.0577051, -0.0166844, 0.0324948, -0.0574952, 0.0743895
2: -0.0628079, 0.0445847, -0.0457053, 0.0225558, -0.0853637, 0.0902900
3: -0.0388779, 0.0740361, -0.0294895, 0.0407429, -0.0796207, 0.1035256
4: -0.0853493, 0.0495821, -0.0583189, 0.0276646, -0.1130139, 0.1079010

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0552904
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0552904
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0278679, 0.0321473, -0.0177528, 0.0200240, -0.0478919, 0.0499001
1: -0.0329683, 0.0750322, -0.0189958, 0.0363287, -0.0692969, 0.0940280
2: -0.0702526, 0.0525257, -0.0453282, 0.0254624, -0.0957149, 0.0978539
3: -0.0474031, 0.0969662, -0.0320662, 0.0460069, -0.0934100, 0.1290324
4: -0.0962531, 0.0578964, -0.0614046, 0.0284768, -0.1247298, 0.1193010

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0273508, 0.0306702, -0.0177528, 0.0200240, -0.0473748, 0.0484230
1: -0.0321755, 0.0699852, -0.0189958, 0.0363287, -0.0685042, 0.0889811
2: -0.0691589, 0.0502736, -0.0453282, 0.0254624, -0.0946212, 0.0956018
3: -0.0471808, 0.0909255, -0.0320662, 0.0460069, -0.0931877, 0.1229917
4: -0.0932161, 0.0562352, -0.0614046, 0.0284768, -0.1216929, 0.1176398

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0234873, 0.0266846, -0.0177528, 0.0200240, -0.0435114, 0.0444374
1: -0.0250003, 0.0577051, -0.0189958, 0.0363287, -0.0613290, 0.0767009
2: -0.0628079, 0.0445847, -0.0453282, 0.0254624, -0.0882702, 0.0899129
3: -0.0388779, 0.0740361, -0.0320662, 0.0460069, -0.0848848, 0.1061023
4: -0.0853493, 0.0495821, -0.0614046, 0.0284768, -0.1138261, 0.1109867

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0546241, upper bound: 0.0552821
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549924, upper bound: 0.0554125
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0270626, 0.0304198, -0.0397151, 0.0488666, -0.0759293, 0.0701350
1: -0.0316554, 0.0691338, -0.0569106, 0.1248108, -0.1564662, 0.1260444
2: -0.0684051, 0.0499098, -0.0939133, 0.0682135, -0.1366186, 0.1438231
3: -0.0462560, 0.0896830, -0.0793627, 0.1845545, -0.2308105, 0.1690457
4: -0.0926542, 0.0556223, -0.1542809, 0.0764101, -0.1690642, 0.2099032

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0550682
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0553158
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0234873, 0.0266846, -0.0397151, 0.0488666, -0.0723540, 0.0663997
1: -0.0250003, 0.0577051, -0.0569106, 0.1248108, -0.1498111, 0.1146157
2: -0.0628079, 0.0445847, -0.0939133, 0.0682135, -0.1310214, 0.1384980
3: -0.0388779, 0.0740361, -0.0793627, 0.1845545, -0.2234323, 0.1533988
4: -0.0853493, 0.0495821, -0.1542809, 0.0764101, -0.1617594, 0.2038630

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549470, upper bound: 0.0552091
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552813, upper bound: 0.0553153
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0270626, 0.0304198, -0.0232206, 0.0272127, -0.0542753, 0.0536405
1: -0.0316554, 0.0691338, -0.0260053, 0.0594450, -0.0911004, 0.0951391
2: -0.0684051, 0.0499098, -0.0611328, 0.0432778, -0.1116829, 0.1110425
3: -0.0462560, 0.0896830, -0.0381173, 0.0764618, -0.1227178, 0.1278003
4: -0.0926542, 0.0556223, -0.0869171, 0.0466825, -0.1393366, 0.1425394

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549915, upper bound: 0.0549915
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549915, upper bound: 0.0554409
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0234873, 0.0266846, -0.0232206, 0.0272127, -0.0507001, 0.0499052
1: -0.0250003, 0.0577051, -0.0260053, 0.0594450, -0.0844454, 0.0837104
2: -0.0628079, 0.0445847, -0.0611328, 0.0432778, -0.1060857, 0.1057174
3: -0.0388779, 0.0740361, -0.0381173, 0.0764618, -0.1153396, 0.1121534
4: -0.0853493, 0.0495821, -0.0869171, 0.0466825, -0.1320318, 0.1364992

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550773, upper bound: 0.0549915
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550773, upper bound: 0.0554409
time: 0.22 seconds

## BFS NS instance: NS_A2_A2_A1_B2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0253816, 0.0294349, -0.0471877, 0.0454056
1: -0.0189958, 0.0363287, -0.0282665, 0.0643411, -0.0833370, 0.0645952
2: -0.0453282, 0.0254624, -0.0635668, 0.0452627, -0.0905909, 0.0890291
3: -0.0320662, 0.0460069, -0.0410936, 0.0820307, -0.1140969, 0.0871005
4: -0.0614046, 0.0284768, -0.0872937, 0.0487742, -0.1101788, 0.1157705

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A1_B2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

## BFS NS instance: NS_A2_A2_A1_B2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0247393, 0.0279610, -0.0457139, 0.0447633
1: -0.0189958, 0.0363287, -0.0272721, 0.0596626, -0.0786585, 0.0636007
2: -0.0453282, 0.0254624, -0.0625226, 0.0431439, -0.0884721, 0.0879850
3: -0.0320662, 0.0460069, -0.0404192, 0.0758799, -0.1079461, 0.0864261
4: -0.0614046, 0.0284768, -0.0845790, 0.0471494, -0.1085540, 0.1130558

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A2_A2_A1_B2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0191731, 0.0226172, -0.0403701, 0.0391971
1: -0.0189958, 0.0363287, -0.0188379, 0.0438878, -0.0628836, 0.0551666
2: -0.0453282, 0.0254624, -0.0516788, 0.0353989, -0.0807271, 0.0771412
3: -0.0320662, 0.0460069, -0.0302866, 0.0553400, -0.0874062, 0.0762935
4: -0.0614046, 0.0284768, -0.0726424, 0.0376871, -0.0990917, 0.1011191

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A1_B2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A1_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A1_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A1_B2_B2_B2_B2_B1

### Relational analysis result of NS_A2_A2_A1_B2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553229, upper bound: 0.0546241
time: 0.25 seconds

## Relational analysis of NS_A2_A2_A1_B2_B2_B2_B2_B2

### Relational analysis result of NS_A2_A2_A1_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554089, upper bound: 0.0549924
time: 0.22 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0257471, 0.0290832, -0.0169136, 0.0178243, -0.0435713, 0.0459968
1: -0.0298784, 0.0626745, -0.0172633, 0.0331967, -0.0630751, 0.0799378
2: -0.0645730, 0.0442843, -0.0456765, 0.0224354, -0.0870084, 0.0899608
3: -0.0435096, 0.0804226, -0.0304100, 0.0416375, -0.0851471, 0.1108326
4: -0.0875694, 0.0483158, -0.0582723, 0.0274961, -0.1150656, 0.1065881

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2_B1

### Relational analysis result of NS_A2_A2_A2_B1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0530189, upper bound: 0.0550242
time: 0.22 seconds

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2_B2

### Relational analysis result of NS_A2_A2_A2_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549575, upper bound: 0.0554296
time: 0.21 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0193978, 0.0229309, -0.0174403, 0.0183881, -0.0377859, 0.0403711
1: -0.0194044, 0.0443665, -0.0179365, 0.0346897, -0.0540941, 0.0623030
2: -0.0521740, 0.0359274, -0.0468226, 0.0235523, -0.0757263, 0.0827500
3: -0.0312126, 0.0560299, -0.0312331, 0.0437208, -0.0749335, 0.0872630
4: -0.0734783, 0.0381431, -0.0597628, 0.0287614, -0.1022398, 0.0979059

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553554, upper bound: 0.0551811
time: 0.24 seconds

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553554, upper bound: 0.0551811
time: 0.23 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0244671, 0.0285040, -0.0270626, 0.0304198, -0.0548869, 0.0555666
1: -0.0291369, 0.0633202, -0.0316554, 0.0691338, -0.0982708, 0.0949756
2: -0.0632677, 0.0448866, -0.0684051, 0.0499098, -0.1131775, 0.1132917
3: -0.0417759, 0.0822587, -0.0462560, 0.0896830, -0.1314589, 0.1285148
4: -0.0902513, 0.0482991, -0.0926542, 0.0556223, -0.1458736, 0.1409533

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B1_A1

### Relational analysis result of NS_A2_A2_A2_B1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0550682
time: 0.23 seconds

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B1_A2

### Relational analysis result of NS_A2_A2_A2_B1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0550682
time: 0.23 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0244671, 0.0285040, -0.0234873, 0.0266846, -0.0511517, 0.0519914
1: -0.0291369, 0.0633202, -0.0250003, 0.0577051, -0.0868420, 0.0883205
2: -0.0632677, 0.0448866, -0.0628079, 0.0445847, -0.1078524, 0.1076945
3: -0.0417759, 0.0822587, -0.0388779, 0.0740361, -0.1158120, 0.1211366
4: -0.0902513, 0.0482991, -0.0853493, 0.0495821, -0.1398334, 0.1336484

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B2_B1

### Relational analysis result of NS_A2_A2_A2_B1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553535, upper bound: 0.0549446
time: 0.20 seconds

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B2_B2

### Relational analysis result of NS_A2_A2_A2_B1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554704, upper bound: 0.0553153
time: 0.21 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0263041, 0.0304260, -0.0193350, 0.0216215, -0.0479256, 0.0497610
1: -0.0312516, 0.0671478, -0.0221682, 0.0413110, -0.0725627, 0.0893161
2: -0.0651119, 0.0461771, -0.0484735, 0.0279975, -0.0931095, 0.0946506
3: -0.0443822, 0.0865055, -0.0360067, 0.0530757, -0.0974579, 0.1225122
4: -0.0899085, 0.0495911, -0.0654610, 0.0313325, -0.1212410, 0.1150521

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0252294, 0.0285154, -0.0193350, 0.0216215, -0.0468509, 0.0478504
1: -0.0291304, 0.0612422, -0.0221682, 0.0413110, -0.0704415, 0.0834105
2: -0.0634498, 0.0434587, -0.0484735, 0.0279975, -0.0914473, 0.0919322
3: -0.0425654, 0.0783339, -0.0360067, 0.0530757, -0.0956411, 0.1143406
4: -0.0861801, 0.0473629, -0.0654610, 0.0313325, -0.1175127, 0.1128239

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A2_A2_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0193136, 0.0229116, -0.0193350, 0.0216215, -0.0409351, 0.0422466
1: -0.0194592, 0.0445171, -0.0221682, 0.0413110, -0.0607703, 0.0666853
2: -0.0520530, 0.0358496, -0.0484735, 0.0279975, -0.0800506, 0.0843231
3: -0.0311160, 0.0561660, -0.0360067, 0.0530757, -0.0841917, 0.0921727
4: -0.0733854, 0.0380535, -0.0654610, 0.0313325, -0.1047179, 0.1035145

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A2_B2_B1_A2_A2_A1

### Relational analysis result of NS_A2_A2_A2_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548068, upper bound: 0.0545359
time: 0.23 seconds

## Relational analysis of NS_A2_A2_A2_B2_B1_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550455, upper bound: 0.0552477
time: 0.22 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0238193, 0.0278017, -0.0262929, 0.0295320, -0.0533512, 0.0540947
1: -0.0280888, 0.0614425, -0.0309200, 0.0643362, -0.0924250, 0.0923625
2: -0.0620518, 0.0437358, -0.0655613, 0.0449164, -0.1069682, 0.1092971
3: -0.0403428, 0.0793698, -0.0446447, 0.0830512, -0.1233940, 0.1240145
4: -0.0885203, 0.0470289, -0.0888226, 0.0490772, -0.1375975, 0.1358515

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B1

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555769, upper bound: 0.0549529
time: 0.23 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B2

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555197, upper bound: 0.0549915
time: 0.23 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0238193, 0.0278017, -0.0204420, 0.0239954, -0.0478147, 0.0482437
1: -0.0280888, 0.0614425, -0.0214377, 0.0480555, -0.0761443, 0.0828802
2: -0.0620518, 0.0437358, -0.0541373, 0.0375680, -0.0996197, 0.0978731
3: -0.0403428, 0.0793698, -0.0337864, 0.0613243, -0.1016671, 0.1131562
4: -0.0885203, 0.0470289, -0.0761260, 0.0400765, -0.1285968, 0.1231548

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B2_A1

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0551300, upper bound: 0.0546322
time: 0.21 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B2_A2

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553438, upper bound: 0.0552459
time: 0.22 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 1.16 seconds
NS_A1_B2_A1_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0551795, upper bound: 0.0551795
NS_A1_B2_A1_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0551795, upper bound: 0.0552799
NS_A1_B2_A1_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0552799, upper bound: 0.0551795
NS_A1_B2_A1_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0552799, upper bound: 0.0552799
NS_A1_B2_A1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0545248, upper bound: 0.0555350
NS_A1_B2_A1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0550329, upper bound: 0.0555155
NS_A1_B2_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0549565, upper bound: 0.0555436
NS_A1_B2_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0553620, upper bound: 0.0554881
NS_A1_B2_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0549565, upper bound: 0.0555436
NS_A1_B2_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0553620, upper bound: 0.0555155
NS_A1_B2_A2_B1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0554483
NS_A1_B2_A2_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0550581, upper bound: 0.0555258
NS_A1_B2_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0552904
NS_A1_B2_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0552818, upper bound: 0.0552904
NS_A1_B2_A2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0546241, upper bound: 0.0552821
NS_A1_B2_A2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0549924, upper bound: 0.0554125
NS_A1_B2_A2_B2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0550682
NS_A1_B2_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0553158
NS_A1_B2_A2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0549470, upper bound: 0.0552091
NS_A1_B2_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0552813, upper bound: 0.0553153
NS_A1_B2_A2_B2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0549915, upper bound: 0.0549915
NS_A1_B2_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0549915, upper bound: 0.0554409
NS_A1_B2_A2_B2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0550773, upper bound: 0.0549915
NS_A1_B2_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0550773, upper bound: 0.0554409
NS_A2_A2_A1_B2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0553229, upper bound: 0.0546241
NS_A2_A2_A1_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0554089, upper bound: 0.0549924
NS_A2_A2_A2_B1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0530189, upper bound: 0.0550242
NS_A2_A2_A2_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0549575, upper bound: 0.0554296
NS_A2_A2_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0553554, upper bound: 0.0551811
NS_A2_A2_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0553554, upper bound: 0.0551811
NS_A2_A2_A2_B1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0550682
NS_A2_A2_A2_B1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0550682
NS_A2_A2_A2_B1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0553535, upper bound: 0.0549446
NS_A2_A2_A2_B1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0554704, upper bound: 0.0553153
NS_A2_A2_A2_B2_B1_A2_A2_A1, status: Status.VERIFIED, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0548068, upper bound: 0.0545359
NS_A2_A2_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0550455, upper bound: 0.0552477
NS_A2_A2_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0555769, upper bound: 0.0549529
NS_A2_A2_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0555197, upper bound: 0.0549915
NS_A2_A2_A2_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0551300, upper bound: 0.0546322
NS_A2_A2_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 1.16
Output dim: 0, lower bound: -0.0553438, upper bound: 0.0552459

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0188422, 0.0191683, -0.0172572, 0.0167018, -0.0355440, 0.0364255
1: -0.0187277, 0.0338746, -0.0157410, 0.0275783, -0.0463060, 0.0496156
2: -0.0467068, 0.0241366, -0.0431397, 0.0202850, -0.0669919, 0.0672764
3: -0.0318076, 0.0415401, -0.0279668, 0.0336190, -0.0654266, 0.0695069
4: -0.0558105, 0.0288630, -0.0513968, 0.0250097, -0.0808202, 0.0802599

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546554, upper bound: 0.0551482
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551795, upper bound: 0.0555098
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0188422, 0.0191683, -0.0138982, 0.0124814, -0.0313235, 0.0330666
1: -0.0187277, 0.0338746, -0.0127218, 0.0235826, -0.0423103, 0.0465964
2: -0.0467068, 0.0241366, -0.0372503, 0.0153892, -0.0620960, 0.0613870
3: -0.0318076, 0.0415401, -0.0248601, 0.0280820, -0.0598896, 0.0664002
4: -0.0558105, 0.0288630, -0.0460164, 0.0191962, -0.0750067, 0.0748795

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0546554, upper bound: 0.0551482
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551795, upper bound: 0.0555563
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0148684, 0.0150387, -0.0172572, 0.0167018, -0.0315702, 0.0322958
1: -0.0138386, 0.0261036, -0.0157410, 0.0275783, -0.0414169, 0.0418446
2: -0.0399446, 0.0177537, -0.0431397, 0.0202850, -0.0602296, 0.0608934
3: -0.0261081, 0.0320604, -0.0279668, 0.0336190, -0.0597271, 0.0600272
4: -0.0496877, 0.0220269, -0.0513968, 0.0250097, -0.0746974, 0.0734238

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547762, upper bound: 0.0546554
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552799, upper bound: 0.0551795
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0148684, 0.0150387, -0.0138982, 0.0124814, -0.0273498, 0.0289369
1: -0.0138386, 0.0261036, -0.0127218, 0.0235826, -0.0374211, 0.0388254
2: -0.0399446, 0.0177537, -0.0372503, 0.0153892, -0.0553337, 0.0550040
3: -0.0261081, 0.0320604, -0.0248601, 0.0280820, -0.0541901, 0.0569206
4: -0.0496877, 0.0220269, -0.0460164, 0.0191962, -0.0688839, 0.0680434

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547512, upper bound: 0.0547762
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552799, upper bound: 0.0551795
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0167907, 0.0173994, -0.0157226, 0.0173771, -0.0341678, 0.0331220
1: -0.0166599, 0.0318810, -0.0152636, 0.0291742, -0.0458341, 0.0471446
2: -0.0451706, 0.0220867, -0.0407109, 0.0206817, -0.0658523, 0.0627976
3: -0.0294806, 0.0401354, -0.0271661, 0.0362221, -0.0657027, 0.0673016
4: -0.0579042, 0.0269196, -0.0546506, 0.0232683, -0.0811725, 0.0815702

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0157226, 0.0173771, -0.0336772, 0.0328259
1: -0.0163531, 0.0312095, -0.0152636, 0.0291742, -0.0455274, 0.0464732
2: -0.0443946, 0.0211404, -0.0407109, 0.0206817, -0.0650763, 0.0618513
3: -0.0292436, 0.0388854, -0.0271661, 0.0362221, -0.0654657, 0.0660515
4: -0.0566717, 0.0260474, -0.0546506, 0.0232683, -0.0799401, 0.0806980

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0167907, 0.0173994, -0.0382298, 0.0395165, -0.0563072, 0.0556292
1: -0.0166599, 0.0318810, -0.0538139, 0.0975074, -0.1141673, 0.0856949
2: -0.0451706, 0.0220867, -0.0899229, 0.0613182, -0.1064888, 0.1120096
3: -0.0294806, 0.0401354, -0.0748341, 0.1366450, -0.1661255, 0.1149696
4: -0.0579042, 0.0269196, -0.1234836, 0.0702202, -0.1281244, 0.1504032

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550393, upper bound: 0.0551658
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550393, upper bound: 0.0554881
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0382298, 0.0395165, -0.0558167, 0.0553331
1: -0.0163531, 0.0312095, -0.0538139, 0.0975074, -0.1138605, 0.0850234
2: -0.0443946, 0.0211404, -0.0899229, 0.0613182, -0.1057128, 0.1110633
3: -0.0292436, 0.0388854, -0.0748341, 0.1366450, -0.1658885, 0.1137195
4: -0.0566717, 0.0260474, -0.1234836, 0.0702202, -0.1268919, 0.1495310

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554456, upper bound: 0.0551658
time: 0.27 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0554456, upper bound: 0.0554881
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0167907, 0.0173994, -0.0222534, 0.0252567, -0.0420474, 0.0396527
1: -0.0166599, 0.0318810, -0.0241738, 0.0545417, -0.0712016, 0.0560548
2: -0.0451706, 0.0220867, -0.0590635, 0.0398606, -0.0850312, 0.0811502
3: -0.0294806, 0.0401354, -0.0359105, 0.0701916, -0.0996721, 0.0760459
4: -0.0579042, 0.0269196, -0.0829497, 0.0435772, -0.1014814, 0.1098693

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0527904, upper bound: 0.0550688
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0527904, upper bound: 0.0555436
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0222534, 0.0252567, -0.0415568, 0.0393567
1: -0.0163531, 0.0312095, -0.0241738, 0.0545417, -0.0708949, 0.0553834
2: -0.0443946, 0.0211404, -0.0590635, 0.0398606, -0.0842552, 0.0802039
3: -0.0292436, 0.0388854, -0.0359105, 0.0701916, -0.0994351, 0.0747959
4: -0.0566717, 0.0260474, -0.0829497, 0.0435772, -0.1002490, 0.1089971

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533702, upper bound: 0.0550753
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0533702, upper bound: 0.0555155
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0278452, 0.0310977, -0.0178932, 0.0180688, -0.0459140, 0.0489910
1: -0.0329159, 0.0712517, -0.0170138, 0.0303990, -0.0633149, 0.0882656
2: -0.0701464, 0.0509891, -0.0446541, 0.0223315, -0.0924779, 0.0956432
3: -0.0480456, 0.0928499, -0.0295371, 0.0369350, -0.0849806, 0.1223870
4: -0.0942536, 0.0571052, -0.0533013, 0.0268344, -0.1210880, 0.1104065

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0551114
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0551230
time: 0.27 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0278452, 0.0310977, -0.0140397, 0.0128757, -0.0407209, 0.0451374
1: -0.0329159, 0.0712517, -0.0128741, 0.0241912, -0.0571072, 0.0841258
2: -0.0701464, 0.0509891, -0.0378803, 0.0158756, -0.0860220, 0.0888695
3: -0.0480456, 0.0928499, -0.0250210, 0.0290569, -0.0771025, 0.1178709
4: -0.0942536, 0.0571052, -0.0468782, 0.0198709, -0.1141245, 0.1039835

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0555258
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0551230
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0234873, 0.0266846, -0.0183918, 0.0185941, -0.0420814, 0.0450764
1: -0.0250003, 0.0577051, -0.0176752, 0.0318211, -0.0568215, 0.0753803
2: -0.0628079, 0.0445847, -0.0457768, 0.0233121, -0.0861200, 0.0903615
3: -0.0388779, 0.0740361, -0.0303477, 0.0388390, -0.0777168, 0.1043838
4: -0.0853493, 0.0495821, -0.0546228, 0.0279766, -0.1133259, 0.1042049

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547199, upper bound: 0.0549097
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552813, upper bound: 0.0552904
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0234873, 0.0266846, -0.0144305, 0.0141843, -0.0376717, 0.0411151
1: -0.0250003, 0.0577051, -0.0131581, 0.0251122, -0.0501126, 0.0708632
2: -0.0628079, 0.0445847, -0.0389022, 0.0168341, -0.0796420, 0.0834868
3: -0.0388779, 0.0740361, -0.0252906, 0.0304506, -0.0693285, 0.0993267
4: -0.0853493, 0.0495821, -0.0482360, 0.0209963, -0.1063456, 0.0978181

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547199, upper bound: 0.0549097
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552813, upper bound: 0.0552904
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0236056, 0.0276349, -0.0177528, 0.0200240, -0.0436296, 0.0453877
1: -0.0246522, 0.0609175, -0.0189958, 0.0363287, -0.0609808, 0.0799134
2: -0.0631518, 0.0459309, -0.0453282, 0.0254624, -0.0886142, 0.0912591
3: -0.0380466, 0.0772544, -0.0320662, 0.0460069, -0.0840535, 0.1093206
4: -0.0868571, 0.0502552, -0.0614046, 0.0284768, -0.1153339, 0.1116598

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0230536, 0.0262401, -0.0177528, 0.0200240, -0.0430776, 0.0439930
1: -0.0242822, 0.0563962, -0.0189958, 0.0363287, -0.0606109, 0.0753921
2: -0.0619228, 0.0438159, -0.0453282, 0.0254624, -0.0873852, 0.0891441
3: -0.0380059, 0.0721582, -0.0320662, 0.0460069, -0.0840129, 0.1042244
4: -0.0842291, 0.0486820, -0.0614046, 0.0284768, -0.1127059, 0.1100866

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0270626, 0.0304198, -0.0346156, 0.0423562, -0.0694188, 0.0650355
1: -0.0316554, 0.0691338, -0.0472905, 0.1025978, -0.1342532, 0.1164244
2: -0.0684051, 0.0499098, -0.0845582, 0.0593169, -0.1277221, 0.1344680
3: -0.0462560, 0.0896830, -0.0671641, 0.1508321, -0.1970881, 0.1568471
4: -0.0926542, 0.0556223, -0.1345043, 0.0665109, -0.1591651, 0.1901266

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549505, upper bound: 0.0555821
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0554742
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0236056, 0.0276349, -0.0397151, 0.0488666, -0.0724722, 0.0673500
1: -0.0246522, 0.0609175, -0.0569106, 0.1248108, -0.1494630, 0.1178281
2: -0.0631518, 0.0459309, -0.0939133, 0.0682135, -0.1313653, 0.1398442
3: -0.0380466, 0.0772544, -0.0793627, 0.1845545, -0.2226011, 0.1566171
4: -0.0868571, 0.0502552, -0.1542809, 0.0764101, -0.1632671, 0.2045361

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549470, upper bound: 0.0549621
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549470, upper bound: 0.0549621
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0230536, 0.0262401, -0.0397151, 0.0488666, -0.0719202, 0.0659553
1: -0.0242822, 0.0563962, -0.0569106, 0.1248108, -0.1490930, 0.1133068
2: -0.0619228, 0.0438159, -0.0939133, 0.0682135, -0.1301363, 0.1377291
3: -0.0380059, 0.0721582, -0.0793627, 0.1845545, -0.2225604, 0.1515209
4: -0.0842291, 0.0486820, -0.1542809, 0.0764101, -0.1606391, 0.2029628

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552813, upper bound: 0.0550682
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552813, upper bound: 0.0550682
time: 0.27 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0270626, 0.0304198, -0.0193921, 0.0228804, -0.0499430, 0.0498120
1: -0.0316554, 0.0691338, -0.0188992, 0.0442713, -0.0759267, 0.0880330
2: -0.0684051, 0.0499098, -0.0520622, 0.0359285, -0.1043336, 0.1019719
3: -0.0462560, 0.0896830, -0.0305969, 0.0559185, -0.1021745, 0.1202799
4: -0.0926542, 0.0556223, -0.0731249, 0.0382553, -0.1309095, 0.1287473

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548279, upper bound: 0.0549280
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549915, upper bound: 0.0555230
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0234873, 0.0266846, -0.0193921, 0.0228804, -0.0463677, 0.0460767
1: -0.0250003, 0.0577051, -0.0188992, 0.0442713, -0.0692716, 0.0766043
2: -0.0628079, 0.0445847, -0.0520622, 0.0359285, -0.0987364, 0.0966468
3: -0.0388779, 0.0740361, -0.0305969, 0.0559185, -0.0947963, 0.1046330
4: -0.0853493, 0.0495821, -0.0731249, 0.0382553, -0.1236046, 0.1227070

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549470, upper bound: 0.0548854
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550693, upper bound: 0.0549915
time: 0.26 seconds

## BFS NS instance: NS_A2_A2_A1_B2_B2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0194913, 0.0236550, -0.0414078, 0.0395153
1: -0.0189958, 0.0363287, -0.0191660, 0.0474881, -0.0664840, 0.0554947
2: -0.0453282, 0.0254624, -0.0523977, 0.0369112, -0.0822394, 0.0778601
3: -0.0320662, 0.0460069, -0.0302255, 0.0591861, -0.0912523, 0.0762324
4: -0.0614046, 0.0284768, -0.0745060, 0.0386603, -0.1000649, 0.1029827

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A1_B2_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B2_B2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_A1_B2_B2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0177528, 0.0200240, -0.0188194, 0.0222256, -0.0399784, 0.0388434
1: -0.0189958, 0.0363287, -0.0183187, 0.0428955, -0.0618914, 0.0546474
2: -0.0453282, 0.0254624, -0.0509736, 0.0347879, -0.0801161, 0.0764360
3: -0.0320662, 0.0460069, -0.0296517, 0.0539578, -0.0860240, 0.0756586
4: -0.0614046, 0.0284768, -0.0717398, 0.0369674, -0.0983720, 0.1002166

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A1_B2_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B2_B2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_A2_B1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0257471, 0.0290832, -0.0163001, 0.0171033, -0.0428504, 0.0453833
1: -0.0298784, 0.0626745, -0.0163531, 0.0312095, -0.0610879, 0.0790276
2: -0.0645730, 0.0442843, -0.0443946, 0.0211404, -0.0857135, 0.0886789
3: -0.0435096, 0.0804226, -0.0292436, 0.0388854, -0.0823949, 0.1096662
4: -0.0875694, 0.0483158, -0.0566717, 0.0260474, -0.1136168, 0.1049876

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2_B2_B1

### Relational analysis result of NS_A2_A2_A2_B1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549575, upper bound: 0.0553559
time: 0.24 seconds

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2_B2_B2

### Relational analysis result of NS_A2_A2_A2_B1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549575, upper bound: 0.0554296
time: 0.24 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0193978, 0.0229309, -0.0188422, 0.0191683, -0.0385662, 0.0417730
1: -0.0194044, 0.0443665, -0.0187277, 0.0338746, -0.0532790, 0.0630942
2: -0.0521740, 0.0359274, -0.0467068, 0.0241366, -0.0763106, 0.0826342
3: -0.0312126, 0.0560299, -0.0318076, 0.0415401, -0.0727527, 0.0878375
4: -0.0734783, 0.0381431, -0.0558105, 0.0288630, -0.1023414, 0.0939536

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548233, upper bound: 0.0547931
time: 0.28 seconds

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553554, upper bound: 0.0551811
time: 0.25 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0193978, 0.0229309, -0.0148684, 0.0150387, -0.0344365, 0.0377993
1: -0.0194044, 0.0443665, -0.0138386, 0.0261036, -0.0455080, 0.0582051
2: -0.0521740, 0.0359274, -0.0399446, 0.0177537, -0.0699277, 0.0758720
3: -0.0312126, 0.0560299, -0.0261081, 0.0320604, -0.0632730, 0.0821380
4: -0.0734783, 0.0381431, -0.0496877, 0.0220269, -0.0955053, 0.0878308

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548233, upper bound: 0.0547931
time: 0.24 seconds

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553554, upper bound: 0.0551811
time: 0.26 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0244671, 0.0285040, -0.0236056, 0.0276349, -0.0521019, 0.0521096
1: -0.0291369, 0.0633202, -0.0246522, 0.0609175, -0.0900545, 0.0879723
2: -0.0632677, 0.0448866, -0.0631518, 0.0459309, -0.1091986, 0.1080384
3: -0.0417759, 0.0822587, -0.0380466, 0.0772544, -0.1190303, 0.1203053
4: -0.0902513, 0.0482991, -0.0868571, 0.0502552, -0.1405065, 0.1351562

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B2_B1_A1

### Relational analysis result of NS_A2_A2_A2_B1_B2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548701, upper bound: 0.0549446
time: 0.25 seconds

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B2_B1_A2

### Relational analysis result of NS_A2_A2_A2_B1_B2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548701, upper bound: 0.0549446
time: 0.28 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0244671, 0.0285040, -0.0230536, 0.0262401, -0.0507072, 0.0515576
1: -0.0291369, 0.0633202, -0.0242822, 0.0563962, -0.0855332, 0.0876024
2: -0.0632677, 0.0448866, -0.0619228, 0.0438159, -0.1070836, 0.1068095
3: -0.0417759, 0.0822587, -0.0380059, 0.0721582, -0.1139341, 0.1202647
4: -0.0902513, 0.0482991, -0.0842291, 0.0486820, -0.1389333, 0.1325282

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B2_B2_A1

### Relational analysis result of NS_A2_A2_A2_B1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0553153
time: 0.29 seconds

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B2_B2_A2

### Relational analysis result of NS_A2_A2_A2_B1_B2_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0550682
time: 0.25 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0187581, 0.0223078, -0.0193350, 0.0216215, -0.0403796, 0.0416427
1: -0.0183857, 0.0426489, -0.0221682, 0.0413110, -0.0596968, 0.0648171
2: -0.0510134, 0.0348739, -0.0484735, 0.0279975, -0.0790109, 0.0833474
3: -0.0297386, 0.0535609, -0.0360067, 0.0530757, -0.0828143, 0.0895676
4: -0.0719108, 0.0369572, -0.0654610, 0.0313325, -0.1032434, 0.1024182

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A2_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A2_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A2_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A2_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B2_B1_A2_A2_A2_A1

### Relational analysis result of NS_A2_A2_A2_B2_B1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0545873, upper bound: 0.0552048
time: 0.28 seconds

## Relational analysis of NS_A2_A2_A2_B2_B1_A2_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2_B2_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0550455, upper bound: 0.0552477
time: 0.25 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0238193, 0.0278017, -0.0268412, 0.0308748, -0.0546941, 0.0546429
1: -0.0280888, 0.0614425, -0.0322746, 0.0686580, -0.0967469, 0.0937171
2: -0.0620518, 0.0437358, -0.0661133, 0.0468169, -0.1088687, 0.1098491
3: -0.0403428, 0.0793698, -0.0454444, 0.0886965, -0.1290393, 0.1248142
4: -0.0885203, 0.0470289, -0.0912204, 0.0503456, -0.1388660, 0.1382493

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552702, upper bound: 0.0549529
time: 0.28 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552702, upper bound: 0.0549529
time: 0.28 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0238193, 0.0278017, -0.0257546, 0.0289574, -0.0527767, 0.0535564
1: -0.0280888, 0.0614425, -0.0301562, 0.0628587, -0.0909475, 0.0915987
2: -0.0620518, 0.0437358, -0.0644181, 0.0440842, -0.1061360, 0.1081538
3: -0.0403428, 0.0793698, -0.0437083, 0.0808814, -0.1212242, 0.1230781
4: -0.0885203, 0.0470289, -0.0874114, 0.0481121, -0.1366324, 0.1344403

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550702, upper bound: 0.0549915
time: 0.26 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550702, upper bound: 0.0549915
time: 0.25 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0230710, 0.0270157, -0.0204420, 0.0239954, -0.0470664, 0.0474576
1: -0.0266865, 0.0593025, -0.0214377, 0.0480555, -0.0747420, 0.0807402
2: -0.0609307, 0.0425312, -0.0541373, 0.0375680, -0.0984986, 0.0966685
3: -0.0386650, 0.0763966, -0.0337864, 0.0613243, -0.0999893, 0.1101831
4: -0.0868642, 0.0457464, -0.0761260, 0.0400765, -0.1269407, 0.1218723

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548376, upper bound: 0.0552459
time: 0.26 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548376, upper bound: 0.0548377
time: 0.26 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 1.65 seconds
NS_A1_B2_A1_B2_B1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0546554, upper bound: 0.0551482
NS_A1_B2_A1_B2_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0551795, upper bound: 0.0555098
NS_A1_B2_A1_B2_B1_B1_A1_B2_B1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0546554, upper bound: 0.0551482
NS_A1_B2_A1_B2_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0551795, upper bound: 0.0555563
NS_A1_B2_A1_B2_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0547762, upper bound: 0.0546554
NS_A1_B2_A1_B2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0552799, upper bound: 0.0551795
NS_A1_B2_A1_B2_B1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0547512, upper bound: 0.0547762
NS_A1_B2_A1_B2_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0552799, upper bound: 0.0551795
NS_A1_B2_A1_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0550393, upper bound: 0.0551658
NS_A1_B2_A1_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0550393, upper bound: 0.0554881
NS_A1_B2_A1_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0554456, upper bound: 0.0551658
NS_A1_B2_A1_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0554456, upper bound: 0.0554881
NS_A1_B2_A1_B2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0527904, upper bound: 0.0550688
NS_A1_B2_A1_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0527904, upper bound: 0.0555436
NS_A1_B2_A1_B2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0533702, upper bound: 0.0550753
NS_A1_B2_A1_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0533702, upper bound: 0.0555155
NS_A1_B2_A2_B1_B1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0551114
NS_A1_B2_A2_B1_B1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0551230
NS_A1_B2_A2_B1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0555258
NS_A1_B2_A2_B1_B1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0544713, upper bound: 0.0551230
NS_A1_B2_A2_B1_B1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0547199, upper bound: 0.0549097
NS_A1_B2_A2_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0552813, upper bound: 0.0552904
NS_A1_B2_A2_B1_B1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0547199, upper bound: 0.0549097
NS_A1_B2_A2_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0552813, upper bound: 0.0552904
NS_A1_B2_A2_B2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0549505, upper bound: 0.0555821
NS_A1_B2_A2_B2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0550343, upper bound: 0.0554742
NS_A1_B2_A2_B2_B1_A2_A2_A1_B1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0549470, upper bound: 0.0549621
NS_A1_B2_A2_B2_B1_A2_A2_A1_B2, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0549470, upper bound: 0.0549621
NS_A1_B2_A2_B2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0552813, upper bound: 0.0550682
NS_A1_B2_A2_B2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0552813, upper bound: 0.0550682
NS_A1_B2_A2_B2_B2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0548279, upper bound: 0.0549280
NS_A1_B2_A2_B2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0549915, upper bound: 0.0555230
NS_A1_B2_A2_B2_B2_A2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0549470, upper bound: 0.0548854
NS_A1_B2_A2_B2_B2_A2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0550693, upper bound: 0.0549915
NS_A2_A2_A2_B1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0549575, upper bound: 0.0553559
NS_A2_A2_A2_B1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0549575, upper bound: 0.0554296
NS_A2_A2_A2_B1_B1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0548233, upper bound: 0.0547931
NS_A2_A2_A2_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0553554, upper bound: 0.0551811
NS_A2_A2_A2_B1_B1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0548233, upper bound: 0.0547931
NS_A2_A2_A2_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0553554, upper bound: 0.0551811
NS_A2_A2_A2_B1_B2_B2_B2_B1_A1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0548701, upper bound: 0.0549446
NS_A2_A2_A2_B1_B2_B2_B2_B1_A2, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0548701, upper bound: 0.0549446
NS_A2_A2_A2_B1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0553153
NS_A2_A2_A2_B1_B2_B2_B2_B2_A2, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0549914, upper bound: 0.0550682
NS_A2_A2_A2_B2_B1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0545873, upper bound: 0.0552048
NS_A2_A2_A2_B2_B1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0550455, upper bound: 0.0552477
NS_A2_A2_A2_B2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0552702, upper bound: 0.0549529
NS_A2_A2_A2_B2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0552702, upper bound: 0.0549529
NS_A2_A2_A2_B2_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0550702, upper bound: 0.0549915
NS_A2_A2_A2_B2_B2_A2_B1_B2_A2, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0550702, upper bound: 0.0549915
NS_A2_A2_A2_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0548376, upper bound: 0.0552459
NS_A2_A2_A2_B2_B2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 9, time: 1.65
Output dim: 0, lower bound: -0.0548376, upper bound: 0.0548377

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0188422, 0.0191683, -0.0168130, 0.0163271, -0.0351693, 0.0359814
1: -0.0187277, 0.0338746, -0.0151748, 0.0267792, -0.0455069, 0.0490494
2: -0.0467068, 0.0241366, -0.0420975, 0.0194794, -0.0661862, 0.0662341
3: -0.0318076, 0.0415401, -0.0273803, 0.0324003, -0.0642079, 0.0689204
4: -0.0558105, 0.0288630, -0.0502301, 0.0239952, -0.0798057, 0.0790931

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551833, upper bound: 0.0555098
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551833, upper bound: 0.0555098
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0188422, 0.0191683, -0.0135560, 0.0118012, -0.0306434, 0.0327244
1: -0.0187277, 0.0338746, -0.0124801, 0.0228618, -0.0415895, 0.0463548
2: -0.0467068, 0.0241366, -0.0364058, 0.0146392, -0.0613460, 0.0605424
3: -0.0318076, 0.0415401, -0.0246231, 0.0269644, -0.0587720, 0.0661631
4: -0.0558105, 0.0288630, -0.0449692, 0.0182736, -0.0740841, 0.0738323

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548114, upper bound: 0.0555563
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548114, upper bound: 0.0555563
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0144519, 0.0144736, -0.0172572, 0.0167018, -0.0311536, 0.0317307
1: -0.0134160, 0.0251981, -0.0157410, 0.0275783, -0.0409943, 0.0409391
2: -0.0389422, 0.0167805, -0.0431397, 0.0202850, -0.0592272, 0.0599202
3: -0.0256370, 0.0306983, -0.0279668, 0.0336190, -0.0592560, 0.0586651
4: -0.0483767, 0.0208874, -0.0513968, 0.0250097, -0.0733864, 0.0722842

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0550562, upper bound: 0.0532095
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555463, upper bound: 0.0551549
time: 0.27 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0148684, 0.0150387, -0.0135560, 0.0118012, -0.0266697, 0.0285947
1: -0.0138386, 0.0261036, -0.0124801, 0.0228618, -0.0367004, 0.0385838
2: -0.0399446, 0.0177537, -0.0364058, 0.0146392, -0.0545837, 0.0541595
3: -0.0261081, 0.0320604, -0.0246231, 0.0269644, -0.0530725, 0.0566835
4: -0.0496877, 0.0220269, -0.0449692, 0.0182736, -0.0679612, 0.0669962

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547762, upper bound: 0.0546554
time: 0.27 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0547762, upper bound: 0.0551795
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0167907, 0.0173994, -0.0390108, 0.0403708, -0.0571615, 0.0564101
1: -0.0166599, 0.0318810, -0.0562705, 0.1006465, -0.1173064, 0.0881515
2: -0.0451706, 0.0220867, -0.0906762, 0.0623315, -0.1075021, 0.1127629
3: -0.0294806, 0.0401354, -0.0781949, 0.1409077, -0.1703883, 0.1183303
4: -0.0579042, 0.0269196, -0.1248956, 0.0709120, -0.1288162, 0.1518152

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0526917, upper bound: 0.0550045
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0526917, upper bound: 0.0553539
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0167907, 0.0173994, -0.0377707, 0.0390805, -0.0558712, 0.0551700
1: -0.0166599, 0.0318810, -0.0531288, 0.0961922, -0.1128521, 0.0850098
2: -0.0451706, 0.0220867, -0.0889706, 0.0605443, -0.1057149, 0.1110573
3: -0.0294806, 0.0401354, -0.0739647, 0.1347134, -0.1641940, 0.1141002
4: -0.0579042, 0.0269196, -0.1222433, 0.0693109, -0.1272151, 0.1491629

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0526917, upper bound: 0.0550045
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0526917, upper bound: 0.0555709
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0390108, 0.0403708, -0.0566710, 0.0561141
1: -0.0163531, 0.0312095, -0.0562705, 0.1006465, -0.1169997, 0.0874800
2: -0.0443946, 0.0211404, -0.0906762, 0.0623315, -0.1067261, 0.1118166
3: -0.0292436, 0.0388854, -0.0781949, 0.1409077, -0.1701513, 0.1170803
4: -0.0566717, 0.0260474, -0.1248956, 0.0709120, -0.1275837, 0.1509430

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0531936, upper bound: 0.0549196
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0531936, upper bound: 0.0551658
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0377707, 0.0390805, -0.0553807, 0.0548740
1: -0.0163531, 0.0312095, -0.0531288, 0.0961922, -0.1125453, 0.0843383
2: -0.0443946, 0.0211404, -0.0889706, 0.0605443, -0.1049389, 0.1101111
3: -0.0292436, 0.0388854, -0.0739647, 0.1347134, -0.1639570, 0.1128501
4: -0.0566717, 0.0260474, -0.1222433, 0.0693109, -0.1259826, 0.1482907

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0531936, upper bound: 0.0549738
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0531936, upper bound: 0.0552184
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0167907, 0.0173994, -0.0215896, 0.0245916, -0.0413822, 0.0389889
1: -0.0166599, 0.0318810, -0.0230428, 0.0527036, -0.0693635, 0.0549237
2: -0.0451706, 0.0220867, -0.0578552, 0.0387179, -0.0838885, 0.0799420
3: -0.0294806, 0.0401354, -0.0343761, 0.0674634, -0.0969439, 0.0745115
4: -0.0579042, 0.0269196, -0.0812783, 0.0423099, -0.1002141, 0.1081979

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0527726, upper bound: 0.0553990
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0527726, upper bound: 0.0555436
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0215896, 0.0245916, -0.0408917, 0.0386929
1: -0.0163531, 0.0312095, -0.0230428, 0.0527036, -0.0690568, 0.0542523
2: -0.0443946, 0.0211404, -0.0578552, 0.0387179, -0.0831125, 0.0789957
3: -0.0292436, 0.0388854, -0.0343761, 0.0674634, -0.0967069, 0.0732614
4: -0.0566717, 0.0260474, -0.0812783, 0.0423099, -0.0989817, 0.1073257

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0532586, upper bound: 0.0552615
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0532586, upper bound: 0.0552623
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0278679, 0.0321473, -0.0140397, 0.0128757, -0.0407436, 0.0461869
1: -0.0329683, 0.0750322, -0.0128741, 0.0241912, -0.0571595, 0.0879063
2: -0.0702526, 0.0525257, -0.0378803, 0.0158756, -0.0861282, 0.0904060
3: -0.0474031, 0.0969662, -0.0250210, 0.0290569, -0.0764600, 0.1219872
4: -0.0962531, 0.0578964, -0.0468782, 0.0198709, -0.1161240, 0.1047746

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A1_B2_A2_B1_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0234873, 0.0266846, -0.0178932, 0.0180688, -0.0415561, 0.0445778
1: -0.0250003, 0.0577051, -0.0170138, 0.0303990, -0.0553994, 0.0747190
2: -0.0628079, 0.0445847, -0.0446541, 0.0223315, -0.0851394, 0.0892387
3: -0.0388779, 0.0740361, -0.0295371, 0.0369350, -0.0758129, 0.1035732
4: -0.0853493, 0.0495821, -0.0533013, 0.0268344, -0.1121837, 0.1028834

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551844, upper bound: 0.0551750
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2_B1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0551844, upper bound: 0.0552904
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A2_B1_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0234873, 0.0266846, -0.0140397, 0.0128757, -0.0363631, 0.0407243
1: -0.0250003, 0.0577051, -0.0128741, 0.0241912, -0.0491916, 0.0705792
2: -0.0628079, 0.0445847, -0.0378803, 0.0158756, -0.0786835, 0.0824650
3: -0.0388779, 0.0740361, -0.0250210, 0.0290569, -0.0679348, 0.0990571
4: -0.0853493, 0.0495821, -0.0468782, 0.0198709, -0.1052202, 0.0964603

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533650, upper bound: 0.0542646
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2_B2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B1_B1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0533650, upper bound: 0.0552804
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0272616, 0.0315264, -0.0346156, 0.0423562, -0.0696177, 0.0661421
1: -0.0318267, 0.0731649, -0.0472905, 0.1025978, -0.1344245, 0.1204555
2: -0.0690642, 0.0515455, -0.0845582, 0.0593169, -0.1283811, 0.1361037
3: -0.0456918, 0.0943532, -0.0671641, 0.1508321, -0.1965238, 0.1615172
4: -0.0947821, 0.0565235, -0.1345043, 0.0665109, -0.1612930, 0.1910278

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0530645, upper bound: 0.0550998
time: 0.27 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0530645, upper bound: 0.0555821
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0266347, 0.0300013, -0.0346156, 0.0423562, -0.0689908, 0.0646169
1: -0.0309553, 0.0678936, -0.0472905, 0.1025978, -0.1335531, 0.1151842
2: -0.0675894, 0.0492085, -0.0845582, 0.0593169, -0.1269063, 0.1337667
3: -0.0454101, 0.0878100, -0.0671641, 0.1508321, -0.1962422, 0.1549741
4: -0.0916399, 0.0547648, -0.1345043, 0.0665109, -0.1581509, 0.1892691

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0531590, upper bound: 0.0550959
time: 0.27 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0531590, upper bound: 0.0554742
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0230536, 0.0262401, -0.0377079, 0.0451381, -0.0681917, 0.0639481
1: -0.0242822, 0.0563962, -0.0533660, 0.1142647, -0.1385470, 0.1097622
2: -0.0619228, 0.0438159, -0.0882274, 0.0638561, -0.1257789, 0.1320432
3: -0.0380059, 0.0721582, -0.0744301, 0.1670350, -0.2050409, 0.1465883
4: -0.0842291, 0.0486820, -0.1403323, 0.0711920, -0.1554210, 0.1890142

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548753, upper bound: 0.0549390
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548753, upper bound: 0.0550528
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0230536, 0.0262401, -0.0346156, 0.0423562, -0.0654097, 0.0608558
1: -0.0242822, 0.0563962, -0.0472905, 0.1025978, -0.1268800, 0.1036868
2: -0.0619228, 0.0438159, -0.0845582, 0.0593169, -0.1212398, 0.1283741
3: -0.0380059, 0.0721582, -0.0671641, 0.1508321, -0.1888380, 0.1393223
4: -0.0842291, 0.0486820, -0.1345043, 0.0665109, -0.1507400, 0.1831863

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548753, upper bound: 0.0549390
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548753, upper bound: 0.0550528
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0270626, 0.0304198, -0.0190541, 0.0224912, -0.0495538, 0.0494739
1: -0.0316554, 0.0691338, -0.0184067, 0.0432951, -0.0749505, 0.0875405
2: -0.0684051, 0.0499098, -0.0513755, 0.0353198, -0.1037249, 0.1012853
3: -0.0462560, 0.0896830, -0.0299964, 0.0545656, -0.1008217, 0.1196795
4: -0.0926542, 0.0556223, -0.0722335, 0.0375377, -0.1301918, 0.1278558

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549505, upper bound: 0.0555230
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_B2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549505, upper bound: 0.0555230
time: 0.26 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0257471, 0.0290832, -0.0170747, 0.0169581, -0.0427052, 0.0461579
1: -0.0298784, 0.0626745, -0.0162292, 0.0278331, -0.0577115, 0.0789037
2: -0.0645730, 0.0442843, -0.0424308, 0.0201539, -0.0847269, 0.0867151
3: -0.0435096, 0.0804226, -0.0286224, 0.0335460, -0.0770556, 0.1090450
4: -0.0875694, 0.0483158, -0.0510356, 0.0244890, -0.1120584, 0.0993514

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A2_A2_A2_B1_B1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0543606, upper bound: 0.0553559
time: 0.28 seconds

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2_B2_B1_A2

### Relational analysis result of NS_A2_A2_A2_B1_B1_A1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543606, upper bound: 0.0550311
time: 0.26 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0257471, 0.0290832, -0.0138432, 0.0135909, -0.0393380, 0.0429264
1: -0.0298784, 0.0626745, -0.0128219, 0.0240455, -0.0539238, 0.0754964
2: -0.0645730, 0.0442843, -0.0376750, 0.0155482, -0.0801212, 0.0819593
3: -0.0435096, 0.0804226, -0.0249458, 0.0289503, -0.0724599, 0.1053684
4: -0.0875694, 0.0483158, -0.0467689, 0.0194372, -0.1070067, 0.0950847

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A2_A2_A2_B1_B1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0543606, upper bound: 0.0554296
time: 0.26 seconds

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A2_A2_A2_B1_B1_A1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0543606, upper bound: 0.0550311
time: 0.28 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0193978, 0.0229309, -0.0183495, 0.0186593, -0.0380572, 0.0412803
1: -0.0194044, 0.0443665, -0.0180988, 0.0325174, -0.0519218, 0.0624653
2: -0.0521740, 0.0359274, -0.0456146, 0.0231816, -0.0753556, 0.0815420
3: -0.0312126, 0.0560299, -0.0310402, 0.0397046, -0.0709172, 0.0870701
4: -0.0734783, 0.0381431, -0.0545507, 0.0277470, -0.1012253, 0.0926938

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552627, upper bound: 0.0551256
time: 0.26 seconds

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552627, upper bound: 0.0551256
time: 0.26 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0193978, 0.0229309, -0.0144519, 0.0144736, -0.0338714, 0.0373827
1: -0.0194044, 0.0443665, -0.0134160, 0.0251981, -0.0446025, 0.0577825
2: -0.0521740, 0.0359274, -0.0389422, 0.0167805, -0.0689545, 0.0748696
3: -0.0312126, 0.0560299, -0.0256370, 0.0306983, -0.0619109, 0.0816669
4: -0.0734783, 0.0381431, -0.0483767, 0.0208874, -0.0943657, 0.0865199

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B1

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0533715, upper bound: 0.0542765
time: 0.24 seconds

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B2

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0553252, upper bound: 0.0551638
time: 0.25 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B2_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0257471, 0.0290832, -0.0230536, 0.0262401, -0.0519872, 0.0521368
1: -0.0298784, 0.0626745, -0.0242822, 0.0563962, -0.0862746, 0.0869567
2: -0.0645730, 0.0442843, -0.0619228, 0.0438159, -0.1083889, 0.1062071
3: -0.0435096, 0.0804226, -0.0380059, 0.0721582, -0.1156678, 0.1184285
4: -0.0875694, 0.0483158, -0.0842291, 0.0486820, -0.1362514, 0.1325449

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B2_B2_A1_A1

### Relational analysis result of NS_A2_A2_A2_B1_B2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548499, upper bound: 0.0553077
time: 0.25 seconds

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B2_B2_A1_A2

### Relational analysis result of NS_A2_A2_A2_B1_B2_B2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0548499, upper bound: 0.0549398
time: 0.27 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B1_A2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0196476, 0.0240776, -0.0193350, 0.0216215, -0.0412691, 0.0434126
1: -0.0200759, 0.0487652, -0.0221682, 0.0413110, -0.0613869, 0.0709334
2: -0.0530140, 0.0376667, -0.0484735, 0.0279975, -0.0810115, 0.0861402
3: -0.0314373, 0.0610481, -0.0360067, 0.0530757, -0.0845130, 0.0970548
4: -0.0758036, 0.0392452, -0.0654610, 0.0313325, -0.1071362, 0.1047062

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B2_B1_A2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B2_B1_A2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_A2_B2_B1_A2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0183920, 0.0218985, -0.0193350, 0.0216215, -0.0400135, 0.0412334
1: -0.0178266, 0.0416697, -0.0221682, 0.0413110, -0.0591376, 0.0638379
2: -0.0502763, 0.0342374, -0.0484735, 0.0279975, -0.0782738, 0.0827109
3: -0.0290314, 0.0522087, -0.0360067, 0.0530757, -0.0821071, 0.0882154
4: -0.0709423, 0.0362082, -0.0654610, 0.0313325, -0.1022748, 0.1016692

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B2_B1_A2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B2_B1_A2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0245429, 0.0277138, -0.0268412, 0.0308748, -0.0554177, 0.0545550
1: -0.0281288, 0.0585976, -0.0322746, 0.0686580, -0.0967869, 0.0908721
2: -0.0611955, 0.0420559, -0.0661133, 0.0468169, -0.1080124, 0.1081692
3: -0.0411564, 0.0743495, -0.0454444, 0.0886965, -0.1298529, 0.1197939
4: -0.0831138, 0.0458153, -0.0912204, 0.0503456, -0.1334594, 0.1370357

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552664, upper bound: 0.0549489
time: 0.25 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552664, upper bound: 0.0549529
time: 0.28 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0193136, 0.0229116, -0.0268412, 0.0308748, -0.0501884, 0.0497528
1: -0.0194592, 0.0445171, -0.0322746, 0.0686580, -0.0881173, 0.0767917
2: -0.0520530, 0.0358496, -0.0661133, 0.0468169, -0.0988699, 0.1019629
3: -0.0311160, 0.0561660, -0.0454444, 0.0886965, -0.1198125, 0.1016103
4: -0.0733854, 0.0380535, -0.0912204, 0.0503456, -0.1237310, 0.1292740

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552664, upper bound: 0.0549489
time: 0.25 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0552664, upper bound: 0.0549529
time: 0.25 seconds

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0234625, 0.0267646, -0.0204420, 0.0239954, -0.0474579, 0.0472065
1: -0.0264327, 0.0560786, -0.0214377, 0.0480555, -0.0744882, 0.0775163
2: -0.0590539, 0.0406102, -0.0541373, 0.0375680, -0.0966218, 0.0947475
3: -0.0392054, 0.0707839, -0.0337864, 0.0613243, -0.1005297, 0.1045703
4: -0.0812218, 0.0441357, -0.0761260, 0.0400765, -0.1212983, 0.1202617

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0547640, upper bound: 0.0547954
time: 0.26 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0548376, upper bound: 0.0552459
time: 0.27 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 1.64 seconds
NS_A1_B2_A1_B2_B1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0551833, upper bound: 0.0555098
NS_A1_B2_A1_B2_B1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0551833, upper bound: 0.0555098
NS_A1_B2_A1_B2_B1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0548114, upper bound: 0.0555563
NS_A1_B2_A1_B2_B1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0548114, upper bound: 0.0555563
NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0550562, upper bound: 0.0532095
NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0555463, upper bound: 0.0551549
NS_A1_B2_A1_B2_B1_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0547762, upper bound: 0.0546554
NS_A1_B2_A1_B2_B1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0547762, upper bound: 0.0551795
NS_A1_B2_A1_B2_B2_A2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0526917, upper bound: 0.0550045
NS_A1_B2_A1_B2_B2_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0526917, upper bound: 0.0553539
NS_A1_B2_A1_B2_B2_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0526917, upper bound: 0.0550045
NS_A1_B2_A1_B2_B2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0526917, upper bound: 0.0555709
NS_A1_B2_A1_B2_B2_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0531936, upper bound: 0.0549196
NS_A1_B2_A1_B2_B2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0531936, upper bound: 0.0551658
NS_A1_B2_A1_B2_B2_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0531936, upper bound: 0.0549738
NS_A1_B2_A1_B2_B2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0531936, upper bound: 0.0552184
NS_A1_B2_A1_B2_B2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0527726, upper bound: 0.0553990
NS_A1_B2_A1_B2_B2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0527726, upper bound: 0.0555436
NS_A1_B2_A1_B2_B2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0532586, upper bound: 0.0552615
NS_A1_B2_A1_B2_B2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0532586, upper bound: 0.0552623
NS_A1_B2_A2_B1_B1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0551844, upper bound: 0.0551750
NS_A1_B2_A2_B1_B1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0551844, upper bound: 0.0552904
NS_A1_B2_A2_B1_B1_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0533650, upper bound: 0.0542646
NS_A1_B2_A2_B1_B1_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0533650, upper bound: 0.0552804
NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0530645, upper bound: 0.0550998
NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0530645, upper bound: 0.0555821
NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0531590, upper bound: 0.0550959
NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0531590, upper bound: 0.0554742
NS_A1_B2_A2_B2_B1_A2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0548753, upper bound: 0.0549390
NS_A1_B2_A2_B2_B1_A2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0548753, upper bound: 0.0550528
NS_A1_B2_A2_B2_B1_A2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0548753, upper bound: 0.0549390
NS_A1_B2_A2_B2_B1_A2_A2_A2_B2_B2, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0548753, upper bound: 0.0550528
NS_A1_B2_A2_B2_B2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0549505, upper bound: 0.0555230
NS_A1_B2_A2_B2_B2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0549505, upper bound: 0.0555230
NS_A2_A2_A2_B1_B1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0543606, upper bound: 0.0553559
NS_A2_A2_A2_B1_B1_A1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0543606, upper bound: 0.0550311
NS_A2_A2_A2_B1_B1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0543606, upper bound: 0.0554296
NS_A2_A2_A2_B1_B1_A1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0543606, upper bound: 0.0550311
NS_A2_A2_A2_B1_B1_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0552627, upper bound: 0.0551256
NS_A2_A2_A2_B1_B1_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0552627, upper bound: 0.0551256
NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B1, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0533715, upper bound: 0.0542765
NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0553252, upper bound: 0.0551638
NS_A2_A2_A2_B1_B2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0548499, upper bound: 0.0553077
NS_A2_A2_A2_B1_B2_B2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0548499, upper bound: 0.0549398
NS_A2_A2_A2_B2_B2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0552664, upper bound: 0.0549489
NS_A2_A2_A2_B2_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0552664, upper bound: 0.0549529
NS_A2_A2_A2_B2_B2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0552664, upper bound: 0.0549489
NS_A2_A2_A2_B2_B2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0552664, upper bound: 0.0549529
NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0547640, upper bound: 0.0547954
NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 1.64
Output dim: 0, lower bound: -0.0548376, upper bound: 0.0552459

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0185107, 0.0187518, -0.0168130, 0.0163271, -0.0348378, 0.0355648
1: -0.0178545, 0.0326230, -0.0151748, 0.0267792, -0.0446337, 0.0477978
2: -0.0458835, 0.0238966, -0.0420975, 0.0194794, -0.0653629, 0.0659941
3: -0.0305152, 0.0399753, -0.0273803, 0.0324003, -0.0629155, 0.0673557
4: -0.0555754, 0.0282560, -0.0502301, 0.0239952, -0.0795706, 0.0784860

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0183495, 0.0186593, -0.0168130, 0.0163271, -0.0346766, 0.0354724
1: -0.0180988, 0.0325174, -0.0151748, 0.0267792, -0.0448780, 0.0476923
2: -0.0456146, 0.0231816, -0.0420975, 0.0194794, -0.0650941, 0.0652791
3: -0.0310402, 0.0397046, -0.0273803, 0.0324003, -0.0634405, 0.0670849
4: -0.0545507, 0.0277470, -0.0502301, 0.0239952, -0.0785459, 0.0779770

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0185107, 0.0187518, -0.0135560, 0.0118012, -0.0303119, 0.0323078
1: -0.0178545, 0.0326230, -0.0124801, 0.0228618, -0.0407163, 0.0451032
2: -0.0458835, 0.0238966, -0.0364058, 0.0146392, -0.0605226, 0.0603024
3: -0.0305152, 0.0399753, -0.0246231, 0.0269644, -0.0574796, 0.0645984
4: -0.0555754, 0.0282560, -0.0449692, 0.0182736, -0.0738490, 0.0732252

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0183495, 0.0186593, -0.0135560, 0.0118012, -0.0301507, 0.0322154
1: -0.0180988, 0.0325174, -0.0124801, 0.0228618, -0.0409606, 0.0449976
2: -0.0456146, 0.0231816, -0.0364058, 0.0146392, -0.0602538, 0.0595874
3: -0.0310402, 0.0397046, -0.0246231, 0.0269644, -0.0580046, 0.0643277
4: -0.0545507, 0.0277470, -0.0449692, 0.0182736, -0.0728243, 0.0727162

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0138432, 0.0135909, -0.0172572, 0.0167018, -0.0305450, 0.0308481
1: -0.0128219, 0.0240455, -0.0157410, 0.0275783, -0.0404002, 0.0397865
2: -0.0376750, 0.0155482, -0.0431397, 0.0202850, -0.0579600, 0.0586879
3: -0.0249458, 0.0289503, -0.0279668, 0.0336190, -0.0585648, 0.0569171
4: -0.0467689, 0.0194372, -0.0513968, 0.0250097, -0.0717786, 0.0708341

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555463, upper bound: 0.0547776
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0555463, upper bound: 0.0549557
time: 0.27 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0144519, 0.0144736, -0.0135560, 0.0118012, -0.0262531, 0.0280296
1: -0.0134160, 0.0251981, -0.0124801, 0.0228618, -0.0362779, 0.0376783
2: -0.0389422, 0.0167805, -0.0364058, 0.0146392, -0.0535813, 0.0531863
3: -0.0256370, 0.0306983, -0.0246231, 0.0269644, -0.0526014, 0.0553213
4: -0.0483767, 0.0208874, -0.0449692, 0.0182736, -0.0666503, 0.0658566

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0167907, 0.0173994, -0.0385190, 0.0398805, -0.0566712, 0.0559184
1: -0.0166599, 0.0318810, -0.0555091, 0.0989415, -0.1156014, 0.0873901
2: -0.0451706, 0.0220867, -0.0896124, 0.0614502, -0.1066208, 0.1116991
3: -0.0294806, 0.0401354, -0.0771710, 0.1382393, -0.1677198, 0.1173064
4: -0.0579042, 0.0269196, -0.1234518, 0.0698946, -0.1277988, 0.1503714

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0167907, 0.0173994, -0.0372107, 0.0385487, -0.0553393, 0.0546100
1: -0.0166599, 0.0318810, -0.0522647, 0.0942615, -0.1109214, 0.0841457
2: -0.0451706, 0.0220867, -0.0878225, 0.0596141, -0.1047847, 0.1099092
3: -0.0294806, 0.0401354, -0.0728077, 0.1317676, -0.1612482, 0.1129431
4: -0.0579042, 0.0269196, -0.1206840, 0.0682416, -0.1261458, 0.1476036

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0385190, 0.0398805, -0.0561807, 0.0556223
1: -0.0163531, 0.0312095, -0.0555091, 0.0989415, -0.1152946, 0.0867186
2: -0.0443946, 0.0211404, -0.0896124, 0.0614502, -0.1058448, 0.1107529
3: -0.0292436, 0.0388854, -0.0771710, 0.1382393, -0.1674828, 0.1160563
4: -0.0566717, 0.0260474, -0.1234518, 0.0698946, -0.1265664, 0.1494992

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0372107, 0.0385487, -0.0548488, 0.0543140
1: -0.0163531, 0.0312095, -0.0522647, 0.0942615, -0.1106147, 0.0834743
2: -0.0443946, 0.0211404, -0.0878225, 0.0596141, -0.1040087, 0.1089629
3: -0.0292436, 0.0388854, -0.0728077, 0.1317676, -0.1610112, 0.1116931
4: -0.0566717, 0.0260474, -0.1206840, 0.0682416, -0.1249133, 0.1467314

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0167907, 0.0173994, -0.0219643, 0.0252398, -0.0420304, 0.0393637
1: -0.0166599, 0.0318810, -0.0233759, 0.0556172, -0.0722771, 0.0552569
2: -0.0451706, 0.0220867, -0.0582474, 0.0394656, -0.0846362, 0.0803341
3: -0.0294806, 0.0401354, -0.0343274, 0.0709491, -0.1004296, 0.0744629
4: -0.0579042, 0.0269196, -0.0823332, 0.0426576, -0.1005618, 0.1092528

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0167907, 0.0173994, -0.0212124, 0.0241675, -0.0409581, 0.0386118
1: -0.0166599, 0.0318810, -0.0224719, 0.0516808, -0.0683407, 0.0543529
2: -0.0451706, 0.0220867, -0.0570551, 0.0379869, -0.0831575, 0.0791418
3: -0.0294806, 0.0401354, -0.0336688, 0.0660210, -0.0955016, 0.0738043
4: -0.0579042, 0.0269196, -0.0802179, 0.0414562, -0.0993603, 0.1071375

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0219643, 0.0252398, -0.0415399, 0.0390676
1: -0.0163531, 0.0312095, -0.0233759, 0.0556172, -0.0719704, 0.0545854
2: -0.0443946, 0.0211404, -0.0582474, 0.0394656, -0.0838602, 0.0793878
3: -0.0292436, 0.0388854, -0.0343274, 0.0709491, -0.1001926, 0.0732128
4: -0.0566717, 0.0260474, -0.0823332, 0.0426576, -0.0993294, 0.1083806

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0163001, 0.0171033, -0.0212124, 0.0241675, -0.0404676, 0.0383157
1: -0.0163531, 0.0312095, -0.0224719, 0.0516808, -0.0680339, 0.0536814
2: -0.0443946, 0.0211404, -0.0570551, 0.0379869, -0.0823815, 0.0781955
3: -0.0292436, 0.0388854, -0.0336688, 0.0660210, -0.0952646, 0.0725542
4: -0.0566717, 0.0260474, -0.0802179, 0.0414562, -0.0981279, 0.1062653

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_B1_A2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0236056, 0.0276349, -0.0178932, 0.0180688, -0.0416744, 0.0455281
1: -0.0246522, 0.0609175, -0.0170138, 0.0303990, -0.0550512, 0.0779314
2: -0.0631518, 0.0459309, -0.0446541, 0.0223315, -0.0854833, 0.0905850
3: -0.0380466, 0.0772544, -0.0295371, 0.0369350, -0.0749816, 0.1067915
4: -0.0868571, 0.0502552, -0.0533013, 0.0268344, -0.1136915, 0.1035565

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A2_B1_B1_A2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0230536, 0.0262401, -0.0178932, 0.0180688, -0.0411224, 0.0441334
1: -0.0242822, 0.0563962, -0.0170138, 0.0303990, -0.0546812, 0.0734101
2: -0.0619228, 0.0438159, -0.0446541, 0.0223315, -0.0842544, 0.0884699
3: -0.0380059, 0.0721582, -0.0295371, 0.0369350, -0.0749409, 0.1016954
4: -0.0842291, 0.0486820, -0.0533013, 0.0268344, -0.1110635, 0.1019833

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B2_A2_B1_B1_A2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0234873, 0.0266846, -0.0135135, 0.0120200, -0.0355074, 0.0401981
1: -0.0250003, 0.0577051, -0.0125482, 0.0231994, -0.0481997, 0.0702533
2: -0.0628079, 0.0445847, -0.0367681, 0.0148250, -0.0776329, 0.0813528
3: -0.0388779, 0.0740361, -0.0247198, 0.0275398, -0.0664177, 0.0987559
4: -0.0853493, 0.0495821, -0.0455008, 0.0186069, -0.1039562, 0.0950829

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B1_A2_A2_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0272616, 0.0315264, -0.0340429, 0.0416231, -0.0688847, 0.0655694
1: -0.0318267, 0.0731649, -0.0463589, 0.1000430, -0.1318697, 0.1195238
2: -0.0690642, 0.0515455, -0.0833662, 0.0583165, -0.1273807, 0.1349117
3: -0.0456918, 0.0943532, -0.0660171, 0.1464675, -0.1921593, 0.1603703
4: -0.0947821, 0.0565235, -0.1322550, 0.0653650, -0.1601470, 0.1887785

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0528588, upper bound: 0.0553265
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0528588, upper bound: 0.0555821
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0266347, 0.0300013, -0.0340429, 0.0416231, -0.0682578, 0.0640442
1: -0.0309553, 0.0678936, -0.0463589, 0.1000430, -0.1309983, 0.1142526
2: -0.0675894, 0.0492085, -0.0833662, 0.0583165, -0.1259059, 0.1325747
3: -0.0454101, 0.0878100, -0.0660171, 0.1464675, -0.1918776, 0.1538272
4: -0.0916399, 0.0547648, -0.1322550, 0.0653650, -0.1570049, 0.1870199

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0528618, upper bound: 0.0550689
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0528618, upper bound: 0.0550903
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B2_A2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0272616, 0.0315264, -0.0190541, 0.0224912, -0.0497528, 0.0505805
1: -0.0318267, 0.0731649, -0.0184067, 0.0432951, -0.0751218, 0.0915716
2: -0.0690642, 0.0515455, -0.0513755, 0.0353198, -0.1043840, 0.1029210
3: -0.0456918, 0.0943532, -0.0299964, 0.0545656, -0.1002574, 0.1243496
4: -0.0947821, 0.0565235, -0.0722335, 0.0375377, -0.1323197, 0.1287570

Time for backsubstitution: 1.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

## BFS NS instance: NS_A1_B2_A2_B2_B2_A2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0266347, 0.0300013, -0.0190541, 0.0224912, -0.0491258, 0.0490553
1: -0.0309553, 0.0678936, -0.0184067, 0.0432951, -0.0742504, 0.0863003
2: -0.0675894, 0.0492085, -0.0513755, 0.0353198, -0.1029092, 0.1005840
3: -0.0454101, 0.0878100, -0.0299964, 0.0545656, -0.0999757, 0.1178065
4: -0.0916399, 0.0547648, -0.0722335, 0.0375377, -0.1291776, 0.1269983

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_B2_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

## BFS NS instance: NS_A2_A2_A2_B1_B1_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0263041, 0.0304260, -0.0170747, 0.0169581, -0.0432622, 0.0475008
1: -0.0312516, 0.0671478, -0.0162292, 0.0278331, -0.0590847, 0.0833770
2: -0.0651119, 0.0461771, -0.0424308, 0.0201539, -0.0852658, 0.0886079
3: -0.0443822, 0.0865055, -0.0286224, 0.0335460, -0.0779282, 0.1151278
4: -0.0899085, 0.0495911, -0.0510356, 0.0244890, -0.1143975, 0.1006266

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_A2_B1_B1_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0263041, 0.0304260, -0.0138432, 0.0135909, -0.0398951, 0.0442692
1: -0.0312516, 0.0671478, -0.0128219, 0.0240455, -0.0552971, 0.0799697
2: -0.0651119, 0.0461771, -0.0376750, 0.0155482, -0.0806601, 0.0838521
3: -0.0443822, 0.0865055, -0.0249458, 0.0289503, -0.0733325, 0.1114513
4: -0.0899085, 0.0495911, -0.0467689, 0.0194372, -0.1093457, 0.0963600

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A2_B1_B1_A1_B2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_A2_B1_B1_A2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202285, 0.0246614, -0.0183495, 0.0186593, -0.0388878, 0.0430109
1: -0.0209462, 0.0502541, -0.0180988, 0.0325174, -0.0534637, 0.0683529
2: -0.0540638, 0.0386361, -0.0456146, 0.0231816, -0.0772454, 0.0842507
3: -0.0327265, 0.0632129, -0.0310402, 0.0397046, -0.0724311, 0.0942531
4: -0.0772433, 0.0403433, -0.0545507, 0.0277470, -0.1049903, 0.0948940

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_A2_B1_B1_A2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0190303, 0.0225245, -0.0183495, 0.0186593, -0.0376896, 0.0408739
1: -0.0188390, 0.0434136, -0.0180988, 0.0325174, -0.0513565, 0.0615124
2: -0.0514293, 0.0353010, -0.0456146, 0.0231816, -0.0746109, 0.0809156
3: -0.0304982, 0.0547148, -0.0310402, 0.0397046, -0.0702028, 0.0857550
4: -0.0725133, 0.0374084, -0.0545507, 0.0277470, -0.1002603, 0.0919591

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0193978, 0.0229309, -0.0138432, 0.0135909, -0.0329888, 0.0367740
1: -0.0194044, 0.0443665, -0.0128219, 0.0240455, -0.0434498, 0.0571884
2: -0.0521740, 0.0359274, -0.0376750, 0.0155482, -0.0677222, 0.0736023
3: -0.0312126, 0.0560299, -0.0249458, 0.0289503, -0.0601629, 0.0809757
4: -0.0734783, 0.0381431, -0.0467689, 0.0194372, -0.0929156, 0.0849120

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B2_A1

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0533463
time: 0.27 seconds

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B2_A2

### Relational analysis result of NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0551638
time: 0.28 seconds

## BFS NS instance: NS_A2_A2_A2_B1_B2_B2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0263041, 0.0304260, -0.0230536, 0.0262401, -0.0525443, 0.0534796
1: -0.0312516, 0.0671478, -0.0242822, 0.0563962, -0.0876478, 0.0914300
2: -0.0651119, 0.0461771, -0.0619228, 0.0438159, -0.1089278, 0.1080999
3: -0.0443822, 0.0865055, -0.0380059, 0.0721582, -0.1165404, 0.1245114
4: -0.0899085, 0.0495911, -0.0842291, 0.0486820, -0.1385905, 0.1338201

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A2_B1_B2_B2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0251018, 0.0291385, -0.0268412, 0.0308748, -0.0559766, 0.0559797
1: -0.0295147, 0.0635338, -0.0322746, 0.0686580, -0.0981727, 0.0958084
2: -0.0619103, 0.0441243, -0.0661133, 0.0468169, -0.1087272, 0.1102377
3: -0.0421057, 0.0811660, -0.0454444, 0.0886965, -0.1308022, 0.1266103
4: -0.0861053, 0.0472269, -0.0912204, 0.0503456, -0.1364509, 0.1384474

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0240271, 0.0271807, -0.0268412, 0.0308748, -0.0549019, 0.0540219
1: -0.0274073, 0.0573252, -0.0322746, 0.0686580, -0.0960653, 0.0895998
2: -0.0601453, 0.0412739, -0.0661133, 0.0468169, -0.1069622, 0.1073872
3: -0.0403004, 0.0725133, -0.0454444, 0.0886965, -0.1289969, 0.1179577
4: -0.0819726, 0.0448981, -0.0912204, 0.0503456, -0.1323182, 0.1361186

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0200487, 0.0244717, -0.0268412, 0.0308748, -0.0509235, 0.0513128
1: -0.0208523, 0.0497518, -0.0322746, 0.0686580, -0.0895103, 0.0820264
2: -0.0536549, 0.0382584, -0.0661133, 0.0468169, -0.1004718, 0.1043717
3: -0.0325000, 0.0624496, -0.0454444, 0.0886965, -0.1211966, 0.1078939
4: -0.0766241, 0.0399505, -0.0912204, 0.0503456, -0.1269697, 0.1311709

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0189508, 0.0225059, -0.0268412, 0.0308748, -0.0498256, 0.0493471
1: -0.0189328, 0.0435142, -0.0322746, 0.0686580, -0.0875908, 0.0757888
2: -0.0513224, 0.0352242, -0.0661133, 0.0468169, -0.0981393, 0.1013375
3: -0.0304608, 0.0547802, -0.0454444, 0.0886965, -0.1191573, 0.1002246
4: -0.0724334, 0.0373175, -0.0912204, 0.0503456, -0.1227791, 0.1285379

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0234625, 0.0267646, -0.0200355, 0.0235253, -0.0469877, 0.0468001
1: -0.0264327, 0.0560786, -0.0208371, 0.0468975, -0.0733301, 0.0769157
2: -0.0590539, 0.0406102, -0.0533296, 0.0368664, -0.0959203, 0.0939398
3: -0.0392054, 0.0707839, -0.0330414, 0.0597220, -0.0989273, 0.1038253
4: -0.0812218, 0.0441357, -0.0750898, 0.0392354, -0.1204572, 0.1192256

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0527828, upper bound: 0.0549294
time: 0.27 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B2

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0527828, upper bound: 0.0552786
time: 0.26 seconds

## Summary of splitting at layer (split count: 10)
- Time for NS candidates: 1.88 seconds
NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 1.88
Output dim: 0, lower bound: -0.0555463, upper bound: 0.0547776
NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 1.88
Output dim: 0, lower bound: -0.0555463, upper bound: 0.0549557
NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 11, time: 1.88
Output dim: 0, lower bound: -0.0528588, upper bound: 0.0553265
NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 11, time: 1.88
Output dim: 0, lower bound: -0.0528588, upper bound: 0.0555821
NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_B2_B1, status: Status.VERIFIED, split count: 11, time: 1.88
Output dim: 0, lower bound: -0.0528618, upper bound: 0.0550689
NS_A1_B2_A2_B2_B1_A2_A1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 11, time: 1.88
Output dim: 0, lower bound: -0.0528618, upper bound: 0.0550903
NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B2_A1, status: Status.VERIFIED, split count: 11, time: 1.88
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0533463
NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 1.88
Output dim: 0, lower bound: -0.0549814, upper bound: 0.0551638
NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 11, time: 1.88
Output dim: 0, lower bound: -0.0527828, upper bound: 0.0549294
NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 11, time: 1.88
Output dim: 0, lower bound: -0.0527828, upper bound: 0.0552786

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0138432, 0.0135909, -0.0169668, 0.0165350, -0.0303782, 0.0305578
1: -0.0128219, 0.0240455, -0.0152356, 0.0273660, -0.0401880, 0.0392811
2: -0.0376750, 0.0155482, -0.0423630, 0.0201021, -0.0577770, 0.0579112
3: -0.0249458, 0.0289503, -0.0272670, 0.0333995, -0.0583453, 0.0562173
4: -0.0467689, 0.0194372, -0.0513256, 0.0243630, -0.0711319, 0.0707628

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## BFS NS instance: NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0138432, 0.0135909, -0.0168130, 0.0163271, -0.0301703, 0.0304040
1: -0.0128219, 0.0240455, -0.0151748, 0.0267792, -0.0396011, 0.0392203
2: -0.0376750, 0.0155482, -0.0420975, 0.0194794, -0.0571544, 0.0576457
3: -0.0249458, 0.0289503, -0.0273803, 0.0324003, -0.0573461, 0.0563306
4: -0.0467689, 0.0194372, -0.0502301, 0.0239952, -0.0707641, 0.0696673

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_B1_B1_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0272616, 0.0315264, -0.0348893, 0.0433529, -0.0706145, 0.0664158
1: -0.0318267, 0.0731649, -0.0487626, 0.1052918, -0.1371185, 0.1219275
2: -0.0690642, 0.0515455, -0.0846913, 0.0603575, -0.1294216, 0.1362368
3: -0.0456918, 0.0943532, -0.0691351, 0.1537860, -0.1994778, 0.1634882
4: -0.0947821, 0.0565235, -0.1357993, 0.0669278, -0.1617099, 0.1923228

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B2_A2_B2_B1_A2_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0272616, 0.0315264, -0.0336147, 0.0411007, -0.0683623, 0.0651411
1: -0.0318267, 0.0731649, -0.0456810, 0.0984664, -0.1302931, 0.1188459
2: -0.0690642, 0.0515455, -0.0824923, 0.0575975, -0.1266617, 0.1340378
3: -0.0456918, 0.0943532, -0.0651933, 0.1438725, -0.1895642, 0.1595464
4: -0.0947821, 0.0565235, -0.1306875, 0.0645201, -0.1593021, 0.1872110

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0187581, 0.0223078, -0.0138432, 0.0135909, -0.0323490, 0.0361509
1: -0.0183857, 0.0426489, -0.0128219, 0.0240455, -0.0424312, 0.0554708
2: -0.0510134, 0.0348739, -0.0376750, 0.0155482, -0.0665616, 0.0725489
3: -0.0297386, 0.0535609, -0.0249458, 0.0289503, -0.0586888, 0.0785067
4: -0.0719108, 0.0369572, -0.0467689, 0.0194372, -0.0913481, 0.0837261

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A2_B1_B1_A2_A2_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0234625, 0.0267646, -0.0193763, 0.0228807, -0.0463432, 0.0461409
1: -0.0264327, 0.0560786, -0.0198272, 0.0449618, -0.0713944, 0.0759058
2: -0.0590539, 0.0406102, -0.0521190, 0.0357868, -0.0948407, 0.0927292
3: -0.0392054, 0.0707839, -0.0315510, 0.0568579, -0.0960632, 0.1023349
4: -0.0812218, 0.0441357, -0.0734975, 0.0379837, -0.1192056, 0.1176332

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0526000, upper bound: 0.0552786
time: 0.29 seconds

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0526000, upper bound: 0.0548700
time: 0.35 seconds

## Summary of splitting at layer (split count: 11)
- Time for NS candidates: 1.63 seconds
NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 12, time: 1.63
Output dim: 0, lower bound: -0.0526000, upper bound: 0.0552786
NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B2_A2, status: Status.VERIFIED, split count: 12, time: 1.63
Output dim: 0, lower bound: -0.0526000, upper bound: 0.0548700

## BFS NS instance: NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0240400, 0.0283851, -0.0193763, 0.0228807, -0.0469207, 0.0477614
1: -0.0278738, 0.0615251, -0.0198272, 0.0449618, -0.0728355, 0.0813522
2: -0.0600704, 0.0431052, -0.0521190, 0.0357868, -0.0958572, 0.0952242
3: -0.0402572, 0.0781609, -0.0315510, 0.0568579, -0.0971151, 0.1097119
4: -0.0847134, 0.0459236, -0.0734975, 0.0379837, -0.1226971, 0.1194210

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A2_B2_B2_A2_B2_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.56 + 290.88 = 292.44 seconds
