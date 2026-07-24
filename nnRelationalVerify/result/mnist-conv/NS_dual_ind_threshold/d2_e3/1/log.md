## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.190247616


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.4045985, 0.6086638, -0.4045985, 0.6086638, -0.5856323, 0.5856323)
1: (-12.2102737, -10.9345093, -12.2102737, -10.9345093, -0.6986628, 0.6986628)
2: (-7.8243608, -6.8160505, -7.8243608, -6.8160505, -0.5276794, 0.5276794)
3: (-11.8109741, -10.7529621, -11.8109741, -10.7529621, -0.4937754, 0.4937754)
4: (-2.6709776, -1.8229246, -2.6709776, -1.8229246, -0.4948776, 0.4948779)
5: (-5.3238344, -4.3532982, -5.3238344, -4.3532982, -0.5168405, 0.5168405)
6: (7.1399260, 7.8838181, 7.1399260, 7.8838181, -0.3902711, 0.3902711)
7: (-17.4398041, -16.0133190, -17.4398041, -16.0133190, -0.6233628, 0.6233625)
8: (-3.1521769, -2.2361634, -3.1521769, -2.2361634, -0.4710871, 0.4710871)
9: (-10.1329279, -9.1146822, -10.1329279, -9.1146822, -0.7942631, 0.7942631)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.76 + 32.95 = 54.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.1981739, upper bound: 0.1981734

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 523

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4610

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1981682, upper bound: 0.1963942
time: 3.32 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1981682, upper bound: 0.1981677
time: 3.22 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.74 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.74
Output dim: 6, lower bound: -0.1981682, upper bound: 0.1963942
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.74
Output dim: 6, lower bound: -0.1981682, upper bound: 0.1981677

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.3936980, 0.5961318, -0.4040084, 0.6023209, -0.5688896, 0.5727551
1: -12.2001572, -10.9415398, -12.2057533, -10.9349060, -0.6887665, 0.6878495
2: -7.8179288, -6.8236680, -7.8236437, -6.8192368, -0.5188181, 0.5194486
3: -11.8104477, -10.7548294, -11.8108292, -10.7534695, -0.4922867, 0.4915842
4: -2.6659634, -1.8271134, -2.6686800, -1.8233273, -0.4899607, 0.4891466
5: -5.3197145, -4.3562107, -5.3220205, -4.3534675, -0.5127929, 0.5115037
6: 7.1447687, 7.8761578, 7.1400537, 7.8801775, -0.3809288, 0.3830775
7: -17.4192410, -16.0278130, -17.4301796, -16.0136719, -0.6025562, 0.5996270
8: -3.1449018, -2.2403383, -3.1491513, -2.2362275, -0.4638791, 0.4619093
9: -10.1286163, -9.1189995, -10.1309185, -9.1151724, -0.7905307, 0.7876036

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 523

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4610

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1963942
time: 3.14 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1963932
time: 4.58 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.4045966, 0.6086476, -0.4045978, 0.6086555, -0.5814106, 0.5723287
1: -12.2102528, -10.9345093, -12.2102633, -10.9345093, -0.6903214, 0.6986594
2: -7.8243589, -6.8160591, -7.8243594, -6.8160524, -0.5276737, 0.5225965
3: -11.8109732, -10.7529640, -11.8109722, -10.7529631, -0.4924424, 0.4947616
4: -2.6709673, -1.8229249, -2.6709733, -1.8229239, -0.4909983, 0.4948765
5: -5.3238297, -4.3532996, -5.3238311, -4.3533001, -0.5142181, 0.5161316
6: 7.1399255, 7.8838058, 7.1399260, 7.8838120, -0.3895485, 0.3827214
7: -17.4397736, -16.0133209, -17.4397888, -16.0133209, -0.5998590, 0.6155481
8: -3.1521695, -2.2361636, -3.1521735, -2.2361636, -0.4657285, 0.4699068
9: -10.1329203, -9.1146841, -10.1329231, -9.1146832, -0.7951989, 0.7914257

Time for backsubstitution: 21.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 523

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4610

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1981691
time: 3.20 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1981691
time: 2.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.55 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.55
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1963942
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.55
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1963932
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.55
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1981691
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.55
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1981691

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.3936980, 0.5961318, -0.3936980, 0.5961318, -0.5626390, 0.5626390
1: -12.2001572, -10.9415398, -12.2001572, -10.9415398, -0.6823578, 0.6823578
2: -7.8179288, -6.8236680, -7.8179288, -6.8236680, -0.5141501, 0.5141500
3: -11.8104477, -10.7548294, -11.8104477, -10.7548294, -0.4902463, 0.4902462
4: -2.6659634, -1.8271134, -2.6659634, -1.8271134, -0.4864531, 0.4864528
5: -5.3197145, -4.3562107, -5.3197145, -4.3562107, -0.5094032, 0.5094033
6: 7.1447687, 7.8761578, 7.1447687, 7.8761578, -0.3776814, 0.3776814
7: -17.4192410, -16.0278130, -17.4192410, -16.0278130, -0.5885077, 0.5885077
8: -3.1449018, -2.2403383, -3.1449018, -2.2403383, -0.4586245, 0.4586245
9: -10.1286163, -9.1189995, -10.1286163, -9.1189995, -0.7843385, 0.7843385

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 523

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962110, upper bound: 0.1963921
time: 3.25 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963915, upper bound: 0.1963919
time: 3.12 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.3936980, 0.5961318, -0.4045966, 0.6086476, -0.5713755, 0.5687438
1: -12.2001572, -10.9415398, -12.2102528, -10.9345093, -0.6889729, 0.6920414
2: -7.8179288, -6.8236680, -7.8243589, -6.8160591, -0.5220945, 0.5197231
3: -11.8104477, -10.7548294, -11.8109732, -10.7529640, -0.4913359, 0.4908012
4: -2.6659634, -1.8271134, -2.6709673, -1.8229249, -0.4900271, 0.4913007
5: -5.3197145, -4.3562107, -5.3238297, -4.3532996, -0.5121250, 0.5134046
6: 7.1447687, 7.8761578, 7.1399255, 7.8838058, -0.3848138, 0.3824129
7: -17.4192410, -16.0278130, -17.4397736, -16.0133209, -0.5948825, 0.6020336
8: -3.1449018, -2.2403383, -3.1521695, -2.2361636, -0.4627408, 0.4657837
9: -10.1286163, -9.1189995, -10.1329203, -9.1146841, -0.7883997, 0.7873580

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 523

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962110, upper bound: 0.1963921
time: 3.42 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963916, upper bound: 0.1963918
time: 3.16 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.4045966, 0.6086476, -0.3936980, 0.5961318, -0.5687438, 0.5713756
1: -12.2102528, -10.9345093, -12.2001572, -10.9415398, -0.6920414, 0.6889729
2: -7.8243589, -6.8160591, -7.8179288, -6.8236680, -0.5197229, 0.5220944
3: -11.8109732, -10.7529640, -11.8104477, -10.7548294, -0.4908011, 0.4913359
4: -2.6709673, -1.8229249, -2.6659634, -1.8271134, -0.4913006, 0.4900271
5: -5.3238297, -4.3532996, -5.3197145, -4.3562107, -0.5134046, 0.5121251
6: 7.1399255, 7.8838058, 7.1447687, 7.8761578, -0.3824129, 0.3848138
7: -17.4397736, -16.0133209, -17.4192410, -16.0278130, -0.6020339, 0.5948826
8: -3.1521695, -2.2361636, -3.1449018, -2.2403383, -0.4657837, 0.4627409
9: -10.1329203, -9.1146841, -10.1286163, -9.1189995, -0.7873583, 0.7883997

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 523

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962104, upper bound: 0.1981664
time: 3.38 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963909, upper bound: 0.1981662
time: 3.61 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.4045966, 0.6086476, -0.4045966, 0.6086476, -0.5723279, 0.5723280
1: -12.2102528, -10.9345093, -12.2102528, -10.9345093, -0.6903205, 0.6903205
2: -7.8243589, -6.8160591, -7.8243589, -6.8160591, -0.5225959, 0.5225960
3: -11.8109732, -10.7529640, -11.8109732, -10.7529640, -0.4947600, 0.4947600
4: -2.6709673, -1.8229249, -2.6709673, -1.8229249, -0.4909984, 0.4909983
5: -5.3238297, -4.3532996, -5.3238297, -4.3532996, -0.5142171, 0.5142171
6: 7.1399255, 7.8838058, 7.1399255, 7.8838058, -0.3827206, 0.3827205
7: -17.4397736, -16.0133209, -17.4397736, -16.0133209, -0.5998588, 0.5998588
8: -3.1521695, -2.2361636, -3.1521695, -2.2361636, -0.4657276, 0.4657276
9: -10.1329203, -9.1146841, -10.1329203, -9.1146841, -0.7951972, 0.7951975

Time for backsubstitution: 22.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 523

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962103, upper bound: 0.1981657
time: 4.69 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963908, upper bound: 0.1981655
time: 9.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 36.96 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 36.96
Output dim: 6, lower bound: -0.1962110, upper bound: 0.1963921
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 36.96
Output dim: 6, lower bound: -0.1963915, upper bound: 0.1963919
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 36.96
Output dim: 6, lower bound: -0.1962110, upper bound: 0.1963921
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 36.96
Output dim: 6, lower bound: -0.1963916, upper bound: 0.1963918
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 36.96
Output dim: 6, lower bound: -0.1962104, upper bound: 0.1981664
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 36.96
Output dim: 6, lower bound: -0.1963909, upper bound: 0.1981662
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 36.96
Output dim: 6, lower bound: -0.1962103, upper bound: 0.1981657
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 36.96
Output dim: 6, lower bound: -0.1963908, upper bound: 0.1981655

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.3931885, 0.5960908, -0.3936980, 0.5961318, -0.5621102, 0.5626204
1: -12.2000532, -10.9424925, -12.2001572, -10.9415398, -0.6822414, 0.6813183
2: -7.8167114, -6.8238273, -7.8179288, -6.8236680, -0.5129385, 0.5139575
3: -11.8102951, -10.7551594, -11.8104477, -10.7548294, -0.4900718, 0.4899111
4: -2.6659248, -1.8283474, -2.6659634, -1.8271134, -0.4863887, 0.4852171
5: -5.3196163, -4.3567343, -5.3197145, -4.3562107, -0.5091789, 0.5087699
6: 7.1450696, 7.8761263, 7.1447687, 7.8761578, -0.3774374, 0.3776386
7: -17.4185143, -16.0278797, -17.4192410, -16.0278130, -0.5876284, 0.5882812
8: -3.1437778, -2.2403398, -3.1449018, -2.2403383, -0.4574360, 0.4586195
9: -10.1281900, -9.1191063, -10.1286163, -9.1189995, -0.7838593, 0.7842309

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 523

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962111, upper bound: 0.1962120
time: 3.39 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962111, upper bound: 0.1963926
time: 3.26 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.3960247, 0.6012266, -0.3936949, 0.5961328, -0.5643666, 0.5680528
1: -12.2154675, -10.9406729, -12.2001553, -10.9415436, -0.6923845, 0.6838446
2: -7.8193240, -6.8034239, -7.8179259, -6.8236685, -0.5165770, 0.5299101
3: -11.8153563, -10.7538519, -11.8104477, -10.7548304, -0.4938977, 0.4907601
4: -2.6863663, -1.8262081, -2.6659646, -1.8271179, -0.4950833, 0.4886136
5: -5.3281283, -4.3556046, -5.3197145, -4.3562117, -0.5166671, 0.5126497
6: 7.1441922, 7.8799791, 7.1447687, 7.8761559, -0.3782527, 0.3810787
7: -17.4219761, -16.0181541, -17.4192371, -16.0278130, -0.5950463, 0.5948112
8: -3.1470041, -2.2239394, -3.1449008, -2.2403386, -0.4624175, 0.4657011
9: -10.1305923, -9.1105957, -10.1286144, -9.1189995, -0.7849913, 0.7925174

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 523

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963918, upper bound: 0.1962120
time: 3.32 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963918, upper bound: 0.1963926
time: 3.30 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.3931885, 0.5960908, -0.4045966, 0.6086476, -0.5708462, 0.5687218
1: -12.2000532, -10.9424925, -12.2102528, -10.9345093, -0.6888566, 0.6910019
2: -7.8167114, -6.8238273, -7.8243589, -6.8160591, -0.5208831, 0.5195304
3: -11.8102951, -10.7551594, -11.8109732, -10.7529640, -0.4911616, 0.4904660
4: -2.6659248, -1.8283474, -2.6709673, -1.8229249, -0.4899632, 0.4900650
5: -5.3196163, -4.3567343, -5.3238297, -4.3532996, -0.5119007, 0.5127711
6: 7.1450696, 7.8761263, 7.1399255, 7.8838058, -0.3845698, 0.3823701
7: -17.4185143, -16.0278797, -17.4397736, -16.0133209, -0.5940015, 0.6018138
8: -3.1437778, -2.2403398, -3.1521695, -2.2361636, -0.4615523, 0.4657787
9: -10.1281900, -9.1191063, -10.1329203, -9.1146841, -0.7879205, 0.7872505

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 523

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1979870, upper bound: 0.1962102
time: 3.38 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1979870, upper bound: 0.1963909
time: 3.28 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.3960247, 0.6012266, -0.4045954, 0.6086462, -0.5727859, 0.5695229
1: -12.2154675, -10.9406729, -12.2102547, -10.9345121, -0.6932070, 0.6935282
2: -7.8193240, -6.8034239, -7.8243561, -6.8160586, -0.5245211, 0.5308456
3: -11.8153563, -10.7538519, -11.8109732, -10.7529659, -0.4945130, 0.4913151
4: -2.6863663, -1.8262081, -2.6709673, -1.8229284, -0.4955641, 0.4934610
5: -5.3281283, -4.3556046, -5.3238306, -4.3533010, -0.5170183, 0.5166510
6: 7.1441922, 7.8799791, 7.1399269, 7.8838053, -0.3853853, 0.3858054
7: -17.4219761, -16.0181541, -17.4397697, -16.0133209, -0.5977863, 0.6026263
8: -3.1470041, -2.2239394, -3.1521673, -2.2361631, -0.4665335, 0.4691309
9: -10.1305923, -9.1105957, -10.1329174, -9.1146851, -0.7890520, 0.7955365

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 523

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1981654, upper bound: 0.1962102
time: 3.33 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1981654, upper bound: 0.1963909
time: 3.39 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.4040885, 0.6086047, -0.3936980, 0.5961318, -0.5682150, 0.5713537
1: -12.2101498, -10.9354620, -12.2001572, -10.9415398, -0.6919227, 0.6879339
2: -7.8231435, -6.8162155, -7.8179288, -6.8236680, -0.5185118, 0.5218966
3: -11.8108206, -10.7532930, -11.8104477, -10.7548294, -0.4906271, 0.4910009
4: -2.6709270, -1.8241575, -2.6659634, -1.8271134, -0.4912367, 0.4887906
5: -5.3237295, -4.3538237, -5.3197145, -4.3562107, -0.5131812, 0.5114912
6: 7.1402273, 7.8837738, 7.1447687, 7.8761578, -0.3821683, 0.3847715
7: -17.4390488, -16.0133896, -17.4192410, -16.0278130, -0.6011522, 0.5946623
8: -3.1510458, -2.2361665, -3.1449018, -2.2403383, -0.4645964, 0.4627359
9: -10.1324959, -9.1147947, -10.1286163, -9.1189995, -0.7868791, 0.7882888

Time for backsubstitution: 22.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 523

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962104, upper bound: 0.1979878
time: 3.57 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962104, upper bound: 0.1981662
time: 3.34 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.4068828, 0.6137390, -0.3936949, 0.5961328, -0.5701755, 0.5721519
1: -12.2255898, -10.9336681, -12.2001553, -10.9415436, -0.6965468, 0.6904483
2: -7.8256989, -6.7958360, -7.8179259, -6.8236685, -0.5221324, 0.5338553
3: -11.8158779, -10.7520275, -11.8104477, -10.7548304, -0.4943275, 0.4918516
4: -2.6913769, -1.8220379, -2.6659646, -1.8271179, -0.4971569, 0.4921861
5: -5.3322821, -4.3527064, -5.3197145, -4.3562117, -0.5186944, 0.5153831
6: 7.1393614, 7.8876343, 7.1447687, 7.8761559, -0.3829871, 0.3882415
7: -17.4424706, -16.0036602, -17.4192371, -16.0278130, -0.6049237, 0.5954900
8: -3.1542659, -2.2197685, -3.1449008, -2.2403386, -0.4695741, 0.4657691
9: -10.1349688, -9.1062279, -10.1286144, -9.1189995, -0.7882171, 0.7966459

Time for backsubstitution: 22.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 523

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963910, upper bound: 0.1979878
time: 3.32 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963910, upper bound: 0.1981662
time: 3.27 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.4040885, 0.6086047, -0.4045966, 0.6086476, -0.5717995, 0.5723095
1: -12.2101498, -10.9354620, -12.2102528, -10.9345093, -0.6902018, 0.6892815
2: -7.8231435, -6.8162155, -7.8243589, -6.8160591, -0.5213847, 0.5223988
3: -11.8108206, -10.7532930, -11.8109732, -10.7529640, -0.4945860, 0.4944251
4: -2.6709270, -1.8241575, -2.6709673, -1.8229249, -0.4909345, 0.4897622
5: -5.3237295, -4.3538237, -5.3238297, -4.3532996, -0.5139933, 0.5135839
6: 7.1402273, 7.8837738, 7.1399255, 7.8838058, -0.3824764, 0.3826779
7: -17.4390488, -16.0133896, -17.4397736, -16.0133209, -0.5989795, 0.5996320
8: -3.1510458, -2.2361665, -3.1521695, -2.2361636, -0.4645400, 0.4657226
9: -10.1324959, -9.1147947, -10.1329203, -9.1146841, -0.7947171, 0.7950859

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 523

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962112, upper bound: 0.1979883
time: 3.29 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962112, upper bound: 0.1981666
time: 3.40 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.4068828, 0.6137390, -0.4045954, 0.6086462, -0.5740435, 0.5755622
1: -12.2255898, -10.9336681, -12.2102547, -10.9345121, -0.6988041, 0.6917968
2: -7.8256989, -6.7958360, -7.8243561, -6.8160586, -0.5250058, 0.5363108
3: -11.8158779, -10.7520275, -11.8109732, -10.7529659, -0.4979498, 0.4952753
4: -2.6913769, -1.8220379, -2.6709673, -1.8229284, -0.4984785, 0.4931569
5: -5.3322821, -4.3527064, -5.3238306, -4.3533010, -0.5204426, 0.5175028
6: 7.1393614, 7.8876343, 7.1399269, 7.8838053, -0.3833082, 0.3861418
7: -17.4424706, -16.0036602, -17.4397697, -16.0133209, -0.6063888, 0.6063108
8: -3.1542659, -2.2197685, -3.1521673, -2.2361631, -0.4695226, 0.4728951
9: -10.1349688, -9.1062279, -10.1329174, -9.1146851, -0.7959220, 0.8034389

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 523

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963913, upper bound: 0.1979883
time: 3.21 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963913, upper bound: 0.1981666
time: 3.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.58 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1962111, upper bound: 0.1962120
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1962111, upper bound: 0.1963926
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1963918, upper bound: 0.1962120
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1963918, upper bound: 0.1963926
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1979870, upper bound: 0.1962102
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1979870, upper bound: 0.1963909
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1981654, upper bound: 0.1962102
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1981654, upper bound: 0.1963909
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1962104, upper bound: 0.1979878
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1962104, upper bound: 0.1981662
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1963910, upper bound: 0.1979878
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1963910, upper bound: 0.1981662
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1962112, upper bound: 0.1979883
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1962112, upper bound: 0.1981666
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1963913, upper bound: 0.1979883
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.58
Output dim: 6, lower bound: -0.1963913, upper bound: 0.1981666

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3931885, 0.5960908, -0.3931885, 0.5960908, -0.5620916, 0.5620916
1: -12.2000532, -10.9424925, -12.2000532, -10.9424925, -0.6812019, 0.6812019
2: -7.8167114, -6.8238273, -7.8167114, -6.8238273, -0.5127461, 0.5127460
3: -11.8102951, -10.7551594, -11.8102951, -10.7551594, -0.4897368, 0.4897368
4: -2.6659248, -1.8283474, -2.6659248, -1.8283474, -0.4851531, 0.4851532
5: -5.3196163, -4.3567343, -5.3196163, -4.3567343, -0.5085455, 0.5085458
6: 7.1450696, 7.8761263, 7.1450696, 7.8761263, -0.3773946, 0.3773944
7: -17.4185143, -16.0278797, -17.4185143, -16.0278797, -0.5874019, 0.5874019
8: -3.1437778, -2.2403398, -3.1437778, -2.2403398, -0.4574310, 0.4574310
9: -10.1281900, -9.1191063, -10.1281900, -9.1191063, -0.7837517, 0.7837515

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1256
type: A, layer: 3, pos: 2124
type: A, layer: 3, pos: 2515
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1193
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1844
type: A, layer: 3, pos: 2613
type: A, layer: 3, pos: 1405
type: A, layer: 3, pos: 2826
type: A, layer: 3, pos: 2865
type: A, layer: 3, pos: 549
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 618
type: A, layer: 3, pos: 1914
type: A, layer: 3, pos: 1240

Time for candidate selection: 0.40 seconds

### Candidate
type: A, layer: 3, pos: 1256

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1868395, upper bound: 0.1933335
time: 3.48 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1933320, upper bound: 0.1933335
time: 3.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3931885, 0.5960908, -0.3960247, 0.6012266, -0.5675259, 0.5640261
1: -12.2000532, -10.9424925, -12.2154675, -10.9406729, -0.6826863, 0.6913559
2: -7.8167114, -6.8238273, -7.8193240, -6.8034239, -0.5286984, 0.5145285
3: -11.8102951, -10.7551594, -11.8153563, -10.7538519, -0.4902813, 0.4935628
4: -2.6659248, -1.8283474, -2.6863663, -1.8262081, -0.4866350, 0.4938443
5: -5.3196163, -4.3567343, -5.3281283, -4.3556046, -0.5095134, 0.5160340
6: 7.1450696, 7.8761263, 7.1441922, 7.8799791, -0.3808353, 0.3778572
7: -17.4185143, -16.0278797, -17.4219761, -16.0181541, -0.5939302, 0.5908742
8: -3.1437778, -2.2403398, -3.1470041, -2.2239394, -0.4645122, 0.4606766
9: -10.1281900, -9.1191063, -10.1305923, -9.1105957, -0.7920394, 0.7846892

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1256
type: A, layer: 3, pos: 2124
type: A, layer: 3, pos: 2515
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1193
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1844
type: A, layer: 3, pos: 2613
type: A, layer: 3, pos: 1405
type: A, layer: 3, pos: 2826
type: A, layer: 3, pos: 2865
type: A, layer: 3, pos: 549
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 618
type: A, layer: 3, pos: 1914
type: A, layer: 3, pos: 1240

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 1256

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1868395, upper bound: 0.1935740
time: 3.43 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1933320, upper bound: 0.1935740
time: 3.58 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.3960247, 0.6012266, -0.3931885, 0.5960908, -0.5640261, 0.5675259
1: -12.2154675, -10.9406729, -12.2000532, -10.9424925, -0.6913559, 0.6826863
2: -7.8193240, -6.8034239, -7.8167114, -6.8238273, -0.5145285, 0.5286986
3: -11.8153563, -10.7538519, -11.8102951, -10.7551594, -0.4935629, 0.4902813
4: -2.6863663, -1.8262081, -2.6659248, -1.8283474, -0.4938444, 0.4866349
5: -5.3281283, -4.3556046, -5.3196163, -4.3567343, -0.5160340, 0.5095137
6: 7.1441922, 7.8799791, 7.1450696, 7.8761263, -0.3778572, 0.3808353
7: -17.4219761, -16.0181541, -17.4185143, -16.0278797, -0.5908742, 0.5939302
8: -3.1470041, -2.2239394, -3.1437778, -2.2403398, -0.4606766, 0.4645123
9: -10.1305923, -9.1105957, -10.1281900, -9.1191063, -0.7846892, 0.7920394

Time for backsubstitution: 22.29 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1256
type: A, layer: 3, pos: 2124
type: A, layer: 3, pos: 2515
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1193
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1844
type: A, layer: 3, pos: 2823
type: A, layer: 3, pos: 2613
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1405
type: A, layer: 3, pos: 2826
type: A, layer: 3, pos: 2865
type: A, layer: 3, pos: 549
type: A, layer: 3, pos: 618
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 1914
type: A, layer: 3, pos: 1240

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 1256

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1868696, upper bound: 0.1933331
time: 3.49 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1935725, upper bound: 0.1933331
time: 3.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.3960247, 0.6012266, -0.3960247, 0.6012266, -0.5669672, 0.5669672
1: -12.2154675, -10.9406729, -12.2154675, -10.9406729, -0.6896496, 0.6896496
2: -7.8193240, -6.8034239, -7.8193240, -6.8034239, -0.5204277, 0.5204277
3: -11.8153563, -10.7538519, -11.8153563, -10.7538519, -0.4929252, 0.4929253
4: -2.6863663, -1.8262081, -2.6863663, -1.8262081, -0.4918840, 0.4918842
5: -5.3281283, -4.3556046, -5.3281283, -4.3556046, -0.5131761, 0.5131762
6: 7.1441922, 7.8799791, 7.1441922, 7.8799791, -0.3796453, 0.3796453
7: -17.4219761, -16.0181541, -17.4219761, -16.0181541, -0.5957088, 0.5957088
8: -3.1470041, -2.2239394, -3.1470041, -2.2239394, -0.4636427, 0.4636428
9: -10.1305923, -9.1105957, -10.1305923, -9.1105957, -0.7914894, 0.7914894

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1256
type: A, layer: 3, pos: 2124
type: A, layer: 3, pos: 2515
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1193
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1844
type: A, layer: 3, pos: 2823
type: A, layer: 3, pos: 2613
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 1405
type: A, layer: 3, pos: 2826
type: A, layer: 3, pos: 2865
type: A, layer: 3, pos: 549
type: A, layer: 3, pos: 618
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 1914
type: A, layer: 3, pos: 1240

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 1256

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1868696, upper bound: 0.1933331
time: 3.42 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1935727, upper bound: 0.1933331
time: 3.39 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3931885, 0.5960908, -0.4040885, 0.6086047, -0.5708244, 0.5681931
1: -12.2000532, -10.9424925, -12.2101498, -10.9354620, -0.6878171, 0.6908836
2: -7.8167114, -6.8238273, -7.8231435, -6.8162155, -0.5206852, 0.5183192
3: -11.8102951, -10.7551594, -11.8108206, -10.7532930, -0.4908266, 0.4902921
4: -2.6659248, -1.8283474, -2.6709270, -1.8241575, -0.4887266, 0.4900011
5: -5.3196163, -4.3567343, -5.3237295, -4.3538237, -0.5112669, 0.5125477
6: 7.1450696, 7.8761263, 7.1402273, 7.8837738, -0.3845273, 0.3821254
7: -17.4185143, -16.0278797, -17.4390488, -16.0133896, -0.5937812, 0.6009322
8: -3.1437778, -2.2403398, -3.1510458, -2.2361665, -0.4615471, 0.4645914
9: -10.1281900, -9.1191063, -10.1324959, -9.1147947, -0.7878094, 0.7867715

Time for backsubstitution: 22.28 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 54.71 + 559.52 = 614.23 seconds
