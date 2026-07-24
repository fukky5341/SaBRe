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
execution time: IAR + RelationalAnalysis = 23.08 + 32.93 = 56.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.1981739, upper bound: 0.1981734

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4610

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1981682, upper bound: 0.1963942
time: 3.28 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1981682, upper bound: 0.1981677
time: 3.19 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.70 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.70
Output dim: 6, lower bound: -0.1981682, upper bound: 0.1963942
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.70
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

Time for backsubstitution: 21.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4610

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1963942
time: 3.17 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1963932
time: 4.46 seconds

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

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4610

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1981691
time: 3.18 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1981691
time: 2.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.41 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.41
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1963942
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.41
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1963932
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.41
Output dim: 6, lower bound: -0.1963933, upper bound: 0.1981691
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.41
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

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962110, upper bound: 0.1963921
time: 3.15 seconds

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

Time for backsubstitution: 22.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962110, upper bound: 0.1963921
time: 3.41 seconds

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

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 523

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963912, upper bound: 0.1979878
time: 3.18 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963910, upper bound: 0.1981661
time: 3.22 seconds

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

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 523

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1962103, upper bound: 0.1981657
time: 4.40 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1963908, upper bound: 0.1981655
time: 9.13 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 35.88 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 35.88
Output dim: 6, lower bound: -0.1962110, upper bound: 0.1963921
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 35.88
Output dim: 6, lower bound: -0.1963915, upper bound: 0.1963919
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 35.88
Output dim: 6, lower bound: -0.1962110, upper bound: 0.1963921
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 35.88
Output dim: 6, lower bound: -0.1963916, upper bound: 0.1963918
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 35.88
Output dim: 6, lower bound: -0.1963912, upper bound: 0.1979878
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 35.88
Output dim: 6, lower bound: -0.1963910, upper bound: 0.1981661
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 35.88
Output dim: 6, lower bound: -0.1962103, upper bound: 0.1981657
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 35.88
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

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 2124
type: A, layer: 3, pos: 2124
type: B, layer: 3, pos: 2515
type: A, layer: 3, pos: 2515
type: B, layer: 3, pos: 206
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 1193
type: B, layer: 3, pos: 1193
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1844
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 1405
type: B, layer: 3, pos: 1405
type: A, layer: 3, pos: 2826
type: B, layer: 3, pos: 2826
type: B, layer: 3, pos: 2865
type: A, layer: 3, pos: 2865
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 549
type: B, layer: 3, pos: 549
type: B, layer: 3, pos: 618
type: A, layer: 3, pos: 618
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 1914
type: B, layer: 3, pos: 1914
type: A, layer: 3, pos: 1240
type: B, layer: 3, pos: 1240

Time for candidate selection: 0.41 seconds

### Candidate
type: B, layer: 3, pos: 1256

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1933321, upper bound: 0.1868707
time: 3.34 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1933321, upper bound: 0.1935739
time: 3.42 seconds

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

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 1256
type: B, layer: 3, pos: 2124
type: A, layer: 3, pos: 2124
type: B, layer: 3, pos: 2515
type: A, layer: 3, pos: 2515
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 206
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 1193
type: A, layer: 3, pos: 1193
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1844
type: A, layer: 3, pos: 2823
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 1405
type: B, layer: 3, pos: 1405
type: A, layer: 3, pos: 2826
type: B, layer: 3, pos: 2826
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 2865
type: A, layer: 3, pos: 2865
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 549
type: B, layer: 3, pos: 549
type: A, layer: 3, pos: 618
type: B, layer: 3, pos: 618
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 1914
type: B, layer: 3, pos: 1914
type: A, layer: 3, pos: 1240
type: B, layer: 3, pos: 1240

Time for candidate selection: 0.45 seconds

### Candidate
type: A, layer: 3, pos: 1256

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1868696, upper bound: 0.1935738
time: 3.52 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1935725, upper bound: 0.1935738
time: 3.43 seconds

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

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 1256
type: A, layer: 3, pos: 2124
type: B, layer: 3, pos: 2124
type: B, layer: 3, pos: 2515
type: A, layer: 3, pos: 2515
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1193
type: B, layer: 3, pos: 1193
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1844
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 1405
type: B, layer: 3, pos: 1405
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 2826
type: B, layer: 3, pos: 2826
type: B, layer: 3, pos: 2865
type: A, layer: 3, pos: 2865
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 549
type: A, layer: 3, pos: 549
type: B, layer: 3, pos: 618
type: A, layer: 3, pos: 618
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 1914
type: A, layer: 3, pos: 1914
type: B, layer: 3, pos: 1240
type: A, layer: 3, pos: 1240

Time for candidate selection: 0.42 seconds

### Candidate
type: B, layer: 3, pos: 1256

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1951017, upper bound: 0.1868699
time: 3.33 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1951017, upper bound: 0.1935731
time: 3.33 seconds

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

Time for backsubstitution: 22.20 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 2124
type: B, layer: 3, pos: 2124
type: B, layer: 3, pos: 2515
type: A, layer: 3, pos: 2515
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 1193
type: B, layer: 3, pos: 1193
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1844
type: A, layer: 3, pos: 2823
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 1405
type: B, layer: 3, pos: 1405
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 2826
type: B, layer: 3, pos: 2826
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 2865
type: A, layer: 3, pos: 2865
type: B, layer: 3, pos: 549
type: A, layer: 3, pos: 549
type: B, layer: 3, pos: 618
type: A, layer: 3, pos: 618
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 1914
type: A, layer: 3, pos: 1914
type: B, layer: 3, pos: 1240
type: A, layer: 3, pos: 1240

Time for candidate selection: 0.47 seconds

### Candidate
type: A, layer: 3, pos: 1256

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1868695, upper bound: 0.1935719
time: 5.33 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1953421, upper bound: 0.1935729
time: 3.31 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.4045966, 0.6086476, -0.3931885, 0.5960908, -0.5687218, 0.5708464
1: -12.2102528, -10.9345093, -12.2000532, -10.9424925, -0.6910019, 0.6888566
2: -7.8243589, -6.8160591, -7.8167114, -6.8238273, -0.5195303, 0.5208830
3: -11.8109732, -10.7529640, -11.8102951, -10.7551594, -0.4904661, 0.4911615
4: -2.6709673, -1.8229249, -2.6659248, -1.8283474, -0.4900650, 0.4899631
5: -5.3238297, -4.3532996, -5.3196163, -4.3567343, -0.5127712, 0.5119008
6: 7.1399255, 7.8838058, 7.1450696, 7.8761263, -0.3823701, 0.3845698
7: -17.4397736, -16.0133209, -17.4185143, -16.0278797, -0.6018137, 0.5940015
8: -3.1521695, -2.2361636, -3.1437778, -2.2403398, -0.4657787, 0.4615523
9: -10.1329203, -9.1146841, -10.1281900, -9.1191063, -0.7872508, 0.7879205

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 1256
type: B, layer: 3, pos: 2124
type: A, layer: 3, pos: 2124
type: A, layer: 3, pos: 2515
type: B, layer: 3, pos: 2515
type: B, layer: 3, pos: 206
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 1193
type: A, layer: 3, pos: 1193
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 1844
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 1405
type: A, layer: 3, pos: 1405
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 2826
type: A, layer: 3, pos: 2826
type: A, layer: 3, pos: 2865
type: B, layer: 3, pos: 2865
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 549
type: B, layer: 3, pos: 549
type: A, layer: 3, pos: 618
type: B, layer: 3, pos: 618
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 1914
type: B, layer: 3, pos: 1914
type: A, layer: 3, pos: 1240
type: B, layer: 3, pos: 1240

Time for candidate selection: 0.43 seconds

### Candidate
type: A, layer: 3, pos: 1256

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1868689, upper bound: 0.1951027
time: 3.35 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1935721, upper bound: 0.1951027
time: 3.52 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.4045954, 0.6086462, -0.3960247, 0.6012266, -0.5695229, 0.5727859
1: -12.2102547, -10.9345121, -12.2154675, -10.9406729, -0.6935282, 0.6932070
2: -7.8243561, -6.8160586, -7.8193240, -6.8034239, -0.5308456, 0.5245212
3: -11.8109732, -10.7529659, -11.8153563, -10.7538519, -0.4913151, 0.4945132
4: -2.6709673, -1.8229284, -2.6863663, -1.8262081, -0.4934609, 0.4955641
5: -5.3238306, -4.3533010, -5.3281283, -4.3556046, -0.5166508, 0.5170184
6: 7.1399269, 7.8838053, 7.1441922, 7.8799791, -0.3858054, 0.3853853
7: -17.4397697, -16.0133209, -17.4219761, -16.0181541, -0.6026263, 0.5977864
8: -3.1521673, -2.2361631, -3.1470041, -2.2239394, -0.4691309, 0.4665334
9: -10.1329174, -9.1146851, -10.1305923, -9.1105957, -0.7955365, 0.7890522

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 2124
type: A, layer: 3, pos: 2124
type: A, layer: 3, pos: 2515
type: B, layer: 3, pos: 2515
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 206
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 1193
type: A, layer: 3, pos: 1193
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 1844
type: B, layer: 3, pos: 2823
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 1405
type: A, layer: 3, pos: 1405
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 2826
type: A, layer: 3, pos: 2826
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 2865
type: B, layer: 3, pos: 2865
type: A, layer: 3, pos: 549
type: B, layer: 3, pos: 549
type: A, layer: 3, pos: 618
type: B, layer: 3, pos: 618
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 1914
type: B, layer: 3, pos: 1914
type: A, layer: 3, pos: 1240
type: B, layer: 3, pos: 1240

Time for candidate selection: 0.45 seconds

### Candidate
type: B, layer: 3, pos: 1256

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1935719, upper bound: 0.1886572
time: 3.38 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1935719, upper bound: 0.1953431
time: 3.47 seconds

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

Time for backsubstitution: 22.23 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 2124
type: A, layer: 3, pos: 2124
type: B, layer: 3, pos: 2515
type: A, layer: 3, pos: 2515
type: B, layer: 3, pos: 206
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 1193
type: B, layer: 3, pos: 1193
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1844
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 1405
type: B, layer: 3, pos: 1405
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 2826
type: B, layer: 3, pos: 2826
type: B, layer: 3, pos: 2865
type: A, layer: 3, pos: 2865
type: A, layer: 3, pos: 549
type: B, layer: 3, pos: 549
type: B, layer: 3, pos: 618
type: A, layer: 3, pos: 618
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 1914
type: B, layer: 3, pos: 1914
type: A, layer: 3, pos: 1240
type: B, layer: 3, pos: 1240

Time for candidate selection: 0.50 seconds

### Candidate
type: B, layer: 3, pos: 1256

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1933312, upper bound: 0.1886667
time: 3.24 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1933312, upper bound: 0.1953439
time: 3.27 seconds

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

Time for backsubstitution: 22.23 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1256
type: B, layer: 3, pos: 1256
type: B, layer: 3, pos: 2124
type: A, layer: 3, pos: 2124
type: B, layer: 3, pos: 2515
type: A, layer: 3, pos: 2515
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 1193
type: B, layer: 3, pos: 1193
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1844
type: A, layer: 3, pos: 2823
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 1405
type: B, layer: 3, pos: 1405
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 2826
type: B, layer: 3, pos: 2826
type: B, layer: 3, pos: 2865
type: A, layer: 3, pos: 2865
type: B, layer: 3, pos: 549
type: A, layer: 3, pos: 549
type: B, layer: 3, pos: 618
type: A, layer: 3, pos: 618
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 1914
type: B, layer: 3, pos: 1914
type: B, layer: 3, pos: 1240
type: A, layer: 3, pos: 1240

Time for candidate selection: 0.45 seconds

### Candidate
type: A, layer: 3, pos: 1256

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1868813, upper bound: 0.1953436
time: 3.38 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1935717, upper bound: 0.1953437
time: 3.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.43 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1933321, upper bound: 0.1868707
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1933321, upper bound: 0.1935739
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1868696, upper bound: 0.1935738
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1935725, upper bound: 0.1935738
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1951017, upper bound: 0.1868699
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1951017, upper bound: 0.1935731
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1868695, upper bound: 0.1935719
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1953421, upper bound: 0.1935729
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1868689, upper bound: 0.1951027
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1935721, upper bound: 0.1951027
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1935719, upper bound: 0.1886572
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1935719, upper bound: 0.1953431
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1933312, upper bound: 0.1886667
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1933312, upper bound: 0.1953439
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1868813, upper bound: 0.1953436
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 29.43
Output dim: 6, lower bound: -0.1935717, upper bound: 0.1953437

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3870249, 0.5959752, -0.3804717, 0.5958433, -0.5499957, 0.5387357
1: -12.1999483, -10.9442406, -12.1998844, -10.9459381, -0.6794200, 0.6795034
2: -7.8167109, -6.8261280, -7.8179278, -6.8293548, -0.5070596, 0.5115812
3: -11.8094740, -10.7552204, -11.8083982, -10.7549810, -0.4872880, 0.4843856
4: -2.6646266, -1.8296385, -2.6627221, -1.8303320, -0.4825138, 0.4793142
5: -5.3183994, -4.3567400, -5.3166809, -4.3562241, -0.5068673, 0.5032401
6: 7.1524539, 7.8760333, 7.1628175, 7.8759260, -0.3673282, 0.3543916
7: -17.4185028, -16.0333233, -17.4192142, -16.0414467, -0.5761554, 0.5823870
8: -3.1433361, -2.2403398, -3.1437974, -2.2403383, -0.4570094, 0.4575695
9: -10.1281080, -9.1241055, -10.1284065, -9.1313324, -0.7683349, 0.7777183

Time for backsubstitution: 22.25 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2124
type: B, layer: 3, pos: 2124
type: A, layer: 3, pos: 2515
type: B, layer: 3, pos: 2515
type: A, layer: 3, pos: 1256
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 1193
type: B, layer: 3, pos: 1193
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 1405
type: A, layer: 3, pos: 1405
type: B, layer: 3, pos: 2826
type: A, layer: 3, pos: 2826
type: B, layer: 3, pos: 2865
type: A, layer: 3, pos: 2865
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 549
type: B, layer: 3, pos: 549
type: B, layer: 3, pos: 618
type: A, layer: 3, pos: 618
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 1914
type: B, layer: 3, pos: 1914
type: A, layer: 3, pos: 1240
type: B, layer: 3, pos: 1240

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 2124

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1863402, upper bound: 0.1847487
time: 3.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1913033, upper bound: 0.1848191
time: 3.36 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3834352, 0.5960546, -0.3709607, 0.5941205, -0.5929801, 0.5353953
1: -12.1997385, -10.9434071, -12.2018347, -10.9437513, -0.6802783, 0.6786904
2: -7.8167109, -6.8258095, -7.8179770, -6.8281527, -0.5079403, 0.5147728
3: -11.8077507, -10.7553482, -11.8045979, -10.7586269, -0.4882033, 0.4837736
4: -2.6654849, -1.8289008, -2.6661644, -1.8284588, -0.4847872, 0.4811944
5: -5.3180737, -4.3567481, -5.3164306, -4.3572168, -0.5079355, 0.5051419
6: 7.1483402, 7.8759246, 7.1516647, 7.8878431, -0.4011436, 0.3586249
7: -17.4185028, -16.0332985, -17.4241142, -16.0410919, -0.5762157, 0.5940742
8: -3.1431971, -2.2403398, -3.1434441, -2.2403100, -0.4564359, 0.4570153
9: -10.1279373, -9.1215715, -10.1359396, -9.1247139, -0.7705445, 0.7949407

Time for backsubstitution: 22.83 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2124
type: B, layer: 3, pos: 2124
type: A, layer: 3, pos: 2515
type: B, layer: 3, pos: 2515
type: A, layer: 3, pos: 1256
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 1850
type: A, layer: 3, pos: 1850
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 1193
type: A, layer: 3, pos: 891
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 1193
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 2613
type: B, layer: 3, pos: 1405
type: A, layer: 3, pos: 1405
type: B, layer: 3, pos: 2826
type: A, layer: 3, pos: 2826
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2865
type: A, layer: 3, pos: 2865
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 549
type: B, layer: 3, pos: 549
type: B, layer: 3, pos: 618
type: A, layer: 3, pos: 618
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 1914
type: B, layer: 3, pos: 1914
type: A, layer: 3, pos: 1240
type: B, layer: 3, pos: 1240

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 2124

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1863402, upper bound: 0.1914929
time: 3.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1913033, upper bound: 0.1915451
time: 3.93 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.3822038, 0.6009336, -0.3875480, 0.5960183, -0.5402350, 0.5559795
1: -12.2152052, -10.9451427, -12.2000504, -10.9432850, -0.6906478, 0.6808963
2: -7.8193235, -6.8089609, -7.8179245, -6.8259535, -0.5142229, 0.5238746
3: -11.8133497, -10.7539988, -11.8096342, -10.7548904, -0.4881268, 0.4880006
4: -2.6831000, -1.8294590, -2.6646748, -1.8284056, -0.4891484, 0.4846672
5: -5.3251314, -4.3556166, -5.3185058, -4.3562169, -0.5110261, 0.5103509
6: 7.1625347, 7.8797541, 7.1521173, 7.8760653, -0.3559164, 0.3710473
7: -17.4219513, -16.0313225, -17.4192276, -16.0332127, -0.5891492, 0.5838469
8: -3.1458759, -2.2239394, -3.1444602, -2.2403386, -0.4613389, 0.4652804
9: -10.1303911, -9.1224871, -10.1285315, -9.1239557, -0.7785206, 0.7775342

Time for backsubstitution: 22.79 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2124
type: A, layer: 3, pos: 2124
type: B, layer: 3, pos: 2515
type: A, layer: 3, pos: 2515
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 206
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 1193
type: A, layer: 3, pos: 1193
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 2823
type: A, layer: 3, pos: 2613
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 1405
type: B, layer: 3, pos: 1405
type: A, layer: 3, pos: 2826
type: B, layer: 3, pos: 2826
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 2865
type: B, layer: 3, pos: 2865
type: B, layer: 3, pos: 1255
type: A, layer: 3, pos: 549
type: B, layer: 3, pos: 549
type: A, layer: 3, pos: 618
type: B, layer: 3, pos: 618
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 1143
type: B, layer: 3, pos: 1914
type: A, layer: 3, pos: 1914
type: B, layer: 3, pos: 1240
type: A, layer: 3, pos: 1240

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 2124

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.1847477, upper bound: 0.1865819
time: 4.02 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1848181, upper bound: 0.1915443
time: 3.65 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.3723919, 0.6019568, -0.3839791, 0.5960953, -0.5370042, 0.5992619
1: -12.2175322, -10.9429560, -12.1998386, -10.9424524, -0.6894755, 0.6818080
2: -7.8193722, -6.8083358, -7.8179255, -6.8256550, -0.5173914, 0.5247133
3: -11.8090515, -10.7576885, -11.8079052, -10.7550182, -0.4875181, 0.4888524
4: -2.6867621, -1.8275955, -2.6655271, -1.8276713, -0.4910622, 0.4869678
5: -5.3245878, -4.3566146, -5.3181729, -4.3562260, -0.5128850, 0.5117192
6: 7.1514263, 7.8935823, 7.1480289, 7.8759542, -0.3603899, 0.4068781
7: -17.4268494, -16.0329170, -17.4192276, -16.0332375, -0.6025612, 0.5836892
8: -3.1454961, -2.2238519, -3.1443200, -2.2403386, -0.4607245, 0.4647061
9: -10.1378384, -9.1173105, -10.1283588, -9.1214752, -0.7956049, 0.7783048

Time for backsubstitution: 22.32 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2124
type: A, layer: 3, pos: 2124
type: B, layer: 3, pos: 2515
type: A, layer: 3, pos: 2515
type: B, layer: 3, pos: 1256
type: A, layer: 3, pos: 1850
type: B, layer: 3, pos: 1850
type: B, layer: 3, pos: 206
type: A, layer: 3, pos: 206
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 1193
type: B, layer: 3, pos: 891
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 1193
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 2823
type: A, layer: 3, pos: 2613
type: A, layer: 3, pos: 1405
type: B, layer: 3, pos: 1405
type: A, layer: 3, pos: 2826
type: A, layer: 3, pos: 1255
type: B, layer: 3, pos: 2826
type: B, layer: 3, pos: 2613
type: A, layer: 3, pos: 2865
type: B, layer: 3, pos: 2865
type: B, layer: 3, pos: 1255
type: B, layer: 3, pos: 549
type: A, layer: 3, pos: 549
type: A, layer: 3, pos: 618
type: B, layer: 3, pos: 618
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 1143
type: B, layer: 3, pos: 1914
type: A, layer: 3, pos: 1914
type: B, layer: 3, pos: 1240
type: A, layer: 3, pos: 1240

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 2124

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1914912, upper bound: 0.1865820
time: 3.55 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.1915435, upper bound: 0.1915434
time: 3.56 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.3870249, 0.5959752, -0.3914568, 0.6083579, -0.5587925, 0.5457861
1: -12.1999483, -10.9442406, -12.2099819, -10.9389057, -0.6860266, 0.6891875
2: -7.8167109, -6.8261280, -7.8243575, -6.8217359, -0.5150166, 0.5171545
3: -11.8094740, -10.7552204, -11.8089247, -10.7531185, -0.4883776, 0.4849519
4: -2.6646266, -1.8296385, -2.6677358, -1.8261316, -0.4860731, 0.4841706
5: -5.3183994, -4.3567400, -5.3207984, -4.3533120, -0.5095890, 0.5072505
6: 7.1524539, 7.8760333, 7.1579752, 7.8835745, -0.3744620, 0.3591448
7: -17.4185028, -16.0333233, -17.4397488, -16.0269585, -0.5831929, 0.5963115
8: -3.1433361, -2.2403398, -3.1510646, -2.2361636, -0.4611260, 0.4647238
9: -10.1281080, -9.1241055, -10.1327105, -9.1270123, -0.7723935, 0.7807391

Time for backsubstitution: 22.34 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.01 + 566.25 = 622.26 seconds
