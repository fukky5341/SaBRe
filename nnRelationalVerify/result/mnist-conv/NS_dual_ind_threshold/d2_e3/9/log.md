## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.22581355849999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.3212552, -6.0277715, -7.3212552, -6.0277715, -0.7738495, 0.7738495)
1: (-11.2155113, -10.1836176, -11.2155113, -10.1836176, -0.6459391, 0.6459394)
2: (-7.8833771, -6.8467493, -7.8833771, -6.8467493, -0.6083186, 0.6083186)
3: (-5.0048704, -4.3139172, -5.0048704, -4.3139172, -0.6005569, 0.6005573)
4: (-7.5120955, -6.6229897, -7.5120955, -6.6229897, -0.8191080, 0.8191080)
5: (5.5277600, 6.2615957, 5.5277600, 6.2615957, -0.5779729, 0.5779729)
6: (-9.4402256, -8.2102938, -9.4402256, -8.2102938, -0.8672638, 0.8672638)
7: (-14.8832645, -13.7124090, -14.8832645, -13.7124090, -0.7276466, 0.7276464)
8: (-3.3201313, -2.2244248, -3.3201313, -2.2244248, -0.6108687, 0.6108685)
9: (-6.4222074, -5.5684242, -6.4222074, -5.5684242, -0.6705360, 0.6705360)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.95 + 33.99 = 56.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.2269478, upper bound: 0.2269479

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4576
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 4576

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269374, upper bound: 0.2266847
time: 3.78 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269369, upper bound: 0.2269370
time: 3.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.90 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.90
Output dim: 5, lower bound: -0.2269374, upper bound: 0.2266847
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.90
Output dim: 5, lower bound: -0.2269369, upper bound: 0.2269370

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -7.3212495, -6.0277967, -7.3212552, -6.0277715, -0.7738490, 0.7738175
1: -11.2155113, -10.1836224, -11.2155113, -10.1836176, -0.6459363, 0.6459348
2: -7.8832331, -6.8467493, -7.8833771, -6.8467493, -0.6081712, 0.6083181
3: -5.0046511, -4.3139176, -5.0048704, -4.3139172, -0.6003265, 0.6005487
4: -7.5120950, -6.6232681, -7.5120955, -6.6229897, -0.8190970, 0.8188188
5: 5.5277605, 6.2613273, 5.5277600, 6.2615957, -0.5779724, 0.5777018
6: -9.4400406, -8.2102928, -9.4402256, -8.2102938, -0.8670702, 0.8672643
7: -14.8832626, -13.7126274, -14.8832645, -13.7124090, -0.7276442, 0.7274222
8: -3.3200922, -2.2244267, -3.3201313, -2.2244248, -0.6108172, 0.6108682
9: -6.4221163, -5.5684290, -6.4222074, -5.5684242, -0.6704464, 0.6705332

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4576
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4576

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266843, upper bound: 0.2266847
time: 4.16 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266843, upper bound: 0.2266847
time: 3.98 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7.3225260, -6.0277462, -7.3212547, -6.0277715, -0.7756262, 0.7741551
1: -11.2155743, -10.1832409, -11.2155132, -10.1836176, -0.6460261, 0.6462762
2: -7.8841066, -6.8416414, -7.8833771, -6.8467493, -0.6098752, 0.6134207
3: -5.0049586, -4.3065090, -5.0048695, -4.3139172, -0.6019440, 0.6078711
4: -7.5215316, -6.6229124, -7.5120964, -6.6229901, -0.8284183, 0.8209896
5: 5.5185575, 6.2622099, 5.5277596, 6.2615933, -0.5860310, 0.5800164
6: -9.4409475, -8.2036915, -9.4402266, -8.2102938, -0.8682837, 0.8737702
7: -14.8906670, -13.7123652, -14.8832645, -13.7124090, -0.7349756, 0.7289844
8: -3.3202724, -2.2229609, -3.3201318, -2.2244248, -0.6116574, 0.6123393
9: -6.4223604, -5.5652456, -6.4222069, -5.5684242, -0.6709709, 0.6737137

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4576
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266790, upper bound: 0.2269238
time: 4.00 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269233, upper bound: 0.2269235
time: 3.92 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.10 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 30.10
Output dim: 5, lower bound: -0.2266843, upper bound: 0.2266847
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 30.10
Output dim: 5, lower bound: -0.2266843, upper bound: 0.2266847
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.10
Output dim: 5, lower bound: -0.2266790, upper bound: 0.2269238
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.10
Output dim: 5, lower bound: -0.2269233, upper bound: 0.2269235

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -7.3212495, -6.0277967, -7.3212495, -6.0277967, -0.7738166, 0.7738166
1: -11.2155113, -10.1836224, -11.2155113, -10.1836224, -0.6459317, 0.6459317
2: -7.8832331, -6.8467493, -7.8832331, -6.8467493, -0.6081710, 0.6081707
3: -5.0046511, -4.3139176, -5.0046511, -4.3139176, -0.6003184, 0.6003184
4: -7.5120950, -6.6232681, -7.5120950, -6.6232681, -0.8188076, 0.8188078
5: 5.5277605, 6.2613273, 5.5277605, 6.2613273, -0.5777006, 0.5777013
6: -9.4400406, -8.2102928, -9.4400406, -8.2102928, -0.8670707, 0.8670712
7: -14.8832626, -13.7126274, -14.8832626, -13.7126274, -0.7274199, 0.7274199
8: -3.3200922, -2.2244267, -3.3200922, -2.2244267, -0.6108170, 0.6108172
9: -6.4221163, -5.5684290, -6.4221163, -5.5684290, -0.6704438, 0.6704435

Time for backsubstitution: 22.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266719, upper bound: 0.2264263
time: 4.56 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266719, upper bound: 0.2266709
time: 4.16 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -7.3212495, -6.0277967, -7.3225260, -6.0277462, -0.7738805, 0.7755942
1: -11.2155113, -10.1836224, -11.2155743, -10.1832409, -0.6462731, 0.6459463
2: -7.8832331, -6.8467493, -7.8841066, -6.8416414, -0.6132746, 0.6090059
3: -5.0046511, -4.3139176, -5.0049586, -4.3065090, -0.6076412, 0.6005597
4: -7.5120950, -6.6232681, -7.5215316, -6.6229124, -0.8191514, 0.8281302
5: 5.5277605, 6.2613273, 5.5185575, 6.2622099, -0.5785236, 0.5857580
6: -9.4400406, -8.2102928, -9.4409475, -8.2036915, -0.8735785, 0.8673625
7: -14.8832626, -13.7126274, -14.8906670, -13.7123652, -0.7276931, 0.7347496
8: -3.3200922, -2.2244267, -3.3202724, -2.2229609, -0.6122887, 0.6109786
9: -6.4221163, -5.5684290, -6.4223604, -5.5652456, -0.6736245, 0.6705894

Time for backsubstitution: 22.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266719, upper bound: 0.2264268
time: 4.42 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266719, upper bound: 0.2266706
time: 4.17 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7.3225260, -6.0277462, -7.3209934, -6.0278072, -0.7755976, 0.7738867
1: -11.2155743, -10.1832409, -11.2154846, -10.1838655, -0.6457779, 0.6462331
2: -7.8841066, -6.8416414, -7.8833447, -6.8468833, -0.6097393, 0.6133964
3: -5.0049586, -4.3065090, -5.0048223, -4.3143311, -0.6015186, 0.6077976
4: -7.5215316, -6.6229124, -7.5120449, -6.6232939, -0.8281012, 0.8209445
5: 5.5185575, 6.2622099, 5.5277677, 6.2612977, -0.5857227, 0.5800045
6: -9.4409475, -8.2036915, -9.4401360, -8.2103109, -0.8682756, 0.8736782
7: -14.8906670, -13.7123652, -14.8832302, -13.7125721, -0.7348115, 0.7289228
8: -3.3202724, -2.2229609, -3.3201251, -2.2246938, -0.6113770, 0.6123271
9: -6.4223604, -5.5652456, -6.4221668, -5.5687647, -0.6706326, 0.6736865

Time for backsubstitution: 22.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266777, upper bound: 0.2266777
time: 3.98 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266777, upper bound: 0.2269241
time: 4.02 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7.3225236, -6.0277448, -7.3214989, -6.0120730, -0.7891250, 0.7756691
1: -11.2155771, -10.1832438, -11.2300568, -10.1833038, -0.6491327, 0.6582394
2: -7.8841081, -6.8416433, -7.8963060, -6.8467531, -0.6108501, 0.6159995
3: -5.0049572, -4.3065124, -5.0307655, -4.3132477, -0.6089070, 0.6107421
4: -7.5215311, -6.6229134, -7.5311079, -6.6229959, -0.8296442, 0.8382163
5: 5.5185571, 6.2622070, 5.5109091, 6.2623920, -0.5866923, 0.5870984
6: -9.4409456, -8.2036896, -9.4404154, -8.2048712, -0.8737497, 0.8745961
7: -14.8906660, -13.7123661, -14.8926382, -13.7121315, -0.7353361, 0.7350063
8: -3.3202729, -2.2229619, -3.3368878, -2.2241507, -0.6149919, 0.6201320
9: -6.4223590, -5.5652475, -6.4484501, -5.5680413, -0.6745620, 0.6865265

Time for backsubstitution: 22.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4571
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269239, upper bound: 0.2266775
time: 4.13 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269239, upper bound: 0.2269238
time: 4.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.82 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 5, lower bound: -0.2266719, upper bound: 0.2264263
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 5, lower bound: -0.2266719, upper bound: 0.2266709
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 5, lower bound: -0.2266719, upper bound: 0.2264268
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 5, lower bound: -0.2266719, upper bound: 0.2266706
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 5, lower bound: -0.2266777, upper bound: 0.2266777
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 5, lower bound: -0.2266777, upper bound: 0.2269241
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 5, lower bound: -0.2269239, upper bound: 0.2266775
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.82
Output dim: 5, lower bound: -0.2269239, upper bound: 0.2269238

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.3209863, -6.0278316, -7.3212495, -6.0277967, -0.7735486, 0.7737889
1: -11.2154837, -10.1838713, -11.2155113, -10.1836224, -0.6458888, 0.6456831
2: -7.8832006, -6.8468843, -7.8832331, -6.8467493, -0.6081462, 0.6080358
3: -5.0046024, -4.3143315, -5.0046511, -4.3139176, -0.6002455, 0.5998945
4: -7.5120411, -6.6235700, -7.5120950, -6.6232681, -0.8187623, 0.8184905
5: 5.5277691, 6.2610335, 5.5277605, 6.2613273, -0.5776901, 0.5773942
6: -9.4399529, -8.2103119, -9.4400406, -8.2102928, -0.8669791, 0.8670626
7: -14.8832312, -13.7127914, -14.8832626, -13.7126274, -0.7273574, 0.7272558
8: -3.3200874, -2.2246947, -3.3200922, -2.2244267, -0.6108041, 0.6105368
9: -6.4220772, -5.5687661, -6.4221163, -5.5684290, -0.6704161, 0.6701047

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2264256
time: 4.08 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2264260
time: 3.85 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.3214917, -6.0120993, -7.3212461, -6.0277982, -0.7753305, 0.7885416
1: -11.2300549, -10.1833076, -11.2155113, -10.1836243, -0.6581607, 0.6490381
2: -7.8961620, -6.8467546, -7.8832326, -6.8467498, -0.6158366, 0.6091464
3: -5.0305462, -4.3132486, -5.0046496, -4.3139200, -0.6104951, 0.6072826
4: -7.5311074, -6.6232724, -7.5120945, -6.6232705, -0.8376794, 0.8200331
5: 5.5109105, 6.2621264, 5.5277610, 6.2613273, -0.5862489, 0.5809457
6: -9.4402294, -8.2048712, -9.4400406, -8.2102957, -0.8678980, 0.8725352
7: -14.8926344, -13.7123499, -14.8832626, -13.7126284, -0.7346926, 0.7292905
8: -3.3368502, -2.2241507, -3.3200922, -2.2244287, -0.6200311, 0.6141520
9: -6.4483604, -5.5680451, -6.4221172, -5.5684328, -0.6863787, 0.6740348

Time for backsubstitution: 22.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2266722
time: 3.98 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2266724
time: 3.71 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.3209863, -6.0278316, -7.3225260, -6.0277462, -0.7736125, 0.7755661
1: -11.2154837, -10.1838713, -11.2155743, -10.1832409, -0.6462302, 0.6456978
2: -7.8832006, -6.8468843, -7.8841066, -6.8416414, -0.6132498, 0.6088710
3: -5.0046024, -4.3143315, -5.0049586, -4.3065090, -0.6075687, 0.6001353
4: -7.5120411, -6.6235700, -7.5215316, -6.6229124, -0.8191061, 0.8278129
5: 5.5277691, 6.2610335, 5.5185575, 6.2622099, -0.5785122, 0.5854502
6: -9.4399529, -8.2103119, -9.4409475, -8.2036915, -0.8734870, 0.8673544
7: -14.8832312, -13.7127914, -14.8906670, -13.7123652, -0.7276301, 0.7345850
8: -3.3200874, -2.2246947, -3.3202724, -2.2229609, -0.6122758, 0.6106985
9: -6.4220772, -5.5687661, -6.4223604, -5.5652456, -0.6735969, 0.6702504

Time for backsubstitution: 22.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266778, upper bound: 0.2264251
time: 4.38 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266778, upper bound: 0.2264254
time: 4.83 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.3214917, -6.0120993, -7.3225236, -6.0277448, -0.7753940, 0.7890925
1: -11.2300549, -10.1833076, -11.2155771, -10.1832438, -0.6582363, 0.6490524
2: -7.8961620, -6.8467546, -7.8841081, -6.8416433, -0.6158507, 0.6099823
3: -5.0305462, -4.3132486, -5.0049572, -4.3065124, -0.6105120, 0.6075234
4: -7.5311074, -6.6232724, -7.5215311, -6.6229134, -0.8380337, 0.8293562
5: 5.5109105, 6.2621264, 5.5185571, 6.2622070, -0.5870798, 0.5864158
6: -9.4402294, -8.2048712, -9.4409456, -8.2036896, -0.8744049, 0.8728261
7: -14.8926344, -13.7123499, -14.8906660, -13.7123661, -0.7349753, 0.7351098
8: -3.3368502, -2.2241507, -3.3202729, -2.2229619, -0.6200809, 0.6143134
9: -6.4483604, -5.5680451, -6.4223590, -5.5652475, -0.6864357, 0.6741815

Time for backsubstitution: 22.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266778, upper bound: 0.2266715
time: 4.63 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266778, upper bound: 0.2266718
time: 4.23 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.3222637, -6.0277820, -7.3209934, -6.0278072, -0.7753301, 0.7738600
1: -11.2155495, -10.1834888, -11.2154846, -10.1838655, -0.6457345, 0.6459851
2: -7.8840733, -6.8417773, -7.8833447, -6.8468833, -0.6097147, 0.6132610
3: -5.0049100, -4.3069229, -5.0048223, -4.3143311, -0.6014452, 0.6073737
4: -7.5214791, -6.6232138, -7.5120449, -6.6232939, -0.8280563, 0.8206272
5: 5.5185671, 6.2619147, 5.5277677, 6.2612977, -0.5857120, 0.5796974
6: -9.4408579, -8.2037048, -9.4401360, -8.2103109, -0.8681841, 0.8736701
7: -14.8906355, -13.7125244, -14.8832302, -13.7125721, -0.7347481, 0.7287593
8: -3.3202677, -2.2232294, -3.3201251, -2.2246938, -0.6113646, 0.6120474
9: -6.4223218, -5.5655832, -6.4221668, -5.5687647, -0.6706054, 0.6733484

Time for backsubstitution: 22.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4576
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4576

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264263, upper bound: 0.2266774
time: 4.00 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264263, upper bound: 0.2266778
time: 3.92 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -7.3227701, -6.0120492, -7.3209934, -6.0278072, -0.7758436, 0.7883427
1: -11.2301197, -10.1829262, -11.2154846, -10.1838655, -0.6579485, 0.6466699
2: -7.8970361, -6.8416481, -7.8833447, -6.8468833, -0.6165442, 0.6133952
3: -5.0308552, -4.3058376, -5.0048223, -4.3143311, -0.6103678, 0.6088028
4: -7.5405436, -6.6229162, -7.5120449, -6.6232939, -0.8377709, 0.8210618
5: 5.5017071, 6.2630091, 5.5277677, 6.2612977, -0.5862839, 0.5806572
6: -9.4411354, -8.1982689, -9.4401360, -8.2103109, -0.8684735, 0.8791432
7: -14.9000416, -13.7120857, -14.8832302, -13.7125721, -0.7348049, 0.7292647
8: -3.3370304, -2.2226853, -3.3201251, -2.2246938, -0.6199493, 0.6125042
9: -6.4486027, -5.5648594, -6.4221668, -5.5687647, -0.6862001, 0.6741371

Time for backsubstitution: 22.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4576
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4576

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264263, upper bound: 0.2269238
time: 4.21 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264263, upper bound: 0.2269242
time: 4.12 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.3222637, -6.0277820, -7.3214989, -6.0120730, -0.7888560, 0.7743740
1: -11.2155495, -10.1834888, -11.2300568, -10.1833038, -0.6464190, 0.6579909
2: -7.8840733, -6.8417773, -7.8963060, -6.8467531, -0.6098495, 0.6158640
3: -5.0049100, -4.3069229, -5.0307655, -4.3132477, -0.6028721, 0.6103144
4: -7.5214791, -6.6232138, -7.5311079, -6.6229959, -0.8284907, 0.8377547
5: 5.5185671, 6.2619147, 5.5109091, 6.2623920, -0.5866766, 0.5867906
6: -9.4408579, -8.2037048, -9.4404154, -8.2048712, -0.8736582, 0.8739591
7: -14.8906355, -13.7125244, -14.8926382, -13.7121315, -0.7352715, 0.7348421
8: -3.3202677, -2.2232294, -3.3368878, -2.2241507, -0.6118197, 0.6198523
9: -6.4223218, -5.5655832, -6.4484501, -5.5680413, -0.6713943, 0.6861880

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4576
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4576

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266777
time: 3.89 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266781
time: 3.82 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -7.3227701, -6.0120492, -7.3214989, -6.0120730, -0.7850046, 0.7835355
1: -11.2301197, -10.1829262, -11.2300568, -10.1833038, -0.6495132, 0.6497619
2: -7.8970361, -6.8416481, -7.8963060, -6.8467531, -0.6179731, 0.6160026
3: -5.0308552, -4.3058376, -5.0307655, -4.3132477, -0.6100562, 0.6118298
4: -7.5405436, -6.6229162, -7.5311079, -6.6229959, -0.8393278, 0.8340266
5: 5.5017071, 6.2630091, 5.5109091, 6.2623920, -0.5898266, 0.5838218
6: -9.4411354, -8.1982689, -9.4404154, -8.2048712, -0.8705087, 0.8759942
7: -14.9000416, -13.7120857, -14.8926382, -13.7121315, -0.7368708, 0.7308640
8: -3.3370304, -2.2226853, -3.3368878, -2.2241507, -0.6152751, 0.6159570
9: -6.4486027, -5.5648594, -6.4484501, -5.5680413, -0.6806169, 0.6833606

Time for backsubstitution: 22.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4576
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4576

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266777
time: 4.31 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266781
time: 4.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.26 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2264256
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2264260
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2266722
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2266724
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2266778, upper bound: 0.2264251
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2266778, upper bound: 0.2264254
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2266778, upper bound: 0.2266715
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2266778, upper bound: 0.2266718
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2264263, upper bound: 0.2266774
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2264263, upper bound: 0.2266778
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2264263, upper bound: 0.2269238
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2264263, upper bound: 0.2269242
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266777
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266781
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266777
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266781

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.3209863, -6.0278316, -7.3209863, -6.0278316, -0.7735209, 0.7735209
1: -11.2154837, -10.1838713, -11.2154837, -10.1838713, -0.6456404, 0.6456401
2: -7.8832006, -6.8468843, -7.8832006, -6.8468843, -0.6080110, 0.6080110
3: -5.0046024, -4.3143315, -5.0046024, -4.3143315, -0.5998218, 0.5998220
4: -7.5120411, -6.6235700, -7.5120411, -6.6235700, -0.8184447, 0.8184450
5: 5.5277691, 6.2610335, 5.5277691, 6.2610335, -0.5773830, 0.5773828
6: -9.4399529, -8.2103119, -9.4399529, -8.2103119, -0.8669710, 0.8669710
7: -14.8832312, -13.7127914, -14.8832312, -13.7127914, -0.7271934, 0.7271934
8: -3.3200874, -2.2246947, -3.3200874, -2.2246947, -0.6105244, 0.6105242
9: -6.4220772, -5.5687661, -6.4220772, -5.5687661, -0.6700771, 0.6700771

Time for backsubstitution: 22.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4610

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264212, upper bound: 0.2264066
time: 3.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264211, upper bound: 0.2264226
time: 3.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -7.3209863, -6.0278316, -7.3214917, -6.0120993, -0.7882719, 0.7740352
1: -11.2154837, -10.1838713, -11.2300549, -10.1833076, -0.6463249, 0.6579120
2: -7.8832006, -6.8468843, -7.8961620, -6.8467546, -0.6081455, 0.6157010
3: -5.0046024, -4.3143315, -5.0305462, -4.3132486, -0.6012485, 0.6100681
4: -7.5120411, -6.6235700, -7.5311074, -6.6232724, -0.8188796, 0.8373613
5: 5.5277691, 6.2610335, 5.5109105, 6.2621264, -0.5783396, 0.5859411
6: -9.4399529, -8.2103119, -9.4402294, -8.2048712, -0.8724442, 0.8672590
7: -14.8832312, -13.7127914, -14.8926344, -13.7123499, -0.7276983, 0.7345283
8: -3.3200874, -2.2246947, -3.3368502, -2.2241507, -0.6109803, 0.6197517
9: -6.4220772, -5.5687661, -6.4483604, -5.5680451, -0.6708665, 0.6860399

Time for backsubstitution: 22.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 4610

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264212, upper bound: 0.2264066
time: 4.03 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264211, upper bound: 0.2264229
time: 4.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -7.3214917, -6.0120993, -7.3209863, -6.0278316, -0.7740350, 0.7882717
1: -11.2300549, -10.1833076, -11.2154837, -10.1838713, -0.6579120, 0.6463249
2: -7.8961620, -6.8467546, -7.8832006, -6.8468843, -0.6157010, 0.6081457
3: -5.0305462, -4.3132486, -5.0046024, -4.3143315, -0.6100681, 0.6012487
4: -7.5311074, -6.6232724, -7.5120411, -6.6235700, -0.8373613, 0.8188796
5: 5.5109105, 6.2621264, 5.5277691, 6.2610335, -0.5859413, 0.5783398
6: -9.4402294, -8.2048712, -9.4399529, -8.2103119, -0.8672590, 0.8724442
7: -14.8926344, -13.7123499, -14.8832312, -13.7127914, -0.7345281, 0.7276983
8: -3.3368502, -2.2241507, -3.3200874, -2.2246947, -0.6197517, 0.6109800
9: -6.4483604, -5.5680451, -6.4220772, -5.5687661, -0.6860399, 0.6708665

Time for backsubstitution: 22.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4610

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264212, upper bound: 0.2266510
time: 4.02 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264211, upper bound: 0.2266671
time: 4.18 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -7.3214917, -6.0120993, -7.3214917, -6.0120993, -0.7831960, 0.7831957
1: -11.2300549, -10.1833076, -11.2300549, -10.1833076, -0.6494167, 0.6494167
2: -7.8961620, -6.8467546, -7.8961620, -6.8467546, -0.6162694, 0.6158397
3: -5.0305462, -4.3132486, -5.0305462, -4.3132486, -0.6084321, 0.6084323
4: -7.5311074, -6.6232724, -7.5311074, -6.6232724, -0.8318439, 0.8318439
5: 5.5109105, 6.2621264, 5.5109105, 6.2621264, -0.5815039, 0.5815041
6: -9.4402294, -8.2048712, -9.4402294, -8.2048712, -0.8692961, 0.8692961
7: -14.8926344, -13.7123499, -14.8926344, -13.7123499, -0.7292976, 0.7292976
8: -3.3368502, -2.2241507, -3.3368502, -2.2241507, -0.6144352, 0.6144350
9: -6.4483604, -5.5680451, -6.4483604, -5.5680451, -0.6800892, 0.6800890

Time for backsubstitution: 22.36 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.93 + 552.39 = 609.32 seconds
