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
execution time: IAR + RelationalAnalysis = 23.32 + 34.09 = 57.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.2269478, upper bound: 0.2269479

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4576
type: B, layer: 1, pos: 4576
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 4576

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269374, upper bound: 0.2266847
time: 3.67 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269369, upper bound: 0.2269370
time: 3.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.69 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.69
Output dim: 5, lower bound: -0.2269374, upper bound: 0.2266847
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.69
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

Time for backsubstitution: 21.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4576
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269244, upper bound: 0.2264265
time: 4.27 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269244, upper bound: 0.2266709
time: 3.93 seconds

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

Time for backsubstitution: 21.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4571
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4576
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4571

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269239, upper bound: 0.2266788
time: 4.01 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2269239, upper bound: 0.2269231
time: 4.10 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.67 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 29.67
Output dim: 5, lower bound: -0.2269244, upper bound: 0.2264265
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 29.67
Output dim: 5, lower bound: -0.2269244, upper bound: 0.2266709
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 29.67
Output dim: 5, lower bound: -0.2269239, upper bound: 0.2266788
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 29.67
Output dim: 5, lower bound: -0.2269239, upper bound: 0.2269231

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -7.3209863, -6.0278316, -7.3212552, -6.0277715, -0.7735815, 0.7737894
1: -11.2154837, -10.1838713, -11.2155113, -10.1836176, -0.6458933, 0.6456864
2: -7.8832006, -6.8468843, -7.8833771, -6.8467493, -0.6081464, 0.6081829
3: -5.0046024, -4.3143315, -5.0048704, -4.3139172, -0.6002541, 0.6001248
4: -7.5120411, -6.6235700, -7.5120955, -6.6229897, -0.8190513, 0.8185015
5: 5.5277691, 6.2610335, 5.5277600, 6.2615957, -0.5779610, 0.5773952
6: -9.4399529, -8.2103119, -9.4402256, -8.2102938, -0.8669782, 0.8672557
7: -14.8832312, -13.7127914, -14.8832645, -13.7124090, -0.7275817, 0.7272582
8: -3.3200874, -2.2246947, -3.3201313, -2.2244248, -0.6108048, 0.6105881
9: -6.4220772, -5.5687661, -6.4222074, -5.5684242, -0.6704187, 0.6701941

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4576
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266781, upper bound: 0.2264254
time: 4.24 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266781, upper bound: 0.2264255
time: 4.56 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -7.3214917, -6.0120993, -7.3212528, -6.0277719, -0.7753630, 0.7885416
1: -11.2300549, -10.1833076, -11.2155151, -10.1836205, -0.6581655, 0.6490409
2: -7.8961620, -6.8467546, -7.8833790, -6.8467493, -0.6158366, 0.6092937
3: -5.0305462, -4.3132486, -5.0048709, -4.3139210, -0.6105032, 0.6075125
4: -7.5311074, -6.6232724, -7.5120955, -6.6229916, -0.8379707, 0.8200450
5: 5.5109105, 6.2621264, 5.5277591, 6.2615948, -0.5865214, 0.5809469
6: -9.4402294, -8.2048712, -9.4402256, -8.2102928, -0.8678975, 0.8727279
7: -14.8926344, -13.7123499, -14.8832626, -13.7124100, -0.7349186, 0.7292938
8: -3.3368502, -2.2241507, -3.3201313, -2.2244277, -0.6200311, 0.6142030
9: -6.4483604, -5.5680451, -6.4222064, -5.5684280, -0.6863818, 0.6741247

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4576
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266781, upper bound: 0.2266719
time: 3.80 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266781, upper bound: 0.2266718
time: 4.06 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -7.3222637, -6.0277820, -7.3212547, -6.0277715, -0.7753582, 0.7741284
1: -11.2155495, -10.1834888, -11.2155132, -10.1836176, -0.6459832, 0.6460283
2: -7.8840733, -6.8417773, -7.8833771, -6.8467493, -0.6098506, 0.6132855
3: -5.0049100, -4.3069229, -5.0048695, -4.3139172, -0.6018696, 0.6074467
4: -7.5214791, -6.6232138, -7.5120964, -6.6229901, -0.8283739, 0.8206725
5: 5.5185671, 6.2619147, 5.5277596, 6.2615933, -0.5860200, 0.5797093
6: -9.4408579, -8.2037048, -9.4402266, -8.2102938, -0.8681922, 0.8737621
7: -14.8906355, -13.7125244, -14.8832645, -13.7124090, -0.7349122, 0.7288208
8: -3.3202677, -2.2232294, -3.3201318, -2.2244248, -0.6116445, 0.6120596
9: -6.4223218, -5.5655832, -6.4222069, -5.5684242, -0.6709437, 0.6733754

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4576
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266777, upper bound: 0.2266777
time: 3.95 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266777, upper bound: 0.2266777
time: 4.84 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -7.3227701, -6.0120492, -7.3212528, -6.0277729, -0.7771397, 0.7886906
1: -11.2301197, -10.1829262, -11.2155132, -10.1836205, -0.6581967, 0.6493833
2: -7.8970361, -6.8416481, -7.8833771, -6.8467493, -0.6167490, 0.6134417
3: -5.0308552, -4.3058376, -5.0048690, -4.3139200, -0.6107950, 0.6107359
4: -7.5405436, -6.6229162, -7.5120955, -6.6229935, -0.8380885, 0.8222148
5: 5.5017071, 6.2630091, 5.5277600, 6.2615919, -0.5865912, 0.5832641
6: -9.4411354, -8.1982689, -9.4402256, -8.2102928, -0.8691106, 0.8792343
7: -14.9000416, -13.7120857, -14.8832626, -13.7124128, -0.7349687, 0.7308574
8: -3.3370304, -2.2226853, -3.3201318, -2.2244277, -0.6202290, 0.6156740
9: -6.4486027, -5.5648594, -6.4222059, -5.5684285, -0.6865928, 0.6773052

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4571
type: B, layer: 1, pos: 4576
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4571

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266777, upper bound: 0.2269241
time: 3.84 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266777, upper bound: 0.2269241
time: 3.95 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.07 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 5, lower bound: -0.2266781, upper bound: 0.2264254
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 5, lower bound: -0.2266781, upper bound: 0.2264255
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 5, lower bound: -0.2266781, upper bound: 0.2266719
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 5, lower bound: -0.2266781, upper bound: 0.2266718
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 5, lower bound: -0.2266777, upper bound: 0.2266777
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 5, lower bound: -0.2266777, upper bound: 0.2266777
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 5, lower bound: -0.2266777, upper bound: 0.2269241
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 5, lower bound: -0.2266777, upper bound: 0.2269241

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -7.3209863, -6.0278316, -7.3209934, -6.0278053, -0.7735519, 0.7735217
1: -11.2154837, -10.1838713, -11.2154846, -10.1838665, -0.6456447, 0.6456432
2: -7.8832006, -6.8468843, -7.8833447, -6.8468838, -0.6080112, 0.6081583
3: -5.0046024, -4.3143315, -5.0048237, -4.3143311, -0.5998292, 0.6000509
4: -7.5120411, -6.6235700, -7.5120449, -6.6232929, -0.8187342, 0.8184571
5: 5.5277691, 6.2610335, 5.5277677, 6.2613020, -0.5776553, 0.5773842
6: -9.4399529, -8.2103119, -9.4401388, -8.2103109, -0.8669710, 0.8671637
7: -14.8832312, -13.7127914, -14.8832302, -13.7125702, -0.7274182, 0.7271962
8: -3.3200874, -2.2246947, -3.3201265, -2.2246938, -0.6105244, 0.6105752
9: -6.4220772, -5.5687661, -6.4221678, -5.5687647, -0.6700799, 0.6701677

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4576
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4576

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2264263
time: 5.10 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2264268
time: 4.15 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -7.3209863, -6.0278316, -7.3214984, -6.0120730, -0.7883043, 0.7740352
1: -11.2154837, -10.1838713, -11.2300568, -10.1833038, -0.6463299, 0.6579151
2: -7.8832006, -6.8468843, -7.8963079, -6.8467550, -0.6081457, 0.6158490
3: -5.0046024, -4.3143315, -5.0307674, -4.3132486, -0.6012566, 0.6102982
4: -7.5120411, -6.6235700, -7.5311079, -6.6229944, -0.8191700, 0.8373728
5: 5.5277691, 6.2610335, 5.5109096, 6.2623935, -0.5786114, 0.5859425
6: -9.4399529, -8.2103119, -9.4404154, -8.2048721, -0.8724451, 0.8674526
7: -14.8832312, -13.7127914, -14.8926382, -13.7121315, -0.7279236, 0.7345307
8: -3.3200874, -2.2246947, -3.3368893, -2.2241507, -0.6109796, 0.6198030
9: -6.4220772, -5.5687661, -6.4484520, -5.5680399, -0.6708691, 0.6861305

Time for backsubstitution: 21.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4576
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4576

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2264268
time: 4.70 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2264268
time: 4.13 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -7.3214917, -6.0120993, -7.3209934, -6.0278053, -0.7740660, 0.7882721
1: -11.2300549, -10.1833076, -11.2154846, -10.1838665, -0.6579168, 0.6463280
2: -7.8961620, -6.8467546, -7.8833447, -6.8468838, -0.6157012, 0.6082928
3: -5.0305462, -4.3132486, -5.0048237, -4.3143311, -0.6100762, 0.6014776
4: -7.5311074, -6.6232724, -7.5120449, -6.6232929, -0.8376532, 0.8188920
5: 5.5109105, 6.2621264, 5.5277677, 6.2613020, -0.5862141, 0.5783412
6: -9.4402294, -8.2048712, -9.4401388, -8.2103109, -0.8672585, 0.8726373
7: -14.8926344, -13.7123499, -14.8832302, -13.7125702, -0.7347546, 0.7277012
8: -3.3368502, -2.2241507, -3.3201265, -2.2246938, -0.6197519, 0.6110311
9: -6.4483604, -5.5680451, -6.4221678, -5.5687647, -0.6860430, 0.6709571

Time for backsubstitution: 21.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4576
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4576

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2266708
time: 5.15 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2266712
time: 4.70 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -7.3214917, -6.0120993, -7.3214984, -6.0120730, -0.7832279, 0.7831962
1: -11.2300549, -10.1833076, -11.2300568, -10.1833038, -0.6494212, 0.6494195
2: -7.8961620, -6.8467546, -7.8963079, -6.8467550, -0.6162698, 0.6159880
3: -5.0305462, -4.3132486, -5.0307674, -4.3132486, -0.6084397, 0.6086617
4: -7.5311074, -6.6232724, -7.5311079, -6.6229944, -0.8321338, 0.8318558
5: 5.5109105, 6.2621264, 5.5109096, 6.2623935, -0.5817752, 0.5815051
6: -9.4402294, -8.2048712, -9.4404154, -8.2048721, -0.8692961, 0.8694882
7: -14.8926344, -13.7123499, -14.8926382, -13.7121315, -0.7295220, 0.7293010
8: -3.3368502, -2.2241507, -3.3368893, -2.2241507, -0.6144350, 0.6144862
9: -6.4483604, -5.5680451, -6.4484520, -5.5680399, -0.6800921, 0.6801808

Time for backsubstitution: 21.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4576
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4576

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2266716
time: 4.47 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2266719
time: 3.94 seconds

## BFS NS instance: NS_A2_A1_B1

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

Time for backsubstitution: 21.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4576
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4576

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266786
time: 5.00 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266794
time: 4.08 seconds

## BFS NS instance: NS_A2_A1_B2

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

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4576
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4576

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266790
time: 3.74 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266794
time: 5.58 seconds

## BFS NS instance: NS_A2_A2_B1

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

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4576
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4576

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2269235
time: 3.88 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2269241
time: 4.59 seconds

## BFS NS instance: NS_A2_A2_B2

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

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4576
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4576

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2269241
time: 4.03 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2269245
time: 4.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.90 seconds
NS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2264263
NS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2264268
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2264268
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2264268
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2266708
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2266712
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2266716
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264256, upper bound: 0.2266719
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266786
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266794
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266790
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2266794
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2269235
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2269241
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2269241
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.90
Output dim: 5, lower bound: -0.2264250, upper bound: 0.2269245

## BFS NS instance: NS_A1_A1_B1_B1

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

Time for backsubstitution: 21.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 4610

## Relational analysis of NS_A1_A1_B1_B1_A1

### Relational analysis result of NS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264225, upper bound: 0.2264060
time: 3.85 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2

### Relational analysis result of NS_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264225, upper bound: 0.2264223
time: 4.15 seconds

## BFS NS instance: NS_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -7.3209863, -6.0278316, -7.3222637, -6.0277820, -0.7735844, 0.7752986
1: -11.2154837, -10.1838713, -11.2155495, -10.1834888, -0.6459823, 0.6456547
2: -7.8832006, -6.8468843, -7.8840733, -6.8417773, -0.6131146, 0.6088469
3: -5.0046024, -4.3143315, -5.0049100, -4.3069229, -0.6071444, 0.6000619
4: -7.5120411, -6.6235700, -7.5214791, -6.6232138, -0.8187885, 0.8277681
5: 5.5277691, 6.2610335, 5.5185671, 6.2619147, -0.5782056, 0.5854394
6: -9.4399529, -8.2103119, -9.4408579, -8.2037048, -0.8734789, 0.8672624
7: -14.8832312, -13.7127914, -14.8906355, -13.7125244, -0.7274666, 0.7345216
8: -3.3200874, -2.2246947, -3.3202677, -2.2232294, -0.6119962, 0.6106861
9: -6.4220772, -5.5687661, -6.4223218, -5.5655832, -0.6732583, 0.6702235

Time for backsubstitution: 22.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572
type: B, layer: 1, pos: 4572

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 4610

## Relational analysis of NS_A1_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264062, upper bound: 0.2264225
time: 3.96 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2

### Relational analysis result of NS_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2264224, upper bound: 0.2264223
time: 4.37 seconds

## BFS NS instance: NS_A1_A1_B2_B1

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

Time for backsubstitution: 22.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4572
type: A, layer: 1, pos: 4610
type: B, layer: 1, pos: 4610
type: A, layer: 1, pos: 4572

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4572

## Relational analysis of NS_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266700, upper bound: 0.2264248
time: 3.96 seconds

## Relational analysis of NS_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2266697, upper bound: 0.2264262
time: 4.02 seconds

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -7.3209863, -6.0278316, -7.3227701, -6.0120492, -0.7883348, 0.7758121
1: -11.2154837, -10.1838713, -11.2301197, -10.1829262, -0.6466670, 0.6579275
2: -7.8832006, -6.8468843, -7.8970361, -6.8416481, -0.6132486, 0.6165335
3: -5.0046024, -4.3143315, -5.0308552, -4.3058376, -0.6085739, 0.6103103
4: -7.5120411, -6.6235700, -7.5405436, -6.6229162, -0.8192234, 0.8374791
5: 5.5277691, 6.2610335, 5.5017071, 6.2630091, -0.5791650, 0.5860114
6: -9.4399529, -8.2103119, -9.4411354, -8.1982689, -0.8789520, 0.8675499
7: -14.8832312, -13.7127914, -14.9000416, -13.7120857, -0.7279716, 0.7345786
8: -3.3200874, -2.2246947, -3.3370304, -2.2226853, -0.6124535, 0.6199114
9: -6.4220772, -5.5687661, -6.4486027, -5.5648594, -0.6740472, 0.6861968

Time for backsubstitution: 22.24 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.41 + 549.95 = 607.35 seconds
