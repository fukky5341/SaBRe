## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 912.9840697103999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-631.4873047, 411.7176819, -631.4873047, 411.7176819, -1043.2049561, 1043.2049561)
1: (-47.2466049, 36.8136635, -47.2466049, 36.8136635, -84.0602646, 84.0602646)
2: (-38.6318474, 55.3112831, -38.6318474, 55.3112831, -93.9431152, 93.9431152)
3: (-44.2114601, 87.6876907, -44.2114601, 87.6876907, -131.8991547, 131.8991547)
4: (-33.6581497, 55.1203232, -33.6581497, 55.1203232, -88.7784729, 88.7784729)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.12 + 1.53 = 4.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -916.6506724, upper bound: 916.6506724

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.3384520, upper bound: 914.9857326
time: 0.47 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -916.6168118, upper bound: 916.6168114
time: 0.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.19 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.19
Output dim: 0, lower bound: -916.3384520, upper bound: 914.9857326
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.19
Output dim: 0, lower bound: -916.6168118, upper bound: 916.6168114

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -511.7307129, 333.6546936, -588.4508057, 386.1644287, -897.8951416, 922.1054688
1: -38.9493980, 29.6902771, -44.5147667, 34.2707787, -73.2201767, 74.2050247
2: -32.5045624, 44.4840584, -36.5841789, 51.7268448, -84.2314072, 81.0682373
3: -36.7355003, 70.9758224, -41.7016068, 82.0571289, -118.7926025, 112.6774216
4: -27.8541012, 44.5239906, -31.7192993, 51.6075249, -79.4616241, 76.2432861

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9857326, upper bound: 914.9857326
time: 0.46 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9857326, upper bound: 914.9857326
time: 0.46 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -534.0440063, 349.2824707, -608.1965942, 397.9551392, -931.9991455, 957.4790649
1: -40.7368507, 31.0746841, -45.7593231, 35.4486732, -76.1855240, 76.8339844
2: -33.6063690, 46.8075294, -37.5224533, 53.4315834, -87.0379486, 84.3299866
3: -38.0319176, 74.4518585, -42.8564987, 84.6857834, -122.7176971, 117.3083420
4: -28.9225883, 46.8524361, -32.6301079, 53.2674866, -82.1900635, 79.4825439

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9857326, upper bound: 916.3384520
time: 0.44 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9857326, upper bound: 916.6168118
time: 0.45 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.03 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 0, lower bound: -914.9857326, upper bound: 914.9857326
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 0, lower bound: -914.9857326, upper bound: 914.9857326
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 0, lower bound: -914.9857326, upper bound: 916.3384520
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.03
Output dim: 0, lower bound: -914.9857326, upper bound: 916.6168118

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -511.7307129, 333.6546936, -511.7307129, 333.6546936, -845.3853760, 845.3853760
1: -38.9493980, 29.6902771, -38.9493980, 29.6902771, -68.6396790, 68.6396790
2: -32.5045624, 44.4840584, -32.5045624, 44.4840584, -76.9886169, 76.9886169
3: -36.7355003, 70.9758224, -36.7355003, 70.9758224, -107.7112961, 107.7112961
4: -27.8541012, 44.5239906, -27.8541012, 44.5239906, -72.3780899, 72.3780899

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7769341, upper bound: 914.9379697
time: 0.43 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7769341, upper bound: 914.8062206
time: 0.45 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -511.7307129, 333.6546936, -532.9789429, 348.5797729, -860.3104248, 866.6336670
1: -38.9493980, 29.6902771, -40.6601448, 31.0109177, -69.9603119, 70.3504181
2: -32.5045624, 44.4840584, -33.5497093, 46.7177048, -79.2222672, 78.0337677
3: -36.7355003, 70.9758224, -37.9622993, 74.3035583, -111.0390320, 108.9381180
4: -27.8541012, 44.5239906, -28.8685322, 46.7647743, -74.6188736, 73.3925247

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7769341, upper bound: 914.9379697
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7769341, upper bound: 914.8062206
time: 0.44 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -534.0440063, 349.2824707, -511.7307129, 333.6546936, -867.6986694, 861.0131836
1: -40.7368507, 31.0746841, -38.9493980, 29.6902771, -70.4271240, 70.0240784
2: -33.6063690, 46.8075294, -32.5045624, 44.4840584, -78.0904236, 79.3120880
3: -38.0319176, 74.4518585, -36.7355003, 70.9758224, -109.0077286, 111.1873398
4: -28.9225883, 46.8524361, -27.8541012, 44.5239906, -73.4465714, 74.7065353

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9501881, upper bound: 916.2970887
time: 0.45 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9857326, upper bound: 916.3303561
time: 0.46 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -534.0440063, 349.2824707, -534.0440063, 349.2824707, -883.3264771, 883.3264771
1: -40.7368507, 31.0746841, -40.7368507, 31.0746841, -71.8115387, 71.8115387
2: -33.6063690, 46.8075294, -33.6063690, 46.8075294, -80.4138947, 80.4138947
3: -38.0319176, 74.4518585, -38.0319176, 74.4518585, -112.4837799, 112.4837799
4: -28.9225883, 46.8524361, -28.9225883, 46.8524361, -75.7750168, 75.7750168

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9501881, upper bound: 916.6158722
time: 0.46 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9857326, upper bound: 916.5777236
time: 0.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.04 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -914.7769341, upper bound: 914.9379697
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -914.7769341, upper bound: 914.8062206
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -914.7769341, upper bound: 914.9379697
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -914.7769341, upper bound: 914.8062206
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -914.9501881, upper bound: 916.2970887
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -914.9857326, upper bound: 916.3303561
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -914.9501881, upper bound: 916.6158722
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 0, lower bound: -914.9857326, upper bound: 916.5777236

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -497.7616577, 320.8284912, -501.4554749, 326.9696960, -824.7313232, 822.2839355
1: -37.5259361, 28.8399277, -38.1673584, 29.0861645, -66.6120758, 67.0072861
2: -31.2865143, 42.9721069, -31.9052010, 43.5773773, -74.8638916, 74.8773041
3: -35.3774185, 68.8286667, -36.0479927, 69.5414581, -104.9188766, 104.8766632
4: -26.8597603, 42.9603653, -27.3339596, 43.6170158, -70.4767761, 70.2943115

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.6355768
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.8062206, upper bound: 914.9379697
time: 0.43 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -442.2951965, 271.8350830, -495.7272949, 320.6048889, -762.9000244, 767.5623169
1: -32.2100601, 25.3963146, -37.5139580, 28.6989040, -60.9089622, 62.9102707
2: -27.2675781, 36.3211746, -31.2848434, 42.7335472, -70.0011139, 67.6060181
3: -30.5139713, 59.8067436, -35.3315773, 68.4257431, -98.9396973, 95.1383209
4: -23.4027367, 36.5298805, -26.8272552, 42.8259697, -66.2287064, 63.3571358

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4776474
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.8062206, upper bound: 914.8062206
time: 0.43 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -497.7616577, 320.8284912, -515.8956909, 337.1313782, -834.8930054, 836.7241211
1: -37.5259361, 28.8399277, -39.3389473, 29.9772015, -67.5031281, 68.1788788
2: -31.2865143, 42.9721069, -32.5166550, 45.1506157, -76.4371338, 75.4887543
3: -35.3774185, 68.8286667, -36.7746277, 71.7892456, -107.1666565, 105.6032944
4: -26.8597603, 42.9603653, -27.9633255, 45.1981049, -72.0578613, 70.9236908

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.6298566
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7769341, upper bound: 914.9379697
time: 0.47 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -442.2951965, 271.8350830, -520.3674927, 337.9748535, -780.2699585, 792.2025757
1: -32.2100601, 25.3963146, -39.5147285, 30.2221680, -62.4322281, 64.9110413
2: -27.2675781, 36.3211746, -32.5637321, 45.3080292, -72.5756073, 68.8848953
3: -30.5139713, 59.8067436, -36.8233376, 72.2299957, -102.7439651, 96.6300659
4: -23.4027367, 36.5298805, -27.9990940, 45.3912926, -68.7940216, 64.5289764

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4716579
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7769341, upper bound: 914.8062206
time: 0.46 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -449.8942566, 269.8144531, -470.2463989, 302.4437561, -752.3379517, 740.0608521
1: -32.4729729, 25.6307487, -35.5299530, 27.0983505, -59.5713120, 61.1607018
2: -28.1071949, 36.3307343, -30.0980110, 40.2008781, -68.3080750, 66.4287415
3: -31.2799892, 60.3920097, -33.8515244, 64.5269318, -95.8069229, 94.2435150
4: -24.0955753, 36.6130714, -25.8732605, 40.3233490, -64.4189148, 62.4863319

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.6298566, upper bound: 914.2752993
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4716579, upper bound: 914.2752993
time: 0.46 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -491.2350464, 315.5464478, -505.8934021, 328.7570496, -819.9920654, 821.4396973
1: -37.0980110, 28.3219643, -38.4302254, 29.3141804, -66.4121933, 66.7521896
2: -31.0157280, 42.1671295, -32.1249733, 43.8256721, -74.8414001, 74.2920837
3: -34.8149338, 67.4614563, -36.2840118, 70.0288391, -104.8437653, 103.7454605
4: -26.6137142, 42.2937050, -27.5078354, 43.8815498, -70.4952621, 69.8015442

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.9379697, upper bound: 914.7769341
time: 0.45 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.8062206, upper bound: 914.7769341
time: 0.45 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -449.8942566, 269.8144531, -491.7192993, 316.7485352, -766.6426392, 761.5337524
1: -32.4729729, 25.6307487, -37.2488632, 28.3684044, -60.8413773, 62.8796043
2: -28.1071949, 36.3307343, -31.2079849, 42.3677979, -70.4749908, 67.5387115
3: -31.2799892, 60.3920097, -35.0360947, 67.6867371, -98.9667282, 95.4281006
4: -24.0955753, 36.6130714, -26.7939739, 42.5173798, -66.6129532, 63.4070435

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7884354, upper bound: 914.2752993
time: 0.46 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4488701, upper bound: 914.2752993
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -491.2350464, 315.5464478, -525.8684692, 343.0766296, -834.3116455, 841.4147339
1: -37.0980110, 28.3219643, -40.0708694, 30.5537052, -67.6517181, 68.3928375
2: -31.0157280, 42.1671295, -33.1103058, 45.9517593, -76.9674835, 75.2773972
3: -34.8149338, 67.4614563, -37.4440536, 73.1466064, -107.9615326, 104.9055099
4: -26.6137142, 42.2937050, -28.4714813, 46.0104141, -72.6241302, 70.7651825

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -915.0779499, upper bound: 914.7769341
time: 0.46 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7769341, upper bound: 914.7769341
time: 0.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.07 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.3037242, upper bound: 914.6355768
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.8062206, upper bound: 914.9379697
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4776474
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.8062206, upper bound: 914.8062206
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.2752993, upper bound: 914.6298566
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.7769341, upper bound: 914.9379697
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4716579
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.7769341, upper bound: 914.8062206
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.6298566, upper bound: 914.2752993
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.4716579, upper bound: 914.2752993
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.9379697, upper bound: 914.7769341
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.8062206, upper bound: 914.7769341
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.7884354, upper bound: 914.2752993
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.4488701, upper bound: 914.2752993
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -915.0779499, upper bound: 914.7769341
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 0, lower bound: -914.7769341, upper bound: 914.7769341

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -457.9891663, 292.0267029, -392.5830078, 222.9909668, -680.9801025, 684.6097412
1: -34.3598442, 26.3446579, -27.2820072, 22.0797024, -56.4395409, 53.6266632
2: -29.0704346, 38.9475479, -24.2352200, 30.0072498, -59.0776825, 63.1827621
3: -32.7265167, 62.7096558, -26.8576736, 51.8467751, -84.5732727, 89.5673218
4: -24.9501266, 39.0196571, -20.9136658, 30.3205070, -55.2706337, 59.9333191

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4972110
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.6355768
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -491.8106995, 316.0624084, -467.3150940, 297.4577637, -789.2684326, 783.3774414
1: -37.0275993, 28.4567299, -35.0203171, 26.8802433, -63.9078407, 63.4770432
2: -30.9230728, 42.3323746, -29.6408157, 39.6168976, -70.5399628, 71.9731903
3: -34.9423218, 67.8971176, -33.3057785, 63.8680191, -98.8103256, 101.2028809
4: -26.5278683, 42.3359070, -25.4721279, 39.7315254, -66.2593918, 67.8080368

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.5103019, upper bound: 914.4972110
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.5103019, upper bound: 914.9379697
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -410.5568237, 244.8344574, -397.8999023, 225.0948639, -635.6516724, 642.7343140
1: -29.2897682, 23.3661289, -27.5871696, 22.3813782, -51.6711464, 50.9532967
2: -25.3300896, 32.6569862, -24.6542530, 30.3059063, -55.6359901, 57.3112411
3: -28.2508698, 54.9206467, -27.3039207, 52.5297127, -80.7805710, 82.2245560
4: -21.8066769, 32.9080772, -21.2676601, 30.6166878, -52.4233551, 54.1757278

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.3037242
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4776474
time: 0.47 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -436.9631653, 266.6635742, -461.6280518, 290.0481262, -727.0111694, 728.2916260
1: -31.6598587, 25.0489330, -34.2528915, 26.4951687, -58.1550217, 59.3018265
2: -26.9165382, 35.6312180, -29.1179352, 38.6470680, -65.5636063, 64.7491455
3: -30.0988865, 58.9689484, -32.6342964, 62.7455788, -92.8444672, 91.6032410
4: -23.1032200, 35.8462906, -25.0040741, 38.8004837, -61.9037018, 60.8503647

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4776474, upper bound: 914.3037242
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4776474, upper bound: 914.8062206
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -457.9891663, 292.0267029, -430.7501831, 255.6035767, -713.5927734, 722.7767944
1: -34.3598442, 26.3446579, -30.8480129, 24.4704437, -58.8302841, 57.1926689
2: -29.0704346, 38.9475479, -26.7775211, 34.4090462, -63.4794807, 65.7250595
3: -32.7265167, 62.7096558, -29.7838764, 57.5557404, -90.2822571, 92.4935226
4: -24.9501266, 39.0196571, -22.9879742, 34.6982079, -59.6483345, 62.0076256

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4972110
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.6298566
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -491.8106995, 316.0624084, -475.1539917, 304.6419678, -796.4526367, 791.2161865
1: -37.0275993, 28.4567299, -35.8461113, 27.3448830, -64.3724747, 64.3028412
2: -30.9230728, 42.3323746, -30.0297451, 40.6656189, -71.5886841, 72.3621216
3: -34.9423218, 67.8971176, -33.6943398, 65.0930023, -100.0353165, 101.5914612
4: -26.5278683, 42.3359070, -25.7831535, 40.8087845, -67.3366394, 68.1190643

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4859956, upper bound: 914.4972110
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4859956, upper bound: 914.9379697
time: 0.47 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -410.5568237, 244.8344574, -443.5166626, 263.2473755, -673.8040161, 688.3511353
1: -29.2897682, 23.3661289, -31.7979584, 25.2185326, -54.5083008, 55.1640816
2: -25.3300896, 32.6569862, -27.6969280, 35.4392509, -60.7693405, 60.3539124
3: -28.2508698, 54.9206467, -30.7875862, 59.3504715, -87.6013336, 85.7082367
4: -21.8066769, 32.9080772, -23.7579861, 35.7358627, -57.5425301, 56.6660576

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.3037242
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4716579
time: 0.44 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -436.9631653, 266.6635742, -478.6080627, 304.4172363, -741.3803101, 745.2716064
1: -31.6598587, 25.0489330, -35.8904762, 27.5209770, -59.1808319, 60.9394073
2: -26.9165382, 35.6312180, -30.0953350, 40.6848412, -67.6013794, 65.7265472
3: -30.0988865, 58.9689484, -33.7341881, 65.3450928, -95.4439774, 92.7031403
4: -23.1032200, 35.8462906, -25.8193207, 40.8501434, -63.9533615, 61.6656113

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4555440, upper bound: 914.3037242
time: 0.46 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4555440, upper bound: 914.8062206
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -430.8133240, 255.6743164, -457.9891663, 292.0267029, -722.8400269, 713.6634521
1: -30.8552132, 24.4746971, -34.3598442, 26.3446579, -57.1998711, 58.8345413
2: -26.7811298, 34.4177246, -29.0704346, 38.9475479, -65.7286758, 63.4881592
3: -29.7882252, 57.5652084, -32.7265167, 62.7096558, -92.4978714, 90.2917252
4: -22.9909534, 34.7068863, -24.9501266, 39.0196571, -62.0106087, 59.6570129

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4716579, upper bound: 914.2752993
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4716579, upper bound: 914.2752993
time: 0.47 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -443.5166626, 263.2473755, -410.5568237, 244.8344574, -688.3511353, 673.8040161
1: -31.7979584, 25.2185326, -29.2897682, 23.3661289, -55.1640816, 54.5083008
2: -27.6969280, 35.4392509, -25.3300896, 32.6569862, -60.3539124, 60.7693405
3: -30.7875862, 59.3504715, -28.2508698, 54.9206467, -85.7082367, 87.6013336
4: -23.7579861, 35.7358627, -21.8066769, 32.9080772, -56.6660576, 57.5425301

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4716579, upper bound: 914.2752993
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4716579, upper bound: 914.2752993
time: 0.47 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -475.4012756, 304.8463135, -491.8106995, 316.0624084, -791.4634399, 796.6569214
1: -35.8683968, 27.3607864, -37.0275993, 28.4567299, -64.3251266, 64.3883820
2: -30.0433178, 40.6918030, -30.9230728, 42.3323746, -72.3756943, 71.6148758
3: -33.7101631, 65.1328278, -34.9423218, 67.8971176, -101.6072769, 100.0751266
4: -25.7950935, 40.8342400, -26.5278683, 42.3359070, -68.1309967, 67.3620911

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.8062206, upper bound: 914.7769341
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.8062206, upper bound: 914.7769341
time: 0.48 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -478.6080627, 304.4172363, -436.9631653, 266.6635742, -745.2716064, 741.3803101
1: -35.8904762, 27.5209770, -31.6598587, 25.0489330, -60.9394073, 59.1808319
2: -30.0953350, 40.6848412, -26.9165382, 35.6312180, -65.7265472, 67.6013794
3: -33.7341881, 65.3450928, -30.0988865, 58.9689484, -92.7031403, 95.4439774
4: -25.8193207, 40.8501434, -23.1032200, 35.8462906, -61.6656113, 63.9533615

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.8062206, upper bound: 914.7769341
time: 0.45 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.8062206, upper bound: 914.7769341
time: 0.45 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -430.8133240, 255.6743164, -468.7427063, 300.9948425, -731.8081665, 724.4169922
1: -30.8552132, 24.4746971, -35.3560333, 27.0256138, -57.8808212, 59.8307304
2: -26.7811298, 34.4177246, -29.4659023, 40.2800636, -67.0611877, 63.8836288
3: -29.7882252, 57.5652084, -33.2280464, 64.4276047, -94.2158279, 90.7932510
4: -22.9909534, 34.7068863, -25.3361225, 40.3726387, -63.3635788, 60.0430031

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4488701, upper bound: 914.2752993
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4488701, upper bound: 914.2752993
time: 0.48 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -443.5166626, 263.2473755, -435.4045410, 263.7632751, -707.2799072, 698.6518555
1: -31.7979584, 25.2185326, -31.4939156, 24.8579311, -56.6558914, 56.7124405
2: -27.6969280, 35.4392509, -26.8571606, 35.3452148, -63.0421448, 62.2964096
3: -30.7875862, 59.3504715, -29.9770126, 58.4413910, -89.2289734, 89.3274689
4: -23.7579861, 35.7358627, -23.0626011, 35.6212006, -59.3791847, 58.7984581

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4488701, upper bound: 914.2752993
time: 0.48 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4488701, upper bound: 914.2752993
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -475.4012756, 304.8463135, -508.1438599, 329.1799011, -804.5810547, 812.9901123
1: -35.8683968, 27.3607864, -38.4071121, 29.5035381, -65.3719330, 65.7678986
2: -30.0433178, 40.6918030, -31.6709480, 44.1785507, -74.2218704, 72.3627472
3: -33.7101631, 65.1328278, -35.8798027, 70.4257889, -104.1359558, 101.0126114
4: -25.7950935, 40.8342400, -27.2797260, 44.1705551, -69.9656525, 68.1139450

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7769341, upper bound: 914.7769341
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7769341, upper bound: 914.7769341
time: 0.46 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -478.6080627, 304.4172363, -463.5883789, 288.9784546, -767.5865479, 768.0056152
1: -35.8904762, 27.5209770, -34.2149048, 26.6473064, -62.5377808, 61.7358818
2: -30.0953350, 40.6848412, -28.5121574, 38.7788620, -68.8741837, 69.1969833
3: -33.7341881, 65.3450928, -31.9227734, 62.8190155, -96.5531998, 97.2678604
4: -25.8193207, 40.8501434, -24.4608536, 39.0140572, -64.8333740, 65.3109970

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7769341, upper bound: 914.7769341
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7769341, upper bound: 914.7769341
time: 0.47 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.21 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4972110
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.3037242, upper bound: 914.6355768
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.5103019, upper bound: 914.4972110
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.5103019, upper bound: 914.9379697
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.3037242, upper bound: 914.3037242
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4776474
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.4776474, upper bound: 914.3037242
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.4776474, upper bound: 914.8062206
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4972110
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.2752993, upper bound: 914.6298566
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.4859956, upper bound: 914.4972110
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.4859956, upper bound: 914.9379697
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.2752993, upper bound: 914.3037242
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4716579
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.4555440, upper bound: 914.3037242
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.4555440, upper bound: 914.8062206
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.4716579, upper bound: 914.2752993
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.4716579, upper bound: 914.2752993
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.4716579, upper bound: 914.2752993
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.4716579, upper bound: 914.2752993
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.8062206, upper bound: 914.7769341
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.8062206, upper bound: 914.7769341
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.8062206, upper bound: 914.7769341
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.8062206, upper bound: 914.7769341
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.4488701, upper bound: 914.2752993
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.4488701, upper bound: 914.2752993
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.4488701, upper bound: 914.2752993
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.4488701, upper bound: 914.2752993
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.7769341, upper bound: 914.7769341
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.7769341, upper bound: 914.7769341
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.7769341, upper bound: 914.7769341
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 0, lower bound: -914.7769341, upper bound: 914.7769341

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -378.7266846, 212.8682556, -392.5830078, 222.9909668, -601.7176514, 605.4512939
1: -26.0240211, 21.2517738, -27.2820072, 22.0797024, -48.1037102, 48.5337715
2: -23.0641403, 28.6052265, -24.2352200, 30.0072498, -53.0713882, 52.8404465
3: -25.5927620, 49.8398476, -26.8576736, 51.8467751, -77.4395370, 76.6975098
4: -19.9385319, 28.9139767, -20.9136658, 30.3205070, -50.2590408, 49.8276291

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4972110
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4972110
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -471.1083984, 298.5914001, -392.5830078, 222.9909668, -694.0993042, 691.1744385
1: -35.1998482, 27.1121502, -27.2820072, 22.0797024, -57.2795448, 54.3941460
2: -29.6198521, 40.0117264, -24.2352200, 30.0072498, -59.6270943, 64.2469406
3: -33.3794823, 64.5453186, -26.8576736, 51.8467751, -85.2262573, 91.4029846
4: -25.4255333, 40.0587883, -20.9136658, 30.3205070, -55.7460365, 60.9724541

Time for backsubstitution: 3.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.6355768
time: 0.43 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.6355768
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -378.7534790, 212.9007263, -467.3150940, 297.4577637, -676.2112427, 680.2157593
1: -26.0272770, 21.2535553, -35.0203171, 26.8802433, -52.9075203, 56.2738724
2: -23.0663757, 28.6091671, -29.6408157, 39.6168976, -62.6832733, 58.2499809
3: -25.5954723, 49.8440170, -33.3057785, 63.8680191, -89.4634933, 83.1497650
4: -19.9403934, 28.9177246, -25.4721279, 39.7315254, -59.6719131, 54.3898544

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4972110
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4972110
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -471.1083984, 298.5914001, -467.3150940, 297.4577637, -768.5661621, 765.9064331
1: -35.1998482, 27.1121502, -35.0203171, 26.8802433, -62.0800934, 62.1324615
2: -29.6198521, 40.0117264, -29.6408157, 39.6168976, -69.2367477, 69.6525345
3: -33.3794823, 64.5453186, -33.3057785, 63.8680191, -97.2474976, 97.8510742
4: -25.4255333, 40.0587883, -25.4721279, 39.7315254, -65.1570587, 65.5309143

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.9379697
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.9379697
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -375.1780396, 201.6742096, -397.8999023, 225.0948639, -600.2728882, 599.5740967
1: -25.1377811, 20.9536781, -27.5871696, 22.3813782, -47.5191536, 48.5408478
2: -23.0599976, 27.3185673, -24.6542530, 30.3059063, -53.3658981, 51.9728203
3: -25.4025040, 48.9396896, -27.3039207, 52.5297127, -77.9322128, 76.2435837
4: -19.9510593, 27.6362190, -21.2676601, 30.6166878, -50.5677490, 48.9038773

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.3037242
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.3037242
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -411.9146423, 241.9918213, -397.8999023, 225.0948639, -637.0095215, 639.8916626
1: -29.0321484, 23.3863163, -27.5871696, 22.3813782, -51.4135284, 50.9734879
2: -25.2123051, 32.3396187, -24.6542530, 30.3059063, -55.5182037, 56.9938736
3: -28.0879021, 54.9645538, -27.3039207, 52.5297127, -80.6175995, 82.2684784
4: -21.6818123, 32.5851517, -21.2676601, 30.6166878, -52.2985001, 53.8528061

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4776474
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4776474
time: 0.47 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -375.1780396, 201.6742096, -461.6280518, 290.0481262, -665.2261353, 663.3022461
1: -25.1377811, 20.9536781, -34.2528915, 26.4951687, -51.6329498, 55.2065697
2: -23.0599976, 27.3185673, -29.1179352, 38.6470680, -61.7070618, 56.4365005
3: -25.4025040, 48.9396896, -32.6342964, 62.7455788, -88.1480865, 81.5739670
4: -19.9510593, 27.6362190, -25.0040741, 38.8004837, -58.7515411, 52.6402931

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.3037242
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.3037242
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -412.3837891, 242.4407043, -461.6280518, 290.0481262, -702.4318848, 704.0687256
1: -29.0793724, 23.4184208, -34.2528915, 26.4951687, -55.5745392, 57.6713104
2: -25.2454643, 32.4001427, -29.1179352, 38.6470680, -63.8925285, 61.5180779
3: -28.1270676, 55.0434875, -32.6342964, 62.7455788, -90.8726425, 87.6777802
4: -21.7095318, 32.6449890, -25.0040741, 38.8004837, -60.5100098, 57.6490631

Time for backsubstitution: 3.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.8062206
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.8062206
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -378.7266846, 212.8682556, -430.7501831, 255.6035767, -634.3302612, 643.6183472
1: -26.0240211, 21.2517738, -30.8480129, 24.4704437, -50.4944572, 52.0997734
2: -23.0641403, 28.6052265, -26.7775211, 34.4090462, -57.4731827, 55.3827438
3: -25.5927620, 49.8398476, -29.7838764, 57.5557404, -83.1484985, 79.6237030
4: -19.9385319, 28.9139767, -22.9879742, 34.6982079, -54.6367378, 51.9019394

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4972110
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4972110
time: 0.47 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -471.1083984, 298.5914001, -430.7501831, 255.6035767, -726.7119751, 729.3414307
1: -35.1998482, 27.1121502, -30.8480129, 24.4704437, -59.6702881, 57.9601517
2: -29.6198521, 40.0117264, -26.7775211, 34.4090462, -64.0288849, 66.7892303
3: -33.3794823, 64.5453186, -29.7838764, 57.5557404, -90.9352264, 94.3291779
4: -25.4255333, 40.0587883, -22.9879742, 34.6982079, -60.1237259, 63.0467606

Time for backsubstitution: 3.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.6298566
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.6298566
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -378.7534790, 212.9007263, -475.1539917, 304.6419678, -683.3954468, 688.0546875
1: -26.0272770, 21.2535553, -35.8461113, 27.3448830, -53.3721619, 57.0996666
2: -23.0663757, 28.6091671, -30.0297451, 40.6656189, -63.7319946, 58.6389084
3: -25.5954723, 49.8440170, -33.6943398, 65.0930023, -90.6884766, 83.5383453
4: -19.9403934, 28.9177246, -25.7831535, 40.8087845, -60.7491760, 54.7008781

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4972110
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4972110
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -471.1083984, 298.5914001, -475.1539917, 304.6419678, -775.7503662, 773.7451782
1: -35.1998482, 27.1121502, -35.8461113, 27.3448830, -62.5447311, 62.9582558
2: -29.6198521, 40.0117264, -30.0297451, 40.6656189, -70.2854614, 70.0414505
3: -33.3794823, 64.5453186, -33.6943398, 65.0930023, -98.4724884, 98.2396545
4: -25.4255333, 40.0587883, -25.7831535, 40.8087845, -66.2343063, 65.8419418

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.9379697
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.9379697
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -375.1780396, 201.6742096, -443.5166626, 263.2473755, -638.4251709, 645.1908569
1: -25.1377811, 20.9536781, -31.7979584, 25.2185326, -50.3563042, 52.7516365
2: -23.0599976, 27.3185673, -27.6969280, 35.4392509, -58.4992485, 55.0154953
3: -25.4025040, 48.9396896, -30.7875862, 59.3504715, -84.7529755, 79.7272720
4: -19.9510593, 27.6362190, -23.7579861, 35.7358627, -55.6869202, 51.3942032

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.3037242
time: 0.48 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.3037242
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -411.9146423, 241.9918213, -443.5166626, 263.2473755, -675.1619263, 685.5084839
1: -29.0321484, 23.3863163, -31.7979584, 25.2185326, -54.2506790, 55.1842728
2: -25.2123051, 32.3396187, -27.6969280, 35.4392509, -60.6515579, 60.0365448
3: -28.0879021, 54.9645538, -30.7875862, 59.3504715, -87.4383545, 85.7521362
4: -21.6818123, 32.5851517, -23.7579861, 35.7358627, -57.4176750, 56.3431358

Time for backsubstitution: 3.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4716579
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4716579
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -375.1780396, 201.6742096, -478.6080627, 304.4172363, -679.5952759, 680.2822876
1: -25.1377811, 20.9536781, -35.8904762, 27.5209770, -52.6587601, 56.8441544
2: -23.0599976, 27.3185673, -30.0953350, 40.6848412, -63.7448387, 57.4139023
3: -25.4025040, 48.9396896, -33.7341881, 65.3450928, -90.7475967, 82.6738663
4: -19.9510593, 27.6362190, -25.8193207, 40.8501434, -60.8012009, 53.4555397

Time for backsubstitution: 3.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.3037242
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.3037242
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -412.3837891, 242.4407043, -478.6080627, 304.4172363, -716.8010254, 721.0487671
1: -29.0793724, 23.4184208, -35.8904762, 27.5209770, -56.6003494, 59.3088989
2: -25.2454643, 32.4001427, -30.0953350, 40.6848412, -65.9302979, 62.4954758
3: -28.1270676, 55.0434875, -33.7341881, 65.3450928, -93.4721527, 88.7776642
4: -21.7095318, 32.6449890, -25.8193207, 40.8501434, -62.5596695, 58.4643097

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.8062206
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.8062206
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -383.0587158, 215.6151276, -457.9891663, 292.0267029, -675.0854492, 673.6043091
1: -26.3961086, 21.5145588, -34.3598442, 26.3446579, -52.7407684, 55.8744049
2: -23.3302536, 29.0277271, -29.0704346, 38.9475479, -62.2778015, 58.0981598
3: -25.8626728, 50.2921333, -32.7265167, 62.7096558, -88.5723267, 83.0186386
4: -20.1150398, 29.3798809, -24.9501266, 39.0196571, -59.1346970, 54.3300095

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4972110, upper bound: 914.2752993
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4972110, upper bound: 914.2752993
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -422.2101440, 240.2527466, -457.9891663, 292.0267029, -714.2368164, 698.2419434
1: -29.4208698, 23.8113670, -34.3598442, 26.3446579, -55.7655258, 58.1712112
2: -26.2112484, 32.3504066, -29.0704346, 38.9475479, -65.1587830, 61.4208412
3: -29.0218887, 55.7464447, -32.7265167, 62.7096558, -91.7315369, 88.4729538
4: -22.5523129, 32.7047234, -24.9501266, 39.0196571, -61.5719681, 57.6548500

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4972110, upper bound: 914.2752993
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4972110, upper bound: 914.2752993
time: 0.49 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -383.0587158, 215.6151276, -410.5568237, 244.8344574, -627.8931885, 626.1719360
1: -26.3961086, 21.5145588, -29.2897682, 23.3661289, -49.7622261, 50.8043289
2: -23.3302536, 29.0277271, -25.3300896, 32.6569862, -55.9872360, 54.3578186
3: -25.8626728, 50.2921333, -28.2508698, 54.9206467, -80.7833176, 78.5429993
4: -20.1150398, 29.3798809, -21.8066769, 32.9080772, -53.0231094, 51.1865501

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.2752993
time: 0.50 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.2752993
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -422.2101440, 240.2527466, -410.5568237, 244.8344574, -667.0446167, 650.8095703
1: -29.4208698, 23.8113670, -29.2897682, 23.3661289, -52.7869911, 53.1011353
2: -26.2112484, 32.3504066, -25.3300896, 32.6569862, -58.8682175, 57.6804962
3: -29.0218887, 55.7464447, -28.2508698, 54.9206467, -83.9425354, 83.9973145
4: -22.5523129, 32.7047234, -21.8066769, 32.9080772, -55.4603806, 54.5113945

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.2752993
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.2752993
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -479.6513367, 307.1000671, -491.8106995, 316.0624084, -795.7136841, 798.9107056
1: -36.0981674, 27.6595688, -37.0275993, 28.4567299, -64.5549011, 64.6871643
2: -29.9652233, 41.1472321, -30.9230728, 42.3323746, -72.2975998, 72.0703049
3: -33.8450699, 65.8640289, -34.9423218, 67.8971176, -101.7421875, 100.8063431
4: -25.7328892, 41.2166634, -26.5278683, 42.3359070, -68.0687943, 67.7445145

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4972110, upper bound: 914.4859955
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4972110, upper bound: 914.7769341
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -432.9024048, 260.8597107, -491.8106995, 316.0624084, -748.9646606, 752.6704102
1: -31.1656742, 24.6685143, -37.0275993, 28.4567299, -59.6224022, 61.6961136
2: -26.5410519, 34.9062462, -30.9230728, 42.3323746, -68.8734283, 65.8293152
3: -29.6041622, 57.9576492, -34.9423218, 67.8971176, -97.5012817, 92.8999710
4: -22.7638168, 35.1666069, -26.5278683, 42.3359070, -65.0997162, 61.6944733

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4972110, upper bound: 914.4859955
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4972110, upper bound: 914.7769341
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -479.6222229, 307.0757751, -436.9631653, 266.6635742, -746.2857666, 744.0388184
1: -36.0955505, 27.6576786, -31.6598587, 25.0489330, -61.1444855, 59.3175163
2: -29.9633808, 41.1440887, -26.9165382, 35.6312180, -65.5945892, 68.0606232
3: -33.8428192, 65.8592758, -30.0988865, 58.9689484, -92.8117676, 95.9581528
4: -25.7314854, 41.2136574, -23.1032200, 35.8462906, -61.5777702, 64.3168716

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4555440
time: 0.47 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.7769341
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -432.9024048, 260.8597107, -436.9631653, 266.6635742, -699.5659790, 697.8228149
1: -31.1656742, 24.6685143, -31.6598587, 25.0489330, -56.2146072, 56.3283691
2: -26.5410519, 34.9062462, -26.9165382, 35.6312180, -62.1722717, 61.8227844
3: -29.6041622, 57.9576492, -30.0988865, 58.9689484, -88.5731125, 88.0565338
4: -22.7638168, 35.1666069, -23.1032200, 35.8462906, -58.6101036, 58.2698288

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4555440
time: 0.46 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3037242, upper bound: 914.7769341
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -383.0587158, 215.6151276, -468.7427063, 300.9948425, -684.0535889, 684.3578491
1: -26.3961086, 21.5145588, -35.3560333, 27.0256138, -53.4217148, 56.8705902
2: -23.3302536, 29.0277271, -29.4659023, 40.2800636, -63.6103172, 58.4936295
3: -25.8626728, 50.2921333, -33.2280464, 64.4276047, -90.2902756, 83.5201797
4: -20.1150398, 29.3798809, -25.3361225, 40.3726387, -60.4876709, 54.7159996

Time for backsubstitution: 3.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.6936326, upper bound: 914.2752993
time: 0.48 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.6936326, upper bound: 914.2752993
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -422.2101440, 240.2527466, -468.7427063, 300.9948425, -723.2049561, 708.9954834
1: -29.4208698, 23.8113670, -35.3560333, 27.0256138, -56.4464798, 59.1674004
2: -26.2112484, 32.3504066, -29.4659023, 40.2800636, -66.4912872, 61.8163071
3: -29.0218887, 55.7464447, -33.2280464, 64.4276047, -93.4494934, 88.9744873
4: -22.5523129, 32.7047234, -25.3361225, 40.3726387, -62.9249496, 58.0408478

Time for backsubstitution: 3.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.6936326, upper bound: 914.2752993
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.6936326, upper bound: 914.2752993
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -383.0587158, 215.6151276, -435.4045410, 263.7632751, -646.8220215, 651.0196533
1: -26.3961086, 21.5145588, -31.4939156, 24.8579311, -51.2540359, 53.0084763
2: -23.3302536, 29.0277271, -26.8571606, 35.3452148, -58.6754684, 55.8848877
3: -25.8626728, 50.2921333, -29.9770126, 58.4413910, -84.3040619, 80.2691422
4: -20.1150398, 29.3798809, -23.0626011, 35.6212006, -55.7362328, 52.4424782

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.2752993
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.2752993
time: 0.48 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -422.2101440, 240.2527466, -435.4045410, 263.7632751, -685.9733887, 675.6572876
1: -29.4208698, 23.8113670, -31.4939156, 24.8579311, -54.2787971, 55.3052826
2: -26.2112484, 32.3504066, -26.8571606, 35.3452148, -61.5564575, 59.2075653
3: -29.0218887, 55.7464447, -29.9770126, 58.4413910, -87.4632797, 85.7234497
4: -22.5523129, 32.7047234, -23.0626011, 35.6212006, -58.1735039, 55.7673225

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.2752993
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.2752993
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -479.6513367, 307.1000671, -508.1438599, 329.1799011, -808.8312378, 815.2438965
1: -36.0981674, 27.6595688, -38.4071121, 29.5035381, -65.6017075, 66.0666809
2: -29.9652233, 41.1472321, -31.6709480, 44.1785507, -74.1437759, 72.8181763
3: -33.8450699, 65.8640289, -35.8798027, 70.4257889, -104.2708588, 101.7438278
4: -25.7328892, 41.2166634, -27.2797260, 44.1705551, -69.9034424, 68.4963684

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.6959062, upper bound: 914.5111248
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.6959062, upper bound: 914.7769341
time: 0.51 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -432.9024048, 260.8597107, -508.1438599, 329.1799011, -762.0822754, 769.0035400
1: -31.1656742, 24.6685143, -38.4071121, 29.5035381, -60.6692047, 63.0756226
2: -26.5410519, 34.9062462, -31.6709480, 44.1785507, -70.7196045, 66.5771942
3: -29.6041622, 57.9576492, -35.8798027, 70.4257889, -100.0299530, 93.8374481
4: -22.7638168, 35.1666069, -27.2797260, 44.1705551, -66.9343643, 62.4463348

Time for backsubstitution: 3.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.6959062, upper bound: 914.5111248
time: 0.48 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.6959062, upper bound: 914.7769341
time: 0.50 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -479.6222229, 307.0757751, -463.5883789, 288.9784546, -768.6007080, 770.6640625
1: -36.0955505, 27.6576786, -34.2149048, 26.6473064, -62.7428551, 61.8725739
2: -29.9633808, 41.1440887, -28.5121574, 38.7788620, -68.7422256, 69.6562500
3: -33.8428192, 65.8592758, -31.9227734, 62.8190155, -96.6618347, 97.7820282
4: -25.7314854, 41.2136574, -24.4608536, 39.0140572, -64.7455444, 65.6745071

Time for backsubstitution: 3.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4503179
time: 0.48 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.7769341
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -432.9024048, 260.8597107, -463.5883789, 288.9784546, -721.8808594, 724.4481201
1: -31.1656742, 24.6685143, -34.2149048, 26.6473064, -57.8129730, 58.8834190
2: -26.5410519, 34.9062462, -28.5121574, 38.7788620, -65.3199158, 63.4183922
3: -29.6041622, 57.9576492, -31.9227734, 62.8190155, -92.4231720, 89.8804245
4: -22.7638168, 35.1666069, -24.4608536, 39.0140572, -61.7778702, 59.6274529

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4503179
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2752993, upper bound: 914.7769341
time: 0.52 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.41 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4972110
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4972110
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.6355768
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.6355768
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4972110
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4972110
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.9379697
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.9379697
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.3037242
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.3037242
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4776474
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4776474
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.3037242
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.3037242
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.8062206
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.8062206
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4972110
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4972110
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.6298566
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.6298566
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4972110
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4972110
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.9379697
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.9379697
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.3037242
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.3037242
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4716579
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4716579
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.3037242
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.3037242
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.8062206
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.8062206
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.4972110, upper bound: 914.2752993
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.4972110, upper bound: 914.2752993
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.4972110, upper bound: 914.2752993
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.4972110, upper bound: 914.2752993
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.2752993
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.2752993
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.2752993
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.2752993
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.4972110, upper bound: 914.4859955
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.4972110, upper bound: 914.7769341
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.4972110, upper bound: 914.4859955
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.4972110, upper bound: 914.7769341
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4555440
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.7769341
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.4555440
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.3037242, upper bound: 914.7769341
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.6936326, upper bound: 914.2752993
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.6936326, upper bound: 914.2752993
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.6936326, upper bound: 914.2752993
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.6936326, upper bound: 914.2752993
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.2752993
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.2752993
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.2752993
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.2752993
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.6959062, upper bound: 914.5111248
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.6959062, upper bound: 914.7769341
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.6959062, upper bound: 914.5111248
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.6959062, upper bound: 914.7769341
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4503179
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.7769341
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.4503179
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.41
Output dim: 0, lower bound: -914.2752993, upper bound: 914.7769341

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -378.7266846, 212.8682556, -378.7534790, 212.9007263, -591.6273804, 591.6217041
1: -26.0240211, 21.2517738, -26.0272770, 21.2535553, -47.2775764, 47.2790489
2: -23.0641403, 28.6052265, -23.0663757, 28.6091671, -51.6733055, 51.6716003
3: -25.5927620, 49.8398476, -25.5954723, 49.8440170, -75.4367752, 75.4353180
4: -19.9385319, 28.9139767, -19.9403934, 28.9177246, -48.8562546, 48.8543625

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.6783087, upper bound: 914.2423715
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.6783087, upper bound: 914.0444803
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -378.7266846, 212.8682556, -375.1780396, 201.6742096, -580.4008789, 588.0462646
1: -26.0240211, 21.2517738, -25.1377811, 20.9536781, -46.9776993, 46.3895454
2: -23.0641403, 28.6052265, -23.0599976, 27.3185673, -50.3827057, 51.6652222
3: -25.5927620, 49.8398476, -25.4025040, 48.9396896, -74.5324554, 75.2423477
4: -19.9385319, 28.9139767, -19.9510593, 27.6362190, -47.5747528, 48.8650284

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.6783087, upper bound: 914.2423715
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.6783087, upper bound: 914.0444803
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -471.1083984, 298.5914001, -378.7534790, 212.9007263, -684.0090942, 677.3447876
1: -35.1998482, 27.1121502, -26.0272770, 21.2535553, -56.4534035, 53.1394234
2: -29.6198521, 40.0117264, -23.0663757, 28.6091671, -58.2290192, 63.0781021
3: -33.3794823, 64.5453186, -25.5954723, 49.8440170, -83.2234955, 90.1407852
4: -25.4255333, 40.0587883, -19.9403934, 28.9177246, -54.3432503, 59.9991837

Time for backsubstitution: 3.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.6783087, upper bound: 914.1557319
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2448421, upper bound: 914.3891970
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2448421, upper bound: 914.5621222
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -471.1083984, 298.5914001, -375.1780396, 201.6742096, -672.7825928, 673.7693481
1: -35.1998482, 27.1121502, -25.1377811, 20.9536781, -56.1535263, 52.2499237
2: -29.6198521, 40.0117264, -23.0599976, 27.3185673, -56.9384193, 63.0717201
3: -33.3794823, 64.5453186, -25.4025040, 48.9396896, -82.3191681, 89.9478149
4: -25.4255333, 40.0587883, -19.9510593, 27.6362190, -53.0617485, 60.0098495

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.6783087, upper bound: 914.1557319
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2448421, upper bound: 914.3891970
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2448421, upper bound: 914.5621222
time: 0.44 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -378.7534790, 212.9007263, -471.1083984, 298.5914001, -677.3447876, 684.0090942
1: -26.0272770, 21.2535553, -35.1998482, 27.1121502, -53.1394234, 56.4534035
2: -23.0663757, 28.6091671, -29.6198521, 40.0117264, -63.0781021, 58.2290192
3: -25.5954723, 49.8440170, -33.3794823, 64.5453186, -90.1407852, 83.2234955
4: -19.9403934, 28.9177246, -25.4255333, 40.0587883, -59.9991837, 54.3432503

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.8459589, upper bound: 913.4218256
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4010523, upper bound: 914.3821704
time: 0.48 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -378.7534790, 212.9007263, -411.9146423, 241.9918213, -620.7453003, 624.8153687
1: -26.0272770, 21.2535553, -29.0321484, 23.3863163, -49.4135933, 50.2857056
2: -23.0663757, 28.6091671, -25.2123051, 32.3396187, -55.4059944, 53.8214722
3: -25.5954723, 49.8440170, -28.0879021, 54.9645538, -80.5600281, 77.9319000
4: -19.9403934, 28.9177246, -21.6818123, 32.5851517, -52.5255394, 50.5995369

Time for backsubstitution: 3.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.8459589, upper bound: 913.4218256
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4010523, upper bound: 914.3821704
time: 0.47 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -471.1083984, 298.5914001, -471.1083984, 298.5914001, -769.6997070, 769.6997070
1: -35.1998482, 27.1121502, -35.1998482, 27.1121502, -62.3119926, 62.3119965
2: -29.6198521, 40.0117264, -29.6198521, 40.0117264, -69.6315689, 69.6315689
3: -33.3794823, 64.5453186, -33.3794823, 64.5453186, -97.9248047, 97.9248047
4: -25.4255333, 40.0587883, -25.4255333, 40.0587883, -65.4843216, 65.4843216

Time for backsubstitution: 3.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4998516, upper bound: 914.7747849
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4998516, upper bound: 914.6138313
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -471.1083984, 298.5914001, -411.9146423, 241.9918213, -713.1002197, 710.5060425
1: -35.1998482, 27.1121502, -29.0321484, 23.3863163, -58.5861664, 56.1442986
2: -29.6198521, 40.0117264, -25.2123051, 32.3396187, -61.9594612, 65.2240295
3: -33.3794823, 64.5453186, -28.0879021, 54.9645538, -88.3440399, 92.6331940
4: -25.4255333, 40.0587883, -21.6818123, 32.5851517, -58.0106735, 61.7406006

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4998516, upper bound: 914.8773406
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4998516, upper bound: 914.6138313
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -375.1780396, 201.6742096, -378.7534790, 212.9007263, -588.0787354, 580.4276733
1: -25.1377811, 20.9536781, -26.0272770, 21.2535553, -46.3913345, 46.9809570
2: -23.0599976, 27.3185673, -23.0663757, 28.6091671, -51.6691628, 50.3849411
3: -25.4025040, 48.9396896, -25.5954723, 49.8440170, -75.2465210, 74.5351562
4: -19.9510593, 27.6362190, -19.9403934, 28.9177246, -48.8687820, 47.5766144

Time for backsubstitution: 3.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2743651, upper bound: 913.6930440
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2743651, upper bound: 914.2743649
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -375.1780396, 201.6742096, -375.1780396, 201.6742096, -576.8522339, 576.8522339
1: -25.1377811, 20.9536781, -25.1377811, 20.9536781, -46.0914574, 46.0914574
2: -23.0599976, 27.3185673, -23.0599976, 27.3185673, -50.3785629, 50.3785629
3: -25.4025040, 48.9396896, -25.4025040, 48.9396896, -74.3421860, 74.3421936
4: -19.9510593, 27.6362190, -19.9510593, 27.6362190, -47.5872803, 47.5872803

Time for backsubstitution: 3.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2743651, upper bound: 913.6930440
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2743651, upper bound: 914.2743649
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -411.9146423, 241.9918213, -378.7534790, 212.9007263, -624.8153687, 620.7453003
1: -29.0321484, 23.3863163, -26.0272770, 21.2535553, -50.2857056, 49.4135933
2: -25.2123051, 32.3396187, -23.0663757, 28.6091671, -53.8214722, 55.4059944
3: -28.0879021, 54.9645538, -25.5954723, 49.8440170, -77.9319077, 80.5600281
4: -21.6818123, 32.5851517, -19.9403934, 28.9177246, -50.5995369, 52.5255394

Time for backsubstitution: 3.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.6783087, upper bound: 914.0526812
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2743651, upper bound: 913.5841019
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2743651, upper bound: 914.4499498
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -411.9146423, 241.9918213, -375.1780396, 201.6742096, -613.5888672, 617.1698608
1: -29.0321484, 23.3863163, -25.1377811, 20.9536781, -49.9858246, 48.5240974
2: -25.2123051, 32.3396187, -23.0599976, 27.3185673, -52.5308723, 55.3996162
3: -28.0879021, 54.9645538, -25.4025040, 48.9396896, -77.0275879, 80.3670578
4: -21.6818123, 32.5851517, -19.9510593, 27.6362190, -49.3180313, 52.5362091

Time for backsubstitution: 3.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.6783087, upper bound: 914.0526812
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2743651, upper bound: 913.6876280
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2743651, upper bound: 914.4499498
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -375.1780396, 201.6742096, -471.1083984, 298.5914001, -673.7693481, 672.7825928
1: -25.1377811, 20.9536781, -35.1998482, 27.1121502, -52.2499237, 56.1535263
2: -23.0599976, 27.3185673, -29.6198521, 40.0117264, -63.0717201, 56.9384193
3: -25.4025040, 48.9396896, -33.3794823, 64.5453186, -89.9478149, 82.3191681
4: -19.9510593, 27.6362190, -25.4255333, 40.0587883, -60.0098495, 53.0617485

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3268633, upper bound: 913.6930440
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4499500, upper bound: 914.2743649
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -375.1780396, 201.6742096, -412.3837891, 242.4407043, -617.6187744, 614.0579834
1: -25.1377811, 20.9536781, -29.0793724, 23.4184208, -48.5561981, 50.0330505
2: -23.0599976, 27.3185673, -25.2454643, 32.4001427, -55.4601402, 52.5640335
3: -25.4025040, 48.9396896, -28.1270676, 55.0434875, -80.4459915, 77.0667419
4: -19.9510593, 27.6362190, -21.7095318, 32.6449890, -52.5960464, 49.3457489

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.3268633, upper bound: 913.6930440
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.4499500, upper bound: 914.2743649
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -412.3837891, 242.4407043, -471.1083984, 298.5914001, -710.9751587, 713.5490723
1: -29.0793724, 23.4184208, -35.1998482, 27.1121502, -56.1915207, 58.6182709
2: -25.2454643, 32.4001427, -29.6198521, 40.0117264, -65.2571716, 62.0199890
3: -28.1270676, 55.0434875, -33.3794823, 64.5453186, -92.6723633, 88.4229736
4: -21.7095318, 32.6449890, -25.4255333, 40.0587883, -61.7683182, 58.0705109

Time for backsubstitution: 3.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.6074731, upper bound: 913.5811968
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.7916953, upper bound: 914.7916949
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -412.3837891, 242.4407043, -412.3837891, 242.4407043, -654.8244629, 654.8244629
1: -29.0793724, 23.4184208, -29.0793724, 23.4184208, -52.4977951, 52.4977951
2: -25.2454643, 32.4001427, -25.2454643, 32.4001427, -57.6456070, 57.6456070
3: -28.1270676, 55.0434875, -28.1270676, 55.0434875, -83.1705399, 83.1705399
4: -21.7095318, 32.6449890, -21.7095318, 32.6449890, -54.3545151, 54.3545151

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.8062206, upper bound: 914.7988905
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.8062206, upper bound: 914.8062206
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -378.7266846, 212.8682556, -383.0587158, 215.6151276, -594.3417969, 595.9270020
1: -26.0240211, 21.2517738, -26.3961086, 21.5145588, -47.5385742, 47.6478729
2: -23.0641403, 28.6052265, -23.3302536, 29.0277271, -52.0918655, 51.9354782
3: -25.5927620, 49.8398476, -25.8626728, 50.2921333, -75.8848877, 75.7025146
4: -19.9385319, 28.9139767, -20.1150398, 29.3798809, -49.3184128, 49.0290070

Time for backsubstitution: 3.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.8493072, upper bound: 913.4218256
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1664939, upper bound: 914.3821704
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -378.7266846, 212.8682556, -422.2101440, 240.2527466, -618.9794312, 635.0783691
1: -26.0240211, 21.2517738, -29.4208698, 23.8113670, -49.8353844, 50.6726379
2: -23.0641403, 28.6052265, -26.2112484, 32.3504066, -55.4145470, 54.8164711
3: -25.5927620, 49.8398476, -29.0218887, 55.7464447, -81.3391876, 78.8617325
4: -19.9385319, 28.9139767, -22.5523129, 32.7047234, -52.6432571, 51.4662819

Time for backsubstitution: 3.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -913.8493072, upper bound: 913.4218256
time: 0.47 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.1664939, upper bound: 914.3821704
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -471.1083984, 298.5914001, -383.0587158, 215.6151276, -686.7234497, 681.6500244
1: -35.1998482, 27.1121502, -26.3961086, 21.5145588, -56.7144089, 53.5082474
2: -29.6198521, 40.0117264, -23.3302536, 29.0277271, -58.6475792, 63.3419800
3: -33.3794823, 64.5453186, -25.8626728, 50.2921333, -83.6716156, 90.4079819
4: -25.4255333, 40.0587883, -20.1150398, 29.3798809, -54.8054085, 60.1738281

Time for backsubstitution: 3.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2098967, upper bound: 914.3659256
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -914.2098967, upper bound: 914.5566231
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -471.1083984, 298.5914001, -422.2101440, 240.2527466, -711.3611450, 720.8014526
1: -35.1998482, 27.1121502, -29.4208698, 23.8113670, -59.0112152, 56.5330162
2: -29.6198521, 40.0117264, -26.2112484, 32.3504066, -61.9702568, 66.2229538
3: -33.3794823, 64.5453186, -29.0218887, 55.7464447, -89.1259308, 93.5671921
4: -25.4255333, 40.0587883, -22.5523129, 32.7047234, -58.1302528, 62.6110992

Time for backsubstitution: 3.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.65 + 415.75 = 420.40 seconds
