## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 7)
Time budget: 1800 seconds
Split limit: 100
Threshold: 5.9257269272


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=57, inp2_unstable=57, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=129, inp2_unstable=129, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=7, inp2_unstable=7, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-22.5208092, -7.9731917, -22.5208092, -7.9731917, -8.7436066, 8.7436066)
1: (-9.2077713, -1.1274239, -9.2077713, -1.1274239, -5.6194916, 5.6194916)
2: (-10.1279774, -1.6378818, -10.1279774, -1.6378818, -5.1707211, 5.1707211)
3: (-16.8262272, -5.7670259, -16.8262272, -5.7670259, -8.2020531, 8.2020531)
4: (-13.9884014, -3.8142982, -13.9884014, -3.8142982, -7.3747368, 7.3747368)
5: (-13.8391609, -1.8440387, -13.8391609, -1.8440387, -8.0852203, 8.0852203)
6: (-14.1477938, -2.8791432, -14.1477938, -2.8791432, -7.6299133, 7.6299114)
7: (-16.0471992, -3.6275334, -16.0471992, -3.6275334, -8.2773857, 8.2773857)
8: (-22.6316147, -8.7984657, -22.6316147, -8.7984657, -10.8417549, 10.8417511)
9: (-13.1974850, -2.8590405, -13.1974850, -2.8590405, -8.1662102, 8.1662102)
10: (-9.5338402, 1.3198309, -9.5338402, 1.3198309, -10.4590797, 10.4590797)
11: (3.7692699, 10.7754021, 3.7692699, 10.7754021, -5.1546040, 5.1546040)
12: (-4.6074867, 13.1440716, -4.6074867, 13.1440716, -13.8459702, 13.8459702)
13: (-17.4544086, -1.6790624, -17.4544086, -1.6790624, -15.4921188, 15.4921188)
14: (-17.9668274, 0.6517401, -17.9668274, 0.6517401, -18.0359344, 18.0359497)
15: (-11.7280416, -3.3923748, -11.7280416, -3.3923748, -7.5929909, 7.5929909)
16: (-10.6599693, -1.5840062, -10.6599693, -1.5840062, -8.6743279, 8.6743279)
17: (-10.7281418, 4.8268633, -10.7281418, 4.8268633, -15.5069122, 15.5069122)
18: (4.1554890, 16.7664967, 4.1554890, 16.7664967, -11.8557358, 11.8557320)
19: (3.1253505, 8.5648842, 3.1253505, 8.5648842, -4.9915123, 4.9915142)
20: (-1.8722749, 5.9504194, -1.8722749, 5.9504194, -7.0276489, 7.0276489)
21: (4.9270263, 12.4471560, 4.9270263, 12.4471560, -7.1925087, 7.1925087)
22: (2.9299645, 11.2143536, 2.9299645, 11.2143536, -6.4172363, 6.4172363)
23: (3.3086083, 9.7840385, 3.3086083, 9.7840385, -4.4192467, 4.4192467)
24: (2.9455552, 11.5603294, 2.9455552, 11.5603294, -6.8244553, 6.8244553)
25: (3.5490971, 12.6621485, 3.5490971, 12.6621485, -7.1003723, 7.1003723)
26: (2.1465859, 15.6662560, 2.1465859, 15.6662560, -13.2550735, 13.2550735)
27: (-0.6436005, 9.8099976, -0.6436005, 9.8099976, -9.2262688, 9.2262688)
28: (1.6828536, 9.6099014, 1.6828536, 9.6099014, -6.0401707, 6.0401726)
29: (4.6526537, 11.1147413, 4.6526537, 11.1147413, -4.9413567, 4.9413567)
30: (0.8888544, 10.2611542, 0.8888544, 10.2611542, -8.6654205, 8.6654205)
31: (4.2493477, 12.6472197, 4.2493477, 12.6472197, -6.9564056, 6.9564056)
32: (-15.5157452, -4.1003132, -15.5157452, -4.1003132, -8.4422913, 8.4422913)
33: (-27.0337715, -9.0484381, -27.0337715, -9.0484381, -13.2708969, 13.2708969)
34: (-25.5029335, -10.6915417, -25.5029335, -10.6915417, -10.5571251, 10.5571251)
35: (-14.8872604, -0.5253859, -14.8872604, -0.5253859, -12.8504333, 12.8504333)
36: (-13.3170547, 2.0410190, -13.3170547, 2.0410190, -14.7596359, 14.7596359)
37: (-24.9147606, -7.8220968, -24.9147606, -7.8220968, -14.0781555, 14.0781555)
38: (-18.3238468, -0.8664770, -18.3238468, -0.8664770, -17.4573708, 17.4573708)
39: (-28.8319817, -9.8166685, -28.8319817, -9.8166685, -16.3936081, 16.3936081)
40: (-30.2451859, -17.9318886, -30.2451859, -17.9318886, -7.8182373, 7.8182373)
41: (-16.2457924, -2.8425083, -16.2457924, -2.8425083, -10.2350540, 10.2350578)
42: (-16.2273617, -7.6026998, -16.2273617, -7.6026998, -6.5956554, 6.5956535)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.41 + 27.83 = 30.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 18, lower bound: -5.9435576, upper bound: 5.9435576

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1699
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1591

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.9389132, upper bound: 5.9222439
time: 19.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.9303644, upper bound: 5.9303639
time: 32.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 52.96 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 52.96
Output dim: 18, lower bound: -5.9389132, upper bound: 5.9222439
IS_A2, status: Status.UNKNOWN, split count: 1, time: 52.96
Output dim: 18, lower bound: -5.9303644, upper bound: 5.9303639

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -22.5199165, -7.9740391, -22.5208092, -7.9731917, -8.7429161, 8.7413940
1: -9.2072859, -1.1279429, -9.2077713, -1.1274239, -5.6190643, 5.6184692
2: -10.1272602, -1.6384108, -10.1279774, -1.6378818, -5.1702061, 5.1697788
3: -16.8209839, -5.7676315, -16.8262272, -5.7670259, -8.1968498, 8.2015266
4: -13.9876080, -3.8166299, -13.9884014, -3.8142982, -7.3741913, 7.3715401
5: -13.8364010, -1.8448181, -13.8391609, -1.8440387, -8.0826797, 8.0842705
6: -14.1432610, -2.8801265, -14.1477938, -2.8791432, -7.6253548, 7.6289425
7: -16.0450058, -3.6279907, -16.0471992, -3.6275334, -8.2757874, 8.2767906
8: -22.6309910, -8.8002262, -22.6316147, -8.7984657, -10.8410072, 10.8379936
9: -13.1956587, -2.8597176, -13.1974850, -2.8590405, -8.1647110, 8.1648369
10: -9.5322151, 1.3191357, -9.5338402, 1.3198309, -10.4577827, 10.4571075
11: 3.7722135, 10.7750931, 3.7692699, 10.7754021, -5.1516171, 5.1543427
12: -4.6045947, 13.1429806, -4.6074867, 13.1440716, -13.8429565, 13.8448944
13: -17.4464836, -1.6799586, -17.4544086, -1.6790624, -15.4836960, 15.4911575
14: -17.9653149, 0.6483603, -17.9668274, 0.6517401, -18.0341339, 18.0301437
15: -11.7271671, -3.3967097, -11.7280416, -3.3923748, -7.5921211, 7.5887909
16: -10.6585903, -1.5846581, -10.6599693, -1.5840062, -8.6736488, 8.6721382
17: -10.7268057, 4.8246655, -10.7281418, 4.8268633, -15.5055389, 15.5045013
18: 4.1565599, 16.7589874, 4.1554890, 16.7664967, -11.8548050, 11.8484688
19: 3.1258628, 8.5620241, 3.1253505, 8.5648842, -4.9911041, 4.9886246
20: -1.8713927, 5.9479833, -1.8722749, 5.9504194, -7.0268631, 7.0248375
21: 4.9278922, 12.4449825, 4.9270263, 12.4471560, -7.1917915, 7.1898155
22: 2.9307618, 11.2083569, 2.9299645, 11.2143536, -6.4163589, 6.4118347
23: 3.3090260, 9.7815571, 3.3086083, 9.7840385, -4.4188614, 4.4167290
24: 2.9464738, 11.5560694, 2.9455552, 11.5603294, -6.8236656, 6.8201675
25: 3.5498390, 12.6578188, 3.5490971, 12.6621485, -7.0995255, 7.0965271
26: 2.1478949, 15.6590614, 2.1465859, 15.6662560, -13.2538147, 13.2472916
27: -0.6425598, 9.8040314, -0.6436005, 9.8099976, -9.2253151, 9.2197380
28: 1.6834799, 9.6069460, 1.6828536, 9.6099014, -6.0396633, 6.0372963
29: 4.6531539, 11.1115751, 4.6526537, 11.1147413, -4.9408188, 4.9379845
30: 0.8896399, 10.2604313, 0.8888544, 10.2611542, -8.6648254, 8.6643524
31: 4.2500563, 12.6431885, 4.2493477, 12.6472197, -6.9557419, 6.9526253
32: -15.5103951, -4.1010437, -15.5157452, -4.1003132, -8.4373283, 8.4416580
33: -27.0287323, -9.0490046, -27.0337715, -9.0484381, -13.2658157, 13.2704391
34: -25.4968033, -10.6918907, -25.5029335, -10.6915417, -10.5505638, 10.5568047
35: -14.8814945, -0.5257993, -14.8872604, -0.5253859, -12.8445816, 12.8500290
36: -13.3134747, 2.0408678, -13.3170547, 2.0410190, -14.7559814, 14.7594757
37: -24.9127541, -7.8224382, -24.9147606, -7.8220968, -14.0762863, 14.0769806
38: -18.3198891, -0.8667474, -18.3238468, -0.8664770, -17.4534111, 17.4570999
39: -28.8263893, -9.8169765, -28.8319817, -9.8166685, -16.3883057, 16.3933563
40: -30.2420540, -17.9320259, -30.2451859, -17.9318886, -7.8173828, 7.8167782
41: -16.2415771, -2.8430882, -16.2457924, -2.8425083, -10.2310257, 10.2345543
42: -16.2223835, -7.6035471, -16.2273617, -7.6026998, -6.5908375, 6.5948277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=56, inp2_unstable=57, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=129, inp2_unstable=129, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=7, inp2_unstable=7, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -5.9222439, upper bound: 5.9222439
time: 10.27 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 18, lower bound: -5.9222439, upper bound: 5.9222439
time: 9.05 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -22.5268745, -7.9654646, -22.5202808, -7.9734049, -8.7559624, 8.7476196
1: -9.2067947, -1.1171346, -9.2074442, -1.1275949, -5.6207199, 5.6272755
2: -10.1295547, -1.6271353, -10.1275339, -1.6380386, -5.1758480, 5.1773453
3: -16.8277493, -5.7175598, -16.8252773, -5.7673297, -8.2069931, 8.2499428
4: -14.0003872, -3.8095551, -13.9881401, -3.8154159, -7.3907166, 7.3795204
5: -13.8395958, -1.8173850, -13.8385715, -1.8442719, -8.0869293, 8.1095428
6: -14.1488657, -2.8412051, -14.1467800, -2.8794065, -7.6343880, 7.6673889
7: -16.0448761, -3.6041656, -16.0458965, -3.6276946, -8.2806015, 8.2939529
8: -22.6382141, -8.7888041, -22.6314812, -8.7991848, -10.8674164, 10.8438568
9: -13.1979380, -2.8437803, -13.1968775, -2.8593807, -8.1727829, 8.1758842
10: -9.5411301, 1.3376417, -9.5329876, 1.3195181, -10.4797707, 10.4667397
11: 3.7621527, 10.7899208, 3.7706094, 10.7753057, -5.1619301, 5.1699486
12: -4.6115727, 13.1585026, -4.6065898, 13.1438322, -13.8516617, 13.8615799
13: -17.4567032, -1.6012268, -17.4532299, -1.6794147, -15.4953156, 15.5689468
14: -18.0091972, 0.6561069, -17.9663506, 0.6506681, -18.0963898, 18.0349960
15: -11.7569065, -3.3862765, -11.7276897, -3.3929639, -7.6216049, 7.5987473
16: -10.6594124, -1.5575485, -10.6587486, -1.5842260, -8.6975517, 8.6793327
17: -10.7467995, 4.8291235, -10.7276993, 4.8256946, -15.5243683, 15.5119095
18: 4.0729914, 16.7684116, 4.1559887, 16.7652359, -11.9360428, 11.8603325
19: 3.0977583, 8.5653105, 3.1255691, 8.5644951, -5.0208397, 4.9924583
20: -1.8961062, 5.9505124, -1.8719893, 5.9496102, -7.0529747, 7.0293427
21: 4.9040318, 12.4490204, 4.9273257, 12.4468145, -7.2178421, 7.1940613
22: 2.8834295, 11.2143459, 2.9302056, 11.2137680, -6.4638901, 6.4223709
23: 3.2719145, 9.7873516, 3.3087540, 9.7836723, -4.4556732, 4.4232674
24: 2.8953519, 11.5604019, 2.9458590, 11.5599108, -6.8739891, 6.8270912
25: 3.5091062, 12.6634865, 3.5493383, 12.6613131, -7.1372299, 7.1055527
26: 2.0677514, 15.6690350, 2.1470141, 15.6653042, -13.3331528, 13.2609329
27: -0.7043982, 9.8126965, -0.6432782, 9.8091345, -9.2877388, 9.2320290
28: 1.6468027, 9.6134224, 1.6830873, 9.6094475, -6.0761814, 6.0440636
29: 4.6273265, 11.1158314, 4.6528587, 11.1142359, -4.9662704, 4.9464817
30: 0.8639232, 10.2674351, 0.8891234, 10.2607441, -8.6896896, 8.6720963
31: 4.2141662, 12.6468344, 4.2495289, 12.6465693, -6.9912910, 6.9567451
32: -15.5181408, -4.0535755, -15.5145235, -4.1005745, -8.4483719, 8.4894218
33: -27.0460434, -9.0074959, -27.0329819, -9.0486031, -13.2840881, 13.3130417
34: -25.5076981, -10.6520948, -25.5020103, -10.6916714, -10.5666542, 10.5971146
35: -14.8916702, -0.4806510, -14.8862572, -0.5254692, -12.8585510, 12.8948212
36: -13.3187752, 2.0673985, -13.3162136, 2.0409684, -14.7607880, 14.7848129
37: -24.9228783, -7.8147326, -24.9139557, -7.8222885, -14.0924759, 14.0824432
38: -18.3306236, -0.8443804, -18.3230839, -0.8665605, -17.4640636, 17.4787025
39: -28.8377743, -9.7695675, -28.8311462, -9.8167534, -16.4000931, 16.4399796
40: -30.2499123, -17.9078426, -30.2436256, -17.9319267, -7.8503571, 7.8258381
41: -16.2456970, -2.8143737, -16.2449512, -2.8426688, -10.2364502, 10.2626991
42: -16.2308292, -7.5637364, -16.2265453, -7.6030731, -6.5992355, 6.6349564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=56, inp2_unstable=57, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=129, inp2_unstable=129, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=7, inp2_unstable=7, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1699
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 1680

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1699

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.9275032, upper bound: 5.9291270
time: 9.73 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.9291220, upper bound: 5.9291215
time: 20.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 32.33 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 32.33
Output dim: 18, lower bound: -5.9222439, upper bound: 5.9222439
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 32.33
Output dim: 18, lower bound: -5.9222439, upper bound: 5.9222439
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 32.33
Output dim: 18, lower bound: -5.9275032, upper bound: 5.9291270
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 32.33
Output dim: 18, lower bound: -5.9291220, upper bound: 5.9291215

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -22.5267334, -7.9678330, -22.5191288, -7.9810739, -8.7488174, 8.7454987
1: -9.2066536, -1.1194699, -9.2073746, -1.1353402, -5.6125927, 5.6238098
2: -10.1295338, -1.6282265, -10.1279736, -1.6422782, -5.1715336, 5.1756191
3: -16.8275642, -5.7199211, -16.8230324, -5.7750063, -8.1985931, 8.2444305
4: -14.0001268, -3.8110051, -13.9875641, -3.8205042, -7.3859520, 7.3731880
5: -13.8395329, -1.8213754, -13.8370638, -1.8561702, -8.0751076, 8.1038895
6: -14.1477604, -2.8415411, -14.1426668, -2.8816733, -7.6293983, 7.6628685
7: -16.0446701, -3.6070886, -16.0444927, -3.6372480, -8.2701416, 8.2868767
8: -22.6379395, -8.7897825, -22.6312599, -8.8040047, -10.8610687, 10.8369789
9: -13.1977701, -2.8486683, -13.1930189, -2.8740413, -8.1576080, 8.1669960
10: -9.5408859, 1.3331652, -9.5284386, 1.3052468, -10.4645042, 10.4553223
11: 3.7634118, 10.7889595, 3.7771003, 10.7719488, -5.1576958, 5.1629467
12: -4.6056061, 13.1578846, -4.5884638, 13.1348114, -13.8338318, 13.8405914
13: -17.4492302, -1.6023626, -17.4307632, -1.6930759, -15.4738998, 15.5454178
14: -18.0078468, 0.6553535, -17.9577179, 0.6470671, -18.0868378, 18.0292816
15: -11.7555904, -3.3880653, -11.7220755, -3.4005382, -7.6140709, 7.5916939
16: -10.6589785, -1.5630350, -10.6525230, -1.6010007, -8.6793823, 8.6663742
17: -10.7392988, 4.8290510, -10.7037296, 4.8169184, -15.5075531, 15.4877090
18: 4.0737925, 16.7669735, 4.1656075, 16.7606544, -11.9315796, 11.8507614
19: 3.0993152, 8.5628748, 3.1329691, 8.5572777, -5.0117683, 4.9812279
20: -1.8939428, 5.9467320, -1.8598392, 5.9379449, -7.0401154, 7.0139523
21: 4.9060407, 12.4433498, 4.9413781, 12.4299040, -7.2002068, 7.1755142
22: 2.8851585, 11.2142220, 2.9370341, 11.2132406, -6.4605331, 6.4147263
23: 3.2738268, 9.7842026, 3.3191471, 9.7738400, -4.4439487, 4.4091854
24: 2.8969665, 11.5597315, 2.9524465, 11.5578623, -6.8697433, 6.8179665
25: 3.5113029, 12.6615171, 3.5606189, 12.6551771, -7.1310768, 7.0942268
26: 2.0707769, 15.6689701, 2.1591110, 15.6649065, -13.3294525, 13.2480164
27: -0.7037389, 9.8113422, -0.6382293, 9.8048449, -9.2813568, 9.2234154
28: 1.6486859, 9.6100578, 1.6948433, 9.5987377, -6.0639839, 6.0288219
29: 4.6285071, 11.1155949, 4.6572394, 11.1133842, -4.9617023, 4.9429741
30: 0.8657357, 10.2642488, 0.9033936, 10.2497587, -8.6778946, 8.6564522
31: 4.2159972, 12.6437359, 4.2588038, 12.6373701, -6.9802094, 6.9440575
32: -15.5153618, -4.0543003, -15.5062923, -4.1094136, -8.4369431, 8.4824905
33: -27.0372944, -9.0077362, -27.0067558, -9.0562944, -13.2665558, 13.2865067
34: -25.5040455, -10.6523943, -25.4904575, -10.6953163, -10.5585480, 10.5855331
35: -14.8858280, -0.4808428, -14.8691006, -0.5317571, -12.8460464, 12.8763885
36: -13.3107138, 2.0671754, -13.2924337, 2.0323641, -14.7439575, 14.7608414
37: -24.9126930, -7.8148603, -24.8824749, -7.8319511, -14.0727386, 14.0512466
38: -18.3229637, -0.8447332, -18.2994709, -0.8760343, -17.4469299, 17.4547386
39: -28.8257866, -9.7697086, -28.7961540, -9.8298454, -16.3751602, 16.4050140
40: -30.2438469, -17.9080315, -30.2242699, -17.9403191, -7.8343811, 7.8070469
41: -16.2416725, -2.8150771, -16.2332516, -2.8508430, -10.2243423, 10.2509499
42: -16.2279301, -7.5642490, -16.2184200, -7.6089058, -6.5935402, 6.6301346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=56, inp2_unstable=56, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=129, inp2_unstable=129, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=7, inp2_unstable=7, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 1680

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -5.9243965, upper bound: 5.9254892
time: 19.82 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.9243965, upper bound: 5.9270106
time: 23.27 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -22.5268688, -7.9655766, -22.5202789, -7.9744458, -8.7581444, 8.7471123
1: -9.2067804, -1.1172276, -9.2073917, -1.1284258, -5.6192474, 5.6271420
2: -10.1295366, -1.6271901, -10.1275024, -1.6383584, -5.1755142, 5.1774940
3: -16.8277397, -5.7176199, -16.8252678, -5.7677059, -8.2058258, 8.2499046
4: -14.0003662, -3.8095698, -13.9879837, -3.8155394, -7.3899040, 7.3839912
5: -13.8395824, -1.8174677, -13.8384838, -1.8450146, -8.0818329, 8.1093979
6: -14.1488123, -2.8412030, -14.1463137, -2.8794565, -7.6342812, 7.6634865
7: -16.0448780, -3.6042824, -16.0457859, -3.6285410, -8.2795181, 8.2937889
8: -22.6382008, -8.7888641, -22.6314335, -8.7997561, -10.8664093, 10.8482475
9: -13.1979265, -2.8438396, -13.1968403, -2.8598740, -8.1716309, 8.1758041
10: -9.5411301, 1.3375726, -9.5329666, 1.3190088, -10.4790649, 10.4690781
11: 3.7621689, 10.7898827, 3.7707415, 10.7749996, -5.1613483, 5.1698227
12: -4.6114182, 13.1585102, -4.6056981, 13.1438246, -13.8516083, 13.8549271
13: -17.4565983, -1.6012640, -17.4521027, -1.6796150, -15.4955215, 15.5677185
14: -18.0091476, 0.6561174, -17.9661751, 0.6505318, -18.1064148, 18.0336456
15: -11.7568617, -3.3863103, -11.7272282, -3.3931992, -7.6213226, 7.5979538
16: -10.6594105, -1.5576791, -10.6587296, -1.5854187, -8.6970444, 8.6793251
17: -10.7467117, 4.8291225, -10.7269344, 4.8256483, -15.5242538, 15.5085373
18: 4.0730000, 16.7683334, 4.1561794, 16.7644444, -11.9336548, 11.8600578
19: 3.0977895, 8.5652723, 3.1257749, 8.5642366, -5.0167618, 4.9922218
20: -1.8960707, 5.9504728, -1.8717711, 5.9492822, -7.0466728, 7.0290985
21: 4.9040761, 12.4489584, 4.9276218, 12.4462833, -7.2094193, 7.1937943
22: 2.8834658, 11.2143488, 2.9304085, 11.2136745, -6.4637260, 6.4215736
23: 3.2719409, 9.7873135, 3.3089175, 9.7833109, -4.4477406, 4.4231091
24: 2.8953862, 11.5603580, 2.9460278, 11.5594406, -6.8737907, 6.8269653
25: 3.5091219, 12.6634693, 3.5495496, 12.6610441, -7.1341209, 7.1051502
26: 2.0677872, 15.6690254, 2.1473265, 15.6652946, -13.3330460, 13.2602234
27: -0.7043893, 9.8125973, -0.6431235, 9.8081770, -9.2871933, 9.2323494
28: 1.6468155, 9.6133776, 1.6832469, 9.6090775, -6.0686035, 6.0438595
29: 4.6273460, 11.1158142, 4.6530018, 11.1141787, -4.9707375, 4.9457035
30: 0.8639418, 10.2674103, 0.8893090, 10.2605419, -8.6873856, 8.6719780
31: 4.2141838, 12.6467857, 4.2497110, 12.6461744, -6.9863815, 6.9565125
32: -15.5181217, -4.0535836, -15.5143509, -4.1006212, -8.4482422, 8.4837646
33: -27.0459442, -9.0075006, -27.0324631, -9.0486403, -13.2840576, 13.2984467
34: -25.5075760, -10.6520510, -25.5010147, -10.6916866, -10.5665054, 10.5895424
35: -14.8916035, -0.4806530, -14.8856506, -0.5255105, -12.8584518, 12.8916168
36: -13.3186588, 2.0673823, -13.3153706, 2.0409398, -14.7606888, 14.7828903
37: -24.9227924, -7.8147326, -24.9129925, -7.8222857, -14.0924149, 14.0711365
38: -18.3305378, -0.8443885, -18.3223152, -0.8665662, -17.4639721, 17.4779263
39: -28.8377151, -9.7695560, -28.8310089, -9.8167553, -16.4000549, 16.4213943
40: -30.2498512, -17.9078407, -30.2431908, -17.9319687, -7.8500519, 7.8088512
41: -16.2456741, -2.8143959, -16.2447224, -2.8427639, -10.2363892, 10.2535095
42: -16.2307854, -7.5637655, -16.2260857, -7.6032376, -6.5988922, 6.6346817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=56, inp2_unstable=56, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=129, inp2_unstable=129, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=7, inp2_unstable=7, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1687
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1705
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1687

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.9260170, upper bound: 5.9254840
time: 13.91 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.9260170, upper bound: 5.9254840
time: 9.11 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.02 seconds
IS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 25.02
Output dim: 18, lower bound: -5.9243965, upper bound: 5.9254892
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 25.02
Output dim: 18, lower bound: -5.9243965, upper bound: 5.9270106
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 25.02
Output dim: 18, lower bound: -5.9260170, upper bound: 5.9254840
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 25.02
Output dim: 18, lower bound: -5.9260170, upper bound: 5.9254840

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -22.5255013, -7.9679594, -22.5199814, -7.9722681, -8.7582893, 8.7477951
1: -9.2065163, -1.1205947, -9.2090244, -1.1340795, -5.6231632, 5.6117268
2: -10.1290779, -1.6284313, -10.1300611, -1.6248882, -5.1880856, 5.1762238
3: -16.8264446, -5.7201266, -16.8212509, -5.7676811, -8.2091293, 8.2377739
4: -13.9980335, -3.8111348, -13.9897089, -3.8075495, -7.4069290, 7.3687515
5: -13.8378735, -1.8215773, -13.8349533, -1.8426783, -8.0931549, 8.1041107
6: -14.1475143, -2.8421445, -14.1438580, -2.8778734, -7.6353035, 7.6581268
7: -16.0438786, -3.6073756, -16.0452938, -3.6313891, -8.2837334, 8.2756920
8: -22.6372910, -8.7901344, -22.6355038, -8.7968407, -10.8745346, 10.8291893
9: -13.1976452, -2.8494329, -13.2113571, -2.8714545, -8.1562462, 8.1848602
10: -9.5406981, 1.3317723, -9.5620918, 1.3074298, -10.4624977, 10.4880981
11: 3.7636616, 10.7877998, 3.7570300, 10.7712307, -5.1507816, 5.1801014
12: -4.6052504, 13.1554518, -4.6559620, 13.1363697, -13.8267517, 13.9054031
13: -17.4488869, -1.6047759, -17.4567451, -1.6888740, -15.4740295, 15.5624542
14: -18.0072823, 0.6530094, -18.0401211, 0.6443815, -18.0783005, 18.1109467
15: -11.7542658, -3.3883660, -11.7261353, -3.3898458, -7.6165848, 7.5953102
16: -10.6586132, -1.5639480, -10.6535053, -1.5982116, -8.6982651, 8.6506233
17: -10.7390146, 4.8280225, -10.7532301, 4.8201923, -15.5035095, 15.5407257
18: 4.0754752, 16.7662201, 4.1546535, 16.7599030, -11.9248047, 11.8641319
19: 3.1005013, 8.5627565, 3.1302130, 8.5774584, -5.0346222, 4.9807301
20: -1.8937058, 5.9462528, -1.8719223, 5.9383140, -7.0364532, 7.0201054
21: 4.9063463, 12.4429932, 4.9296856, 12.4306269, -7.2023239, 7.1738167
22: 2.8878143, 11.2140722, 2.9377689, 11.2156353, -6.4548492, 6.4156895
23: 3.2740507, 9.7840662, 3.3134303, 9.7900276, -4.4507694, 4.4096985
24: 2.8978372, 11.5596628, 2.9483109, 11.5741730, -6.8828049, 6.8178062
25: 3.5119233, 12.6613312, 3.5561252, 12.6615295, -7.1287422, 7.0992069
26: 2.0712175, 15.6671171, 2.1287308, 15.6612644, -13.3227386, 13.2795868
27: -0.7017599, 9.8111877, -0.6404554, 9.8196344, -9.2865906, 9.2221451
28: 1.6494337, 9.6097794, 1.6909130, 9.6179228, -6.0824375, 6.0298195
29: 4.6291327, 11.1152840, 4.6540780, 11.1154613, -4.9598999, 4.9428120
30: 0.8660485, 10.2636089, 0.8998672, 10.2525711, -8.6774330, 8.6588020
31: 4.2172079, 12.6436596, 4.2551966, 12.6631289, -7.0027313, 6.9430923
32: -15.5150909, -4.0550833, -15.5123434, -4.1077318, -8.4378662, 8.4930878
33: -27.0357075, -9.0078869, -27.0077744, -9.0134821, -13.3081665, 13.2768860
34: -25.5031967, -10.6525869, -25.4917450, -10.6797228, -10.5750961, 10.5822029
35: -14.8846607, -0.4809968, -14.8710947, -0.5147607, -12.8604202, 12.8772278
36: -13.3104229, 2.0662098, -13.2996359, 2.0323710, -14.7419281, 14.7675934
37: -24.9110756, -7.8149452, -24.8852654, -7.8143177, -14.0925522, 14.0477448
38: -18.3224087, -0.8472652, -18.3124542, -0.8821192, -17.4402885, 17.4651890
39: -28.8242626, -9.7697849, -28.7981739, -9.8188648, -16.3850555, 16.4040909
40: -30.2431011, -17.9084816, -30.2270088, -17.9171524, -7.8536072, 7.8058453
41: -16.2409000, -2.8153076, -16.2330379, -2.8300042, -10.2456779, 10.2474251
42: -16.2277584, -7.5664506, -16.2199650, -7.6077223, -6.5940876, 6.6319027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=56, inp2_unstable=55, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=129, inp2_unstable=129, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=7, inp2_unstable=7, delta_unstable=43

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 543

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1687

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.9238673, upper bound: 5.9260213
time: 17.27 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 18, lower bound: -5.9238673, upper bound: 5.9270107
time: 7.58 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -22.5227833, -7.9669037, -22.5074120, -7.9786730, -8.7481194, 8.7310524
1: -9.2057409, -1.1195538, -9.2041388, -1.1357069, -5.6099396, 5.6124649
2: -10.1229057, -1.6282620, -10.1071301, -1.6416829, -5.1658058, 5.1549931
3: -16.8245392, -5.7195554, -16.8153343, -5.7737198, -8.1944847, 8.2317314
4: -13.9942799, -3.8107386, -13.9688110, -3.8192768, -7.3767700, 7.3553848
5: -13.8344765, -1.8193409, -13.8225060, -1.8510535, -8.0705032, 8.0890846
6: -14.1479425, -2.8429105, -14.1434412, -2.8847914, -7.6239052, 7.6514664
7: -16.0419521, -3.6060863, -16.0366402, -3.6341958, -8.2669449, 8.2716827
8: -22.6337891, -8.7903252, -22.6176414, -8.8043318, -10.8528748, 10.8242149
9: -13.1967983, -2.8510175, -13.1932077, -2.8825042, -8.1488152, 8.1657562
10: -9.5393286, 1.3255715, -9.5271969, 1.2812071, -10.4399567, 10.4516983
11: 3.7635422, 10.7807512, 3.7751777, 10.7460670, -5.1303539, 5.1543217
12: -4.6093001, 13.1345139, -4.5991211, 13.0676889, -13.7717133, 13.8236389
13: -17.4548779, -1.6141267, -17.4466553, -1.7198794, -15.4550171, 15.5505905
14: -18.0043926, 0.6331282, -17.9514275, 0.5771341, -18.0248260, 17.9963379
15: -11.7504282, -3.3879209, -11.7070227, -3.3982403, -7.6087112, 7.5786896
16: -10.6577454, -1.5598770, -10.6535368, -1.5921522, -8.6849480, 8.6575928
17: -10.7443323, 4.8145571, -10.7195244, 4.7795010, -15.4665146, 15.4832993
18: 4.0777698, 16.7660809, 4.1712127, 16.7572613, -11.9174347, 11.8428993
19: 3.1051130, 8.5646038, 3.1483679, 8.5620623, -5.0070229, 4.9655418
20: -1.8940222, 5.9471550, -1.8653030, 5.9387236, -7.0342598, 7.0175419
21: 4.9059672, 12.4463959, 4.9335823, 12.4381752, -7.1978455, 7.1787109
22: 2.8881407, 11.2133274, 2.9452677, 11.2104368, -6.4515686, 6.4054527
23: 3.2776022, 9.7868786, 3.3269978, 9.7819214, -4.4401474, 4.4052334
24: 2.9035602, 11.5598497, 2.9713650, 11.5577917, -6.8639908, 6.8003082
25: 3.5128460, 12.6624527, 3.5611448, 12.6577702, -7.1217422, 7.0899601
26: 2.0719376, 15.6579113, 2.1604910, 15.6302357, -13.2940979, 13.2371140
27: -0.6953539, 9.8121004, -0.6146079, 9.8065872, -9.2773628, 9.2058716
28: 1.6541234, 9.6123199, 1.7065713, 9.6056137, -6.0582886, 6.0189686
29: 4.6311140, 11.1136894, 4.6650362, 11.1075411, -4.9592819, 4.9318485
30: 0.8660743, 10.2626352, 0.8959417, 10.2456188, -8.6719818, 8.6601372
31: 4.2248750, 12.6459875, 4.2835913, 12.6435633, -6.9729271, 6.9208870
32: -15.5165062, -4.0591817, -15.5092640, -4.1180091, -8.4269447, 8.4728394
33: -27.0321598, -9.0091333, -26.9883041, -9.0537024, -13.2652893, 13.2531281
34: -25.5002365, -10.6530685, -25.4777451, -10.6948814, -10.5561218, 10.5644531
35: -14.8841209, -0.4817626, -14.8624687, -0.5290223, -12.8473511, 12.8673630
36: -13.3164845, 2.0655012, -13.3084068, 2.0349576, -14.7508163, 14.7727432
37: -24.9146671, -7.8155546, -24.8879776, -7.8248644, -14.0811157, 14.0453949
38: -18.3265057, -0.8500199, -18.3096275, -0.8836365, -17.4428692, 17.4596081
39: -28.8332253, -9.7701302, -28.8167953, -9.8185701, -16.3929520, 16.4055023
40: -30.2422867, -17.9090233, -30.2190838, -17.9356403, -7.8389244, 7.7829609
41: -16.2389622, -2.8155999, -16.2234879, -2.8465712, -10.2234650, 10.2254372
42: -16.2294579, -7.5679760, -16.2217636, -7.6163015, -6.5800323, 6.6252346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=56, inp2_unstable=55, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=129, inp2_unstable=129, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=7, inp2_unstable=7, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1699
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1687
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1705
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1754
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1699

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 18, lower bound: -5.9243965, upper bound: 5.9238668
time: 12.45 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 18, lower bound: -5.9243965, upper bound: 5.9254840
time: 10.63 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -22.5256424, -7.9657254, -22.5211449, -7.9656591, -8.7676468, 8.7494125
1: -9.2066469, -1.1183801, -9.2090435, -1.1271944, -5.6298084, 5.6150627
2: -10.1290979, -1.6273842, -10.1295757, -1.6209621, -5.1920776, 5.1781025
3: -16.8266144, -5.7178545, -16.8234978, -5.7604113, -8.2163391, 8.2432556
4: -13.9982586, -3.8097453, -13.9901285, -3.8025780, -7.4108887, 7.3795471
5: -13.8379154, -1.8176613, -13.8363771, -1.8315036, -8.0998573, 8.1096191
6: -14.1486273, -2.8417745, -14.1474895, -2.8756926, -7.6401939, 7.6587276
7: -16.0440865, -3.6045752, -16.0465508, -3.6226568, -8.2931213, 8.2825890
8: -22.6375561, -8.7891808, -22.6356621, -8.7925816, -10.8798485, 10.8404732
9: -13.1977940, -2.8446043, -13.2151775, -2.8572798, -8.1702499, 8.1936798
10: -9.5409346, 1.3362217, -9.5666122, 1.3211775, -10.4770050, 10.5019073
11: 3.7624211, 10.7887192, 3.7506652, 10.7742634, -5.1544380, 5.1869850
12: -4.6110983, 13.1560450, -4.6732063, 13.1453247, -13.8444977, 13.9197388
13: -17.4562187, -1.6036172, -17.4781189, -1.6753888, -15.4956512, 15.5847702
14: -18.0085773, 0.6537580, -18.0485554, 0.6478796, -18.0978928, 18.1153717
15: -11.7555418, -3.3866057, -11.7312937, -3.3825161, -7.6238327, 7.6015663
16: -10.6590595, -1.5585943, -10.6596937, -1.5826309, -8.7159615, 8.6635742
17: -10.7464409, 4.8281174, -10.7764692, 4.8288937, -15.5201492, 15.5615463
18: 4.0746770, 16.7675934, 4.1451874, 16.7636776, -11.9268570, 11.8734131
19: 3.0989628, 8.5651588, 3.1230168, 8.5844240, -5.0396347, 4.9917297
20: -1.8958333, 5.9499993, -1.8838587, 5.9496536, -7.0430183, 7.0352669
21: 4.9043989, 12.4486113, 4.9159274, 12.4470062, -7.2115593, 7.1921005
22: 2.8861070, 11.2141933, 2.9311593, 11.2160816, -6.4580383, 6.4225273
23: 3.2721786, 9.7871971, 3.3032265, 9.7994976, -4.4545631, 4.4235916
24: 2.8962479, 11.5603008, 2.9419036, 11.5757713, -6.8868637, 6.8267899
25: 3.5097198, 12.6632576, 3.5450349, 12.6673889, -7.1317825, 7.1101360
26: 2.0682364, 15.6671219, 2.1169400, 15.6616497, -13.3263626, 13.2917633
27: -0.7024187, 9.8124599, -0.6453547, 9.8229895, -9.2924423, 9.2310944
28: 1.6475710, 9.6131287, 1.6793182, 9.6282578, -6.0870590, 6.0448608
29: 4.6279745, 11.1155300, 4.6498375, 11.1162291, -4.9689484, 4.9455414
30: 0.8642859, 10.2667942, 0.8857875, 10.2633276, -8.6868896, 8.6743393
31: 4.2153854, 12.6466961, 4.2461386, 12.6719093, -7.0089073, 6.9555435
32: -15.5178328, -4.0543814, -15.5203867, -4.0989084, -8.4491615, 8.4943810
33: -27.0443802, -9.0076656, -27.0334244, -9.0057907, -13.3256302, 13.2888260
34: -25.5066757, -10.6522179, -25.5023155, -10.6761179, -10.5830765, 10.5862350
35: -14.8904724, -0.4807770, -14.8876810, -0.5085278, -12.8728485, 12.8924408
36: -13.3183441, 2.0664563, -13.3225632, 2.0409732, -14.7586441, 14.7896423
37: -24.9211063, -7.8148308, -24.9157753, -7.8047409, -14.1122055, 14.0676270
38: -18.3299789, -0.8469262, -18.3352757, -0.8726707, -17.4573078, 17.4883499
39: -28.8361931, -9.7696323, -28.8330078, -9.8057728, -16.4099426, 16.4204102
40: -30.2490883, -17.9082794, -30.2459469, -17.9088326, -7.8692627, 7.8076267
41: -16.2448864, -2.8146167, -16.2444763, -2.8218789, -10.2577133, 10.2499886
42: -16.2306118, -7.5659456, -16.2276478, -7.6020551, -6.5994396, 6.6364632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=56, inp2_unstable=55, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=129, inp2_unstable=129, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=7, inp2_unstable=7, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 1687
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 905
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 905
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 524
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1754
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 1738
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1738
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1701
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: B, layer: 1, pos: 1494
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 543

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1699

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 18, lower bound: -5.9243965, upper bound: 5.9238668
time: 13.31 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 18, lower bound: -5.9243965, upper bound: 5.9254840
time: 33.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 49.21 seconds
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 49.21
Output dim: 18, lower bound: -5.9238673, upper bound: 5.9260213
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 49.21
Output dim: 18, lower bound: -5.9238673, upper bound: 5.9270107
IS_A2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 49.21
Output dim: 18, lower bound: -5.9243965, upper bound: 5.9238668
IS_A2_B2_B1_A2, status: Status.VERIFIED, split count: 4, time: 49.21
Output dim: 18, lower bound: -5.9243965, upper bound: 5.9254840
IS_A2_B2_B2_A1, status: Status.VERIFIED, split count: 4, time: 49.21
Output dim: 18, lower bound: -5.9243965, upper bound: 5.9238668
IS_A2_B2_B2_A2, status: Status.VERIFIED, split count: 4, time: 49.21
Output dim: 18, lower bound: -5.9243965, upper bound: 5.9254840

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -22.5139523, -7.9720674, -22.5199814, -7.9722681, -8.7443161, 8.7435074
1: -9.2034168, -1.1267266, -9.2090244, -1.1340795, -5.5977154, 5.6094055
2: -10.1091290, -1.6315320, -10.1300611, -1.6248882, -5.1671314, 5.1735477
3: -16.8176212, -5.7259789, -16.8212509, -5.7676811, -8.1893311, 8.2333336
4: -13.9809265, -3.8147202, -13.9897089, -3.8075495, -7.3740158, 7.3664017
5: -13.8235054, -1.8273811, -13.8349533, -1.8426783, -8.0750542, 8.0996590
6: -14.1448631, -2.8468688, -14.1438580, -2.8778734, -7.6228294, 7.6522331
7: -16.0355492, -3.6127601, -16.0452938, -3.6313891, -8.2545967, 8.2727318
8: -22.6241570, -8.7943621, -22.6355038, -8.7968407, -10.8448410, 10.8260651
9: -13.1941710, -2.8713081, -13.2113571, -2.8714545, -8.1572037, 8.1637955
10: -9.5350904, 1.2953215, -9.5620918, 1.3074298, -10.4614410, 10.4520569
11: 3.7678459, 10.7600412, 3.7570300, 10.7712307, -5.1512985, 5.1520233
12: -4.5989084, 13.0817184, -4.6559620, 13.1363697, -13.8259125, 13.8302155
13: -17.4437599, -1.6426351, -17.4567451, -1.6888740, -15.4701538, 15.5268555
14: -17.9931240, 0.5819550, -18.0401211, 0.6443815, -18.0748062, 18.0359344
15: -11.7354250, -3.3931279, -11.7261353, -3.3898458, -7.6013031, 7.5883942
16: -10.6537971, -1.5698535, -10.6535053, -1.5982116, -8.6602135, 8.6491623
17: -10.7318363, 4.7829108, -10.7532301, 4.8201923, -15.4974365, 15.4828949
18: 4.0888023, 16.7598305, 4.1546535, 16.7599030, -11.9131393, 11.8503342
19: 3.1219056, 8.5607185, 3.1302130, 8.5774584, -5.0093079, 4.9814816
20: -1.8874936, 5.9361615, -1.8719223, 5.9383140, -7.0311394, 7.0113354
21: 4.9119835, 12.4352312, 4.9296856, 12.4306269, -7.1835632, 7.1685944
22: 2.9000165, 11.2109938, 2.9377689, 11.2156353, -6.4437599, 6.4078922
23: 3.2918756, 9.7828026, 3.3134303, 9.7900276, -4.4340153, 4.4124985
24: 2.9223113, 11.5580626, 2.9483109, 11.5741730, -6.8575439, 6.8194351
25: 3.5229321, 12.6582375, 3.5561252, 12.6615295, -7.1174278, 7.0885468
26: 2.0839014, 15.6339483, 2.1287308, 15.6612644, -13.3156281, 13.2459869
27: -0.6752286, 9.8097553, -0.6404554, 9.8196344, -9.2635612, 9.2212372
28: 1.6719844, 9.6065950, 1.6909130, 9.6179228, -6.0592136, 6.0294094
29: 4.6405196, 11.1089659, 4.6540780, 11.1154613, -4.9492168, 4.9379044
30: 0.8723330, 10.2493000, 0.8998672, 10.2525711, -8.6731491, 8.6464729
31: 4.2498198, 12.6411438, 4.2551966, 12.6631289, -6.9691124, 6.9445076
32: -15.5102978, -4.0717211, -15.5123434, -4.1077318, -8.4337578, 8.4727707
33: -26.9931412, -9.0127544, -27.0077744, -9.0134821, -13.2659836, 13.2834473
34: -25.4807777, -10.6555901, -25.4917450, -10.6797228, -10.5517502, 10.5842361
35: -14.8625326, -0.4843652, -14.8710947, -0.5147607, -12.8387375, 12.8742981
36: -13.3037701, 2.0611246, -13.2996359, 2.0323710, -14.7348633, 14.7600250
37: -24.8876705, -7.8174200, -24.8852654, -7.8143177, -14.0690613, 14.0527725
38: -18.3103199, -0.8617306, -18.3124542, -0.8821192, -17.4281998, 17.4507236
39: -28.8116245, -9.7715111, -28.7981739, -9.8188648, -16.3712921, 16.4043884
40: -30.2197552, -17.9117126, -30.2270088, -17.9171524, -7.8299637, 7.8058414
41: -16.2204285, -2.8188901, -16.2330379, -2.8300042, -10.2199287, 10.2457962
42: -16.2235832, -7.5772958, -16.2199650, -7.6077223, -6.5908451, 6.6147079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=55, inp2_unstable=55, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=129, inp2_unstable=129, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=7, inp2_unstable=7, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 1680

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 931

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -5.9218828, upper bound: 5.9210898
time: 14.47 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 18, lower bound: -5.9218828, upper bound: 5.9210898
time: 10.50 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -22.5276184, -7.9590435, -22.5199814, -7.9722681, -8.7598076, 8.7564774
1: -9.2082882, -1.1182299, -9.2090244, -1.1340795, -5.6240673, 5.6352997
2: -10.1315804, -1.6108201, -10.1300611, -1.6248882, -5.1897488, 5.1938190
3: -16.8257904, -5.7126508, -16.8212509, -5.7676811, -8.2091713, 8.2549667
4: -14.0022335, -3.7980518, -13.9897089, -3.8075495, -7.4095535, 7.3968010
5: -13.8374081, -1.8078179, -13.8349533, -1.8426783, -8.0932350, 8.1220322
6: -14.1488962, -2.8377848, -14.1438580, -2.8778734, -7.6361046, 7.6696167
7: -16.0454941, -3.6011982, -16.0452938, -3.6313891, -8.2849846, 8.3017845
8: -22.6422234, -8.7825880, -22.6355038, -8.7968407, -10.8786850, 10.8546066
9: -13.2160988, -2.8460960, -13.2113571, -2.8714545, -8.1636925, 8.1730576
10: -9.5745630, 1.3353167, -9.5620918, 1.3074298, -10.4878120, 10.4785385
11: 3.7433534, 10.7882347, 3.7570300, 10.7712307, -5.1582870, 5.1635303
12: -4.6731176, 13.1593857, -4.6559620, 13.1363697, -13.8837509, 13.8905411
13: -17.4752159, -1.5982103, -17.4567451, -1.6888740, -15.4917374, 15.5632248
14: -18.0902958, 0.6526890, -18.0401211, 0.6443815, -18.1510849, 18.0934448
15: -11.7596741, -3.3773966, -11.7261353, -3.3898458, -7.6206131, 7.5982094
16: -10.6599598, -1.5602603, -10.6535053, -1.5982116, -8.6988182, 8.6858063
17: -10.7888260, 4.8323064, -10.7532301, 4.8201923, -15.5637741, 15.5439072
18: 4.0627789, 16.7662315, 4.1546535, 16.7599030, -11.9439468, 11.8631096
19: 3.0965645, 8.5830727, 3.1302130, 8.5774584, -5.0321770, 5.0016556
20: -1.9060423, 5.9470820, -1.8719223, 5.9383140, -7.0425224, 7.0163250
21: 4.8943472, 12.4440556, 4.9296856, 12.4306269, -7.2063522, 7.1816521
22: 2.8859160, 11.2166147, 2.9377689, 11.2156353, -6.4630890, 6.4172764
23: 3.2681274, 9.8003740, 3.3134303, 9.7900276, -4.4461212, 4.4113579
24: 2.8928556, 11.5760450, 2.9483109, 11.5741730, -6.8773689, 6.8255997
25: 3.5067935, 12.6678638, 3.5561252, 12.6615295, -7.1381302, 7.1012936
26: 2.0403728, 15.6653595, 2.1287308, 15.6612644, -13.3501511, 13.2687607
27: -0.7059563, 9.8261499, -0.6404554, 9.8196344, -9.2852745, 9.2273178
28: 1.6447453, 9.6292400, 1.6909130, 9.6179228, -6.0802193, 6.0450649
29: 4.6253510, 11.1176739, 4.6540780, 11.1154613, -4.9629955, 4.9442501
30: 0.8622237, 10.2670231, 0.8998672, 10.2525711, -8.6789780, 8.6575317
31: 4.2123933, 12.6694908, 4.2551966, 12.6631289, -6.9941864, 6.9580383
32: -15.5214367, -4.0525846, -15.5123434, -4.1077318, -8.4504128, 8.4959450
33: -27.0383224, -8.9648399, -27.0077744, -9.0134821, -13.2832642, 13.3031769
34: -25.5053787, -10.6367941, -25.4917450, -10.6797228, -10.5631943, 10.5901871
35: -14.8878498, -0.4638960, -14.8710947, -0.5147607, -12.8620071, 12.8923340
36: -13.3179779, 2.0671372, -13.2996359, 2.0323710, -14.7511292, 14.7680130
37: -24.9155025, -7.7972078, -24.8852654, -7.8143177, -14.0782318, 14.0567398
38: -18.3360291, -0.8508310, -18.3124542, -0.8821192, -17.4539108, 17.4616241
39: -28.8278618, -9.7586994, -28.7981739, -9.8188648, -16.3802338, 16.4100800
40: -30.2465858, -17.8848763, -30.2270088, -17.9171524, -7.8451271, 7.8178101
41: -16.2414436, -2.7942195, -16.2330379, -2.8300042, -10.2387428, 10.2653694
42: -16.2294693, -7.5630646, -16.2199650, -7.6077223, -6.5978374, 6.6344280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=55, inp2_unstable=55, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=129, inp2_unstable=129, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=7, inp2_unstable=7, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1699
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 1384
type: A, layer: 1, pos: 1384
type: B, layer: 1, pos: 841
type: A, layer: 1, pos: 841
type: B, layer: 1, pos: 715
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 523
type: B, layer: 1, pos: 523
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 524
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 552
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 1653
type: B, layer: 1, pos: 1653
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 552
type: B, layer: 1, pos: 1480
type: A, layer: 1, pos: 1480
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1701
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 1680
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1776
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1494
type: B, layer: 1, pos: 1494
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 1680

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 931

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 18, lower bound: -5.9218828, upper bound: 5.9205586
time: 7.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 18, lower bound: -5.9218828, upper bound: 5.9216634
time: 9.03 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 18.47 seconds
IS_A2_B1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 18.47
Output dim: 18, lower bound: -5.9218828, upper bound: 5.9210898
IS_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 18.47
Output dim: 18, lower bound: -5.9218828, upper bound: 5.9210898
IS_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 18.47
Output dim: 18, lower bound: -5.9218828, upper bound: 5.9205586
IS_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 18.47
Output dim: 18, lower bound: -5.9218828, upper bound: 5.9216634

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 30.24 + 323.28 = 353.52 seconds
