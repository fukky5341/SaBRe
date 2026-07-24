## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 3)
Time budget: 3600 seconds
Split limit: 100
Threshold: 12.5154011709


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8566437, 15.8566437)
1: (-30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8220482, 11.8220482)
2: (-21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7193909, 10.7193909)
3: (-30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.8145676, 12.8145676)
4: (-18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6874657, 12.6874638)
5: (-30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2515907, 14.2515907)
6: (-14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0486851, 11.0486870)
7: (-47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0904388, 13.0904388)
8: (-33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9951477, 13.9951477)
9: (-18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7697144, 9.7697124)
10: (-38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2488327, 18.2488289)
11: (-57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3551178, 13.3551159)
12: (-1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0992737, 16.0992737)
13: (-8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8771172, 14.8771172)
14: (-79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3204575, 25.3204575)
15: (-11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1575623, 13.1575603)
16: (-49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7832146, 12.7832165)
17: (-79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1875992, 22.1876030)
18: (-11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2396622, 19.2396622)
19: (-27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6138172, 10.6138191)
20: (-17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4437904, 12.4437904)
21: (-39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1181259, 14.1181259)
22: (-17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7949715, 12.7949715)
23: (-22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1917267, 11.1917267)
24: (-8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3584480, 9.3584461)
25: (-5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0200996, 12.0201035)
26: (-19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8158875, 18.8158875)
27: (-23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9530678, 11.9530678)
28: (-18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5248737, 12.5248756)
29: (-36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6478329, 11.6478310)
30: (-27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5930367, 12.5930367)
31: (-23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2246971, 12.2246971)
32: (-4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9989204, 11.9989204)
33: (17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7373810, 17.7373810)
34: (6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5447445, 13.5447445)
35: (10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6273880, 15.6273918)
36: (5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5914555, 12.5914536)
37: (13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6318493, 13.6318512)
38: (4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1569443, 17.1569481)
39: (8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0235252, 15.0235252)
40: (6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7253189, 12.7253189)
41: (-5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4167938, 11.4167938)
42: (-16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3823166, 11.3823166)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.81 + 16.93 = 19.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 33, lower bound: -12.5279291, upper bound: 12.5279291

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 667
type: B, layer: 1, pos: 667
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: A, layer: 1, pos: 541
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 1758
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 605
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1730
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 1630
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1005
type: A, layer: 1, pos: 1759
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1616
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1725
type: A, layer: 1, pos: 1725
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1021
type: B, layer: 1, pos: 1021
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 580
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1760
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1493
type: A, layer: 1, pos: 1648
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: B, layer: 1, pos: 1422
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 516
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 1632
type: B, layer: 1, pos: 1632
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1649
type: A, layer: 1, pos: 1649
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1413
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 824
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1487
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1432
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1664
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1492
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 1398
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 616
type: B, layer: 1, pos: 616
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 1475
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 1513

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 667

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5160572, upper bound: 12.4698824
time: 5.79 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5096405, upper bound: 12.5096400
time: 6.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.15 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.15
Output dim: 33, lower bound: -12.5160572, upper bound: 12.4698824
IS_A2, status: Status.VERIFIED, split count: 1, time: 12.15
Output dim: 33, lower bound: -12.5096405, upper bound: 12.5096400

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -42.6801376, -18.9411430, -42.6874275, -18.9399910, -15.8547211, 15.8538933
1: -30.4613266, -12.9632349, -30.4739475, -12.9624367, -11.8162766, 11.8204727
2: -21.5898724, -5.9931231, -21.5952587, -5.9921961, -10.7183228, 10.7170410
3: -30.5818176, -12.4461384, -30.5879440, -12.4453697, -12.8133087, 12.8114738
4: -18.8401527, 0.6078751, -18.8445339, 0.6088285, -12.6829681, 12.6865635
5: -30.7230988, -9.8973227, -30.7314377, -9.8963623, -14.2496529, 14.2492218
6: -14.2701473, 1.3312201, -14.2883472, 1.3316171, -11.0331211, 11.0481186
7: -47.4790955, -25.9932384, -47.4976425, -25.9926605, -13.0808029, 13.0892029
8: -33.6756020, -12.1373215, -33.6759033, -12.1367245, -13.9948158, 13.9943848
9: -18.5995884, -6.9908600, -18.5999546, -6.9883556, -9.7692757, 9.7671833
10: -38.3405762, -15.4219418, -38.3487167, -15.4204731, -18.2413254, 18.2473259
11: -57.3627701, -34.7031937, -57.3802986, -34.7022781, -13.3463745, 13.3537598
12: -1.9273813, 17.9379463, -1.9409168, 17.9383068, -16.0860443, 16.0989609
13: -8.2628403, 9.4550171, -8.2643576, 9.4567671, -14.8761978, 14.8734398
14: -79.0421982, -46.7462921, -79.0532837, -46.7443428, -25.3094559, 25.3186798
15: -11.0547495, 7.7988348, -11.0553875, 7.7995930, -13.1565437, 13.1559658
16: -49.1288605, -29.5191689, -49.1398125, -29.5181141, -12.7765083, 12.7818928
17: -79.6196594, -45.0309830, -79.6467896, -45.0305099, -22.1620407, 22.1872139
18: -11.5763683, 9.5346642, -11.5796604, 9.5358448, -19.2328949, 19.2384186
19: -27.5831299, -12.5089245, -27.5895557, -12.5076895, -10.6102581, 10.6119328
20: -17.4544373, -4.3165789, -17.4594021, -4.3159504, -12.4392395, 12.4431839
21: -39.7890854, -20.1266537, -39.7976608, -20.1243858, -14.1126175, 14.1156235
22: -17.6691628, 0.3119321, -17.6748886, 0.3123665, -12.7895775, 12.7946033
23: -22.2929611, -7.4006300, -22.2937069, -7.3993239, -11.1907654, 11.1903667
24: -8.7161636, 4.7452455, -8.7166920, 4.7535696, -9.3571548, 9.3556061
25: -5.7722144, 8.8844843, -5.7729368, 8.8897533, -12.0179787, 12.0188675
26: -19.1801605, 2.4258451, -19.1811810, 2.4279919, -18.8092194, 18.8142166
27: -23.7028656, -7.1738029, -23.7054405, -7.1731005, -11.9492798, 11.9523811
28: -18.3382816, -0.1337910, -18.3387642, -0.1329041, -12.5235920, 12.5244637
29: -36.4196854, -18.7419586, -36.4321899, -18.7416058, -11.6448841, 11.6459198
30: -27.8878822, -7.8068948, -27.8891983, -7.8031111, -12.5909462, 12.5903435
31: -23.1834316, -7.2160611, -23.1913071, -7.2152233, -12.2197304, 12.2239151
32: -4.6181736, 10.3700237, -4.6359253, 10.3701916, -11.9802971, 11.9987030
33: 17.4534073, 40.2334518, 17.4503784, 40.2536736, -17.7345695, 17.7173538
34: 6.0365124, 25.4906311, 6.0333824, 25.5071201, -13.5413876, 13.5382881
35: 10.2638683, 29.9025764, 10.2628212, 29.9235191, -15.6263237, 15.6065254
36: 5.8074012, 23.6232624, 5.8066578, 23.6290283, -12.5907192, 12.5886650
37: 13.6724844, 32.5240440, 13.6705208, 32.5389824, -13.6299725, 13.6167793
38: 4.0694885, 25.2192268, 4.0681987, 25.2226028, -17.1540909, 17.1555595
39: 8.9878712, 30.9616280, 8.9867115, 30.9783421, -15.0223961, 15.0093727
40: 6.9248028, 25.1612892, 6.9200339, 25.1681404, -12.7202225, 12.7223396
41: -5.0993948, 10.3618336, -5.1083040, 10.3620338, -11.4134026, 11.4154015
42: -16.9030151, 0.0339553, -16.9170742, 0.0343406, -11.3734474, 11.3813972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=103, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1746
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 541
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 573
type: B, layer: 1, pos: 573
type: A, layer: 1, pos: 574
type: B, layer: 1, pos: 574
type: A, layer: 1, pos: 1763
type: B, layer: 1, pos: 1763
type: A, layer: 1, pos: 575
type: B, layer: 1, pos: 575
type: A, layer: 1, pos: 1779
type: B, layer: 1, pos: 1779
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1758
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 582
type: A, layer: 1, pos: 582
type: B, layer: 1, pos: 605
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 589
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1597
type: A, layer: 1, pos: 1597
type: B, layer: 1, pos: 1730
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1713
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1562
type: A, layer: 1, pos: 1562
type: B, layer: 1, pos: 606
type: A, layer: 1, pos: 606
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1764
type: B, layer: 1, pos: 1764
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1742
type: A, layer: 1, pos: 1742
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1584
type: B, layer: 1, pos: 1584
type: A, layer: 1, pos: 1726
type: B, layer: 1, pos: 1726
type: A, layer: 1, pos: 1005
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1759
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 590
type: B, layer: 1, pos: 590
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1761
type: A, layer: 1, pos: 1761
type: B, layer: 1, pos: 515
type: A, layer: 1, pos: 515
type: B, layer: 1, pos: 1791
type: A, layer: 1, pos: 1600
type: B, layer: 1, pos: 1600
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1743
type: B, layer: 1, pos: 1743
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1615
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1491
type: B, layer: 1, pos: 1491
type: A, layer: 1, pos: 1568
type: B, layer: 1, pos: 1568
type: A, layer: 1, pos: 1459
type: B, layer: 1, pos: 1459
type: A, layer: 1, pos: 1680
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1665
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1021
type: B, layer: 1, pos: 1021
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 580
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1760
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1493
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1648
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1634
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1422
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1531
type: A, layer: 1, pos: 1531
type: B, layer: 1, pos: 1775
type: A, layer: 1, pos: 1775
type: B, layer: 1, pos: 516
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1552
type: B, layer: 1, pos: 1552
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1786
type: B, layer: 1, pos: 1786
type: A, layer: 1, pos: 1438
type: B, layer: 1, pos: 1438
type: A, layer: 1, pos: 1444
type: B, layer: 1, pos: 1444
type: A, layer: 1, pos: 1420
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1544
type: A, layer: 1, pos: 1544
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1697
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1784
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1784
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 824
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1536
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1487
type: A, layer: 1, pos: 1405
type: B, layer: 1, pos: 1405
type: A, layer: 1, pos: 1471
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1432
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1664
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1398
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1475
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1513

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1746

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5069643, upper bound: 12.4401827
time: 18.84 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5065030, upper bound: 12.4603284
time: 6.16 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.27 seconds
IS_A1_A1, status: Status.VERIFIED, split count: 2, time: 27.27
Output dim: 33, lower bound: -12.5069643, upper bound: 12.4401827
IS_A1_A2, status: Status.VERIFIED, split count: 2, time: 27.27
Output dim: 33, lower bound: -12.5065030, upper bound: 12.4603284

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 19.74 + 39.42 = 59.16 seconds
