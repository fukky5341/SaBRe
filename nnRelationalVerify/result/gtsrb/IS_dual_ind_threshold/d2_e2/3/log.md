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
execution time: IAR + RelationalAnalysis = 2.89 + 17.33 = 20.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 33, lower bound: -12.5279291, upper bound: 12.5279291

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1405

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 541

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4878965, upper bound: 12.5270746
time: 6.10 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5271960, upper bound: 12.5271963
time: 20.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 27.06 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 27.06
Output dim: 33, lower bound: -12.4878965, upper bound: 12.5270746
IS_A2, status: Status.UNKNOWN, split count: 1, time: 27.06
Output dim: 33, lower bound: -12.5271960, upper bound: 12.5271963

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -42.6826248, -18.9538937, -42.6862793, -18.9433365, -15.8477783, 15.8408852
1: -30.4717999, -12.9865513, -30.4734497, -12.9681644, -11.8134727, 11.7967682
2: -21.5863953, -5.9981351, -21.5931244, -5.9935741, -10.7055016, 10.7109604
3: -30.5728245, -12.4529018, -30.5844097, -12.4471292, -12.7967453, 12.8042564
4: -18.8383598, 0.6015921, -18.8431015, 0.6071055, -12.6763382, 12.6713943
5: -30.7228050, -9.9054995, -30.7293148, -9.8985004, -14.2361603, 14.2393951
6: -14.2759733, 1.3283076, -14.2854490, 1.3308630, -11.0321884, 11.0419998
7: -47.4955978, -26.0141106, -47.4971237, -25.9976807, -13.0827789, 13.0674133
8: -33.6711121, -12.1459045, -33.6746826, -12.1388760, -13.9857483, 13.9837646
9: -18.5926208, -7.0116000, -18.5982475, -6.9937568, -9.7577209, 9.7427807
10: -38.3453293, -15.4752483, -38.3479729, -15.4332495, -18.2336769, 18.1931000
11: -57.3773270, -34.7554665, -57.3795738, -34.7146683, -13.3384895, 13.3023262
12: -1.9340796, 17.9074707, -1.9393587, 17.9311142, -16.0872345, 16.0668716
13: -8.2317619, 9.4442205, -8.2567635, 9.4538345, -14.8401604, 14.8567924
14: -79.0464783, -46.8049927, -79.0516891, -46.7584839, -25.3033524, 25.2597504
15: -11.0524750, 7.7826619, -11.0546989, 7.7956553, -13.1489563, 13.1346989
16: -49.1354942, -29.5611839, -49.1388016, -29.5281639, -12.7669983, 12.7357063
17: -79.6410675, -45.0946960, -79.6454239, -45.0454941, -22.1670723, 22.1220703
18: -11.5681238, 9.5020037, -11.5769424, 9.5279646, -19.2206116, 19.2029572
19: -27.5811005, -12.5258951, -27.5875874, -12.5119343, -10.5964680, 10.5924683
20: -17.4506111, -4.3315969, -17.4573517, -4.3195753, -12.4304123, 12.4255180
21: -39.7890778, -20.1610107, -39.7956390, -20.1329193, -14.0985813, 14.0788803
22: -17.6688309, 0.2853928, -17.6734657, 0.3060513, -12.7824821, 12.7665596
23: -22.2860603, -7.4185715, -22.2919254, -7.4038062, -11.1750298, 11.1692448
24: -8.7095470, 4.7374549, -8.7150326, 4.7498255, -9.3461781, 9.3401451
25: -5.7628827, 8.8648300, -5.7705870, 8.8839369, -12.0016174, 11.9913902
26: -19.1705933, 2.3929319, -19.1787186, 2.4198208, -18.7988663, 18.7784805
27: -23.6993523, -7.2024469, -23.7039871, -7.1799517, -11.9400673, 11.9224815
28: -18.3291321, -0.1425529, -18.3365135, -0.1351409, -12.5049286, 12.5103798
29: -36.4299393, -18.7791061, -36.4316788, -18.7504234, -11.6366291, 11.6125336
30: -27.8853207, -7.8389139, -27.8883057, -7.8114328, -12.5800819, 12.5556431
31: -23.1817970, -7.2348266, -23.1890850, -7.2197914, -12.2082405, 12.2022057
32: -4.6186371, 10.3685732, -4.6319156, 10.3698273, -11.9799881, 11.9918556
33: 17.5082569, 40.2517319, 17.4638596, 40.2531967, -17.6791840, 17.7231789
34: 6.0680346, 25.5058632, 6.0415020, 25.5068245, -13.5095367, 13.5347042
35: 10.3143864, 29.9209747, 10.2748632, 29.9229507, -15.5754242, 15.6145134
36: 5.8445120, 23.6281776, 5.8154936, 23.6288395, -12.5529289, 12.5813828
37: 13.6988430, 32.5373001, 13.6771336, 32.5386238, -13.6030083, 13.6242027
38: 4.1207705, 25.2202415, 4.0804558, 25.2220249, -17.1039314, 17.1401100
39: 9.0374088, 30.9781036, 8.9985371, 30.9783039, -14.9723167, 15.0112953
40: 6.9503088, 25.1672058, 6.9270830, 25.1679230, -12.6935844, 12.7163792
41: -5.0884275, 10.3590660, -5.1036711, 10.3613434, -11.3960838, 11.4101982
42: -16.9133511, 0.0282938, -16.9162006, 0.0329356, -11.3737488, 11.3741035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=103, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1405

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1746

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4581897, upper bound: 12.5179832
time: 5.94 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4581897, upper bound: 12.5175237
time: 8.48 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -42.6922722, -18.9390106, -42.6871758, -18.9411545, -15.8627739, 15.8563194
1: -30.4977837, -12.9628344, -30.4738178, -12.9638357, -11.8463707, 11.8189793
2: -21.5925121, -5.9877138, -21.5933914, -5.9927282, -10.7144394, 10.7217369
3: -30.5871735, -12.4182625, -30.5873966, -12.4457026, -12.8101196, 12.8428955
4: -18.8490562, 0.6111374, -18.8442612, 0.6083457, -12.7012062, 12.6794586
5: -30.7252674, -9.8916035, -30.7283115, -9.8969402, -14.2465363, 14.2531395
6: -14.2932158, 1.3367052, -14.2877712, 1.3314168, -11.0511341, 11.0570202
7: -47.5224800, -25.9919128, -47.4974251, -25.9942703, -13.1161003, 13.0888405
8: -33.6761703, -12.1337109, -33.6749649, -12.1372910, -13.9952850, 13.9977036
9: -18.6125984, -6.9814763, -18.5996895, -6.9898558, -9.7859688, 9.7713547
10: -38.4047241, -15.4160328, -38.3483124, -15.4221172, -18.3064575, 18.2452049
11: -57.4552689, -34.7042236, -57.3800201, -34.7040863, -13.4253941, 13.3435287
12: -1.9936042, 17.9376793, -1.9402912, 17.9373989, -16.1542664, 16.0923080
13: -8.2650700, 9.4690266, -8.2632151, 9.4563007, -14.8739052, 14.8820190
14: -79.1422501, -46.7450638, -79.0527344, -46.7460785, -25.4131851, 25.3092041
15: -11.0552864, 7.8009605, -11.0540924, 7.7988615, -13.1615524, 13.1519413
16: -49.1848373, -29.5158310, -49.1395111, -29.5197067, -12.8285828, 12.7743988
17: -79.7510223, -45.0341721, -79.6459732, -45.0323410, -22.3002777, 22.1728745
18: -11.6331959, 9.5350838, -11.5790691, 9.5345955, -19.2936783, 19.2363892
19: -27.6123314, -12.5071030, -27.5892658, -12.5083618, -10.6319370, 10.6100807
20: -17.4774551, -4.3143673, -17.4590816, -4.3167124, -12.4567451, 12.4428520
21: -39.8520737, -20.1224365, -39.7972450, -20.1256695, -14.1707878, 14.1130180
22: -17.7118797, 0.3113537, -17.6743755, 0.3115811, -12.8331413, 12.7906475
23: -22.3004837, -7.3989034, -22.2934303, -7.4000716, -11.1884460, 11.1877251
24: -8.7317228, 4.7527857, -8.7163782, 4.7530422, -9.3684521, 9.3534069
25: -5.7868452, 8.8895931, -5.7725511, 8.8890095, -12.0283813, 12.0165215
26: -19.2294273, 2.4266784, -19.1807041, 2.4267650, -18.8639069, 18.8095169
27: -23.7465782, -7.1743989, -23.7050838, -7.1744189, -11.9925995, 11.9468536
28: -18.3417816, -0.1322846, -18.3384018, -0.1334128, -12.5184555, 12.5236130
29: -36.4782982, -18.7433243, -36.4318504, -18.7427425, -11.6888733, 11.6415253
30: -27.9382019, -7.8021612, -27.8886452, -7.8041596, -12.6418629, 12.5851135
31: -23.2143688, -7.2155504, -23.1909485, -7.2160044, -12.2469482, 12.2213402
32: -4.6373000, 10.3793478, -4.6347570, 10.3700218, -11.9954185, 12.0088463
33: 17.4487896, 40.2953339, 17.4529037, 40.2535133, -17.7354431, 17.7770844
34: 6.0322089, 25.5430641, 6.0359054, 25.5070610, -13.5434265, 13.5791473
35: 10.2623940, 29.9795876, 10.2653570, 29.9234371, -15.6250610, 15.6825714
36: 5.8039412, 23.6659241, 5.8078351, 23.6289558, -12.5900917, 12.6281891
37: 13.6619053, 32.5473709, 13.6718502, 32.5388680, -13.6406059, 13.6393852
38: 4.0615377, 25.2588120, 4.0698152, 25.2221870, -17.1589890, 17.1904526
39: 8.9868107, 31.0088406, 8.9886036, 30.9781876, -15.0190506, 15.0513992
40: 6.9146342, 25.1902351, 6.9215474, 25.1680164, -12.7269745, 12.7493744
41: -5.1100087, 10.3747053, -5.1076069, 10.3618050, -11.4181442, 11.4308758
42: -16.9201221, 0.0337157, -16.9169559, 0.0312424, -11.3788509, 11.3851242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=103, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=141, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1405

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1746

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4974957, upper bound: 12.5181078
time: 13.76 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5176594, upper bound: 12.5176598
time: 5.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.74 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.74
Output dim: 33, lower bound: -12.4581897, upper bound: 12.5179832
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.74
Output dim: 33, lower bound: -12.4581897, upper bound: 12.5175237
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.74
Output dim: 33, lower bound: -12.4974957, upper bound: 12.5181078
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.74
Output dim: 33, lower bound: -12.5176594, upper bound: 12.5176598

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -42.6629066, -18.9546356, -42.6245346, -18.9455185, -15.8258209, 15.7778015
1: -30.4587460, -12.9872465, -30.4326744, -12.9703579, -11.7983398, 11.7550278
2: -21.5743504, -5.9986897, -21.5554581, -5.9953718, -10.6914062, 10.6723633
3: -30.5630264, -12.4535923, -30.5541229, -12.4493332, -12.7844162, 12.7722168
4: -18.8206539, 0.6011069, -18.7876530, 0.6055942, -12.6567497, 12.6150284
5: -30.7107697, -9.9068031, -30.6916294, -9.9025154, -14.2201157, 14.2003021
6: -14.2750959, 1.3182139, -14.2826853, 1.2999663, -11.0007248, 11.0299072
7: -47.4834442, -26.0145054, -47.4591675, -25.9988956, -13.0692940, 13.0288506
8: -33.6510201, -12.1467495, -33.6118851, -12.1414862, -13.9629135, 13.9195709
9: -18.5875721, -7.0125732, -18.5827484, -6.9968266, -9.7500763, 9.7262516
10: -38.3390160, -15.4767361, -38.3284912, -15.4378834, -18.2228088, 18.1727791
11: -57.3760071, -34.7617874, -57.3754807, -34.7343292, -13.3195972, 13.2929916
12: -1.9330518, 17.8995781, -1.9362190, 17.9063435, -16.0603638, 16.0556297
13: -8.2265177, 9.4428930, -8.2405901, 9.4497967, -14.8322792, 14.8440285
14: -79.0327454, -46.8065872, -79.0102539, -46.7635155, -25.2837601, 25.2182846
15: -11.0381393, 7.7818260, -11.0099077, 7.7930074, -13.1325150, 13.0883884
16: -49.1343651, -29.5641708, -49.1352882, -29.5375729, -12.7597427, 12.7295570
17: -79.6318054, -45.0960312, -79.6172943, -45.0496292, -22.1536636, 22.0926323
18: -11.5669928, 9.4966793, -11.5734177, 9.5114918, -19.2042618, 19.1942062
19: -27.5792160, -12.5286999, -27.5816956, -12.5206985, -10.5890636, 10.5876942
20: -17.4488640, -4.3356242, -17.4518623, -4.3318958, -12.4165268, 12.4174423
21: -39.7869949, -20.1658707, -39.7890816, -20.1480045, -14.0811424, 14.0697441
22: -17.6673431, 0.2839355, -17.6689034, 0.3015509, -12.7762222, 12.7607441
23: -22.2844791, -7.4262843, -22.2870102, -7.4279189, -11.1494179, 11.1566868
24: -8.7083549, 4.7333488, -8.7112865, 4.7368951, -9.3330784, 9.3330116
25: -5.7613444, 8.8597555, -5.7657881, 8.8683014, -11.9870052, 11.9838142
26: -19.1692314, 2.3883915, -19.1744957, 2.4064398, -18.7845840, 18.7699509
27: -23.6984196, -7.2095809, -23.7011147, -7.2022219, -11.9163666, 11.9124336
28: -18.3282471, -0.1514854, -18.3338375, -0.1630850, -12.4756413, 12.4995003
29: -36.4290695, -18.7814198, -36.4288177, -18.7578259, -11.6314888, 11.6082573
30: -27.8842411, -7.8498654, -27.8850632, -7.8457084, -12.5443153, 12.5418034
31: -23.1796246, -7.2394695, -23.1821918, -7.2343354, -12.1909370, 12.1929588
32: -4.6173396, 10.3622808, -4.6277790, 10.3501921, -11.9580994, 11.9813690
33: 17.5121765, 40.2456856, 17.4759636, 40.2343102, -17.6571617, 17.7052612
34: 6.0697846, 25.4968491, 6.0470228, 25.4786758, -13.4792633, 13.5200710
35: 10.3158970, 29.9127960, 10.2796078, 29.8975315, -15.5487366, 15.6010246
36: 5.8453894, 23.6198616, 5.8181944, 23.6028786, -12.5254593, 12.5703125
37: 13.7026339, 32.5296021, 13.6890850, 32.5145340, -13.5743351, 13.6030655
38: 4.1229892, 25.2193890, 4.0872884, 25.2193546, -17.0931396, 17.1317024
39: 9.0414162, 30.9744492, 9.0109787, 30.9671364, -14.9571495, 14.9975510
40: 6.9538636, 25.1602211, 6.9382334, 25.1461411, -12.6679401, 12.6980495
41: -5.0866976, 10.3509531, -5.0982442, 10.3360205, -11.3691082, 11.3953571
42: -16.9116974, 0.0240455, -16.9110260, 0.0196791, -11.3599987, 11.3654938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=102, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1405

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 573

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4114494, upper bound: 12.5123566
time: 7.75 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4575234, upper bound: 12.5173207
time: 5.62 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -42.6808777, -18.9539948, -42.6838303, -18.8823853, -15.9082031, 15.8262444
1: -30.4706459, -12.9866552, -30.4710732, -12.9201803, -11.8620872, 11.7852821
2: -21.5853271, -5.9982071, -21.5919170, -5.9580355, -10.7409935, 10.7022667
3: -30.5719299, -12.4530029, -30.5832310, -12.4102917, -12.8344727, 12.7949600
4: -18.8367844, 0.6015594, -18.8440285, 0.6582143, -12.7280807, 12.6611996
5: -30.7217407, -9.9056568, -30.7282982, -9.8529558, -14.2816620, 14.2294273
6: -14.2758846, 1.3273525, -14.3256149, 1.3313046, -11.0284214, 11.0842781
7: -47.4945183, -26.0141640, -47.4963074, -25.9529152, -13.1281662, 13.0570488
8: -33.6693230, -12.1459970, -33.6720390, -12.0701332, -14.0557938, 13.9661446
9: -18.5921478, -7.0117006, -18.5984573, -6.9600682, -9.7921600, 9.7399902
10: -38.3443756, -15.4753294, -38.3481903, -15.3875866, -18.2796974, 18.1918411
11: -57.3772278, -34.7567825, -57.3949890, -34.7132721, -13.3397484, 13.3182335
12: -1.9338691, 17.9068680, -1.9625452, 17.9355659, -16.0871811, 16.0906639
13: -8.2313499, 9.4440536, -8.2754955, 9.4957533, -14.8810730, 14.8841591
14: -79.0454865, -46.8050995, -79.0496826, -46.6999092, -25.3547592, 25.2472687
15: -11.0512238, 7.7825608, -11.0566540, 7.8551779, -13.2089310, 13.1274834
16: -49.1353989, -29.5615120, -49.1461258, -29.5130539, -12.7948990, 12.7393799
17: -79.6403656, -45.0948067, -79.6459732, -45.0000610, -22.2140656, 22.1158333
18: -11.5679951, 9.5013266, -11.6130171, 9.5329685, -19.2253571, 19.2390518
19: -27.5809364, -12.5265551, -27.6195755, -12.5101871, -10.5980263, 10.6293068
20: -17.4504318, -4.3323135, -17.4844475, -4.3190999, -12.4295349, 12.4532700
21: -39.7889099, -20.1622200, -39.8244743, -20.1303635, -14.1004944, 14.1117859
22: -17.6686401, 0.2852602, -17.6852741, 0.3105288, -12.7865982, 12.7822380
23: -22.2858963, -7.4192553, -22.3269672, -7.3998394, -11.1734772, 11.2055244
24: -8.7094221, 4.7371168, -8.7456703, 4.7495904, -9.3431435, 9.3719120
25: -5.7627034, 8.8642435, -5.8066106, 8.8920174, -12.0094604, 12.0296936
26: -19.1704407, 2.3925061, -19.2140179, 2.4252043, -18.8042297, 18.8121719
27: -23.6992340, -7.2030501, -23.7345181, -7.1769352, -11.9397354, 11.9542351
28: -18.3290081, -0.1433225, -18.3807011, -0.1317167, -12.5015736, 12.5566750
29: -36.4296074, -18.7793808, -36.4338684, -18.7434349, -11.6455612, 11.6146851
30: -27.8852119, -7.8398919, -27.9218407, -7.8073435, -12.5765114, 12.5904121
31: -23.1816063, -7.2358351, -23.2267170, -7.2206316, -12.2049637, 12.2433510
32: -4.6184530, 10.3683109, -4.6650853, 10.3713579, -11.9770355, 12.0255089
33: 17.5085773, 40.2511597, 17.3943901, 40.2517090, -17.6715126, 17.7920074
34: 6.0682464, 25.5050812, 5.9933958, 25.5053101, -13.5019531, 13.5819187
35: 10.3145237, 29.9202118, 10.2183065, 29.9219990, -15.5687180, 15.6696358
36: 5.8445907, 23.6274586, 5.7588086, 23.6277695, -12.5468025, 12.6384201
37: 13.6991854, 32.5365829, 13.6292114, 32.5366440, -13.5935230, 13.6700783
38: 4.1211090, 25.2201080, 4.0289617, 25.2226906, -17.0983925, 17.2000084
39: 9.0377178, 30.9777012, 8.9336700, 30.9776211, -14.9680519, 15.0797615
40: 6.9506121, 25.1665916, 6.8846788, 25.1677952, -12.6901970, 12.7602406
41: -5.0882750, 10.3583088, -5.1397114, 10.3603172, -11.3910866, 11.4455090
42: -16.9131966, 0.0277781, -16.9377289, 0.0365813, -11.3773289, 11.3965626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=102, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=141, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1405

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 573

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4317174, upper bound: 12.5119171
time: 17.22 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4777135, upper bound: 12.5168605
time: 15.74 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -42.6724777, -18.9397240, -42.6253929, -18.9433956, -15.8408051, 15.7932396
1: -30.4847584, -12.9635334, -30.4330521, -12.9660492, -11.8312302, 11.7772312
2: -21.5804558, -5.9883003, -21.5557232, -5.9945173, -10.7003517, 10.6831284
3: -30.5773411, -12.4189510, -30.5570965, -12.4479074, -12.7977829, 12.8108749
4: -18.8313141, 0.6106393, -18.7888107, 0.6068156, -12.6816254, 12.6231079
5: -30.7132454, -9.8928738, -30.6906471, -9.9009714, -14.2304916, 14.2140808
6: -14.2923269, 1.3266189, -14.2850037, 1.3005042, -11.0196743, 11.0449390
7: -47.5103531, -25.9923229, -47.4594421, -25.9955635, -13.1026192, 13.0502815
8: -33.6560822, -12.1345558, -33.6120949, -12.1399288, -13.9724197, 13.9334908
9: -18.6075535, -6.9824553, -18.5841866, -6.9928899, -9.7783012, 9.7548103
10: -38.3984070, -15.4175138, -38.3288574, -15.4267406, -18.2955818, 18.2248840
11: -57.4539833, -34.7105179, -57.3759384, -34.7237396, -13.4064884, 13.3341961
12: -1.9926054, 17.9297161, -1.9371839, 17.9126244, -16.1274261, 16.0810776
13: -8.2598457, 9.4676914, -8.2470379, 9.4522810, -14.8660278, 14.8692627
14: -79.1284561, -46.7466965, -79.0112228, -46.7511482, -25.3935928, 25.2677231
15: -11.0409622, 7.8000989, -11.0092525, 7.7962418, -13.1450844, 13.1056252
16: -49.1837463, -29.5188122, -49.1360474, -29.5291424, -12.8213272, 12.7682781
17: -79.7417068, -45.0354462, -79.6178741, -45.0365334, -22.2868805, 22.1434441
18: -11.6320896, 9.5297565, -11.5755491, 9.5181446, -19.2773132, 19.2276306
19: -27.6104336, -12.5098982, -27.5833664, -12.5171270, -10.6245003, 10.6053085
20: -17.4757118, -4.3183951, -17.4535561, -4.3290305, -12.4428596, 12.4347687
21: -39.8499527, -20.1272392, -39.7906303, -20.1407261, -14.1533222, 14.1038857
22: -17.7103920, 0.3099723, -17.6697636, 0.3070602, -12.8268738, 12.7848415
23: -22.2989082, -7.4065952, -22.2885036, -7.4241762, -11.1628380, 11.1751614
24: -8.7305298, 4.7486658, -8.7126493, 4.7401361, -9.3553429, 9.3462772
25: -5.7853017, 8.8845282, -5.7677684, 8.8733749, -12.0137520, 12.0089455
26: -19.2280655, 2.4222264, -19.1764603, 2.4133992, -18.8495941, 18.8009872
27: -23.7456360, -7.1815672, -23.7021465, -7.1967134, -11.9688988, 11.9367943
28: -18.3409348, -0.1412182, -18.3357620, -0.1613445, -12.4891701, 12.5127220
29: -36.4773483, -18.7456741, -36.4290390, -18.7501354, -11.6837292, 11.6372299
30: -27.9371681, -7.8131323, -27.8853722, -7.8384404, -12.6061020, 12.5712700
31: -23.2121868, -7.2201872, -23.1840515, -7.2305827, -12.2296486, 12.2120895
32: -4.6359863, 10.3730602, -4.6306505, 10.3503866, -11.9735069, 11.9983597
33: 17.4526558, 40.2892761, 17.4650059, 40.2346268, -17.7134132, 17.7591476
34: 6.0340004, 25.5340652, 6.0414481, 25.4789124, -13.5131607, 13.5645237
35: 10.2639246, 29.9714050, 10.2701063, 29.8979874, -15.5983887, 15.6691055
36: 5.8047872, 23.6576004, 5.8105583, 23.6030045, -12.5626259, 12.6171188
37: 13.6657372, 32.5396996, 13.6838093, 32.5147896, -13.6119423, 13.6182365
38: 4.0637093, 25.2579269, 4.0766439, 25.2194996, -17.1481819, 17.1820602
39: 8.9908237, 31.0051804, 9.0010300, 30.9670277, -15.0038757, 15.0376396
40: 6.9181867, 25.1832809, 6.9326620, 25.1462440, -12.7013226, 12.7310658
41: -5.1082754, 10.3666029, -5.1021891, 10.3365145, -11.3911934, 11.4160194
42: -16.9184723, 0.0294826, -16.9117584, 0.0179939, -11.3651161, 11.3765144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=102, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=141, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1405

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 573

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4514734, upper bound: 12.5127534
time: 12.81 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4968206, upper bound: 12.5174349
time: 17.42 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -42.6905403, -18.9390907, -42.6846390, -18.8803043, -15.9231949, 15.8416634
1: -30.4966373, -12.9629393, -30.4714298, -12.9158115, -11.8949928, 11.8074894
2: -21.5914440, -5.9878049, -21.5921707, -5.9571829, -10.7499237, 10.7130394
3: -30.5862808, -12.4183578, -30.5861969, -12.4088135, -12.8478470, 12.8336487
4: -18.8474922, 0.6110380, -18.8451691, 0.6594553, -12.7529449, 12.6692791
5: -30.7241783, -9.8917141, -30.7273293, -9.8513851, -14.2919884, 14.2431984
6: -14.2931299, 1.3357806, -14.3279409, 1.3318415, -11.0473804, 11.0993156
7: -47.5214157, -25.9919662, -47.4966087, -25.9495659, -13.1614876, 13.0784798
8: -33.6743965, -12.1338310, -33.6722565, -12.0685778, -14.0653038, 13.9800682
9: -18.6121311, -6.9815726, -18.5998840, -6.9561534, -9.8203926, 9.7685528
10: -38.4037437, -15.4162025, -38.3485718, -15.3764515, -18.3524742, 18.2439613
11: -57.4551392, -34.7055397, -57.3954277, -34.7026978, -13.4266396, 13.3594379
12: -1.9934573, 17.9370747, -1.9635205, 17.9418564, -16.1542130, 16.1161194
13: -8.2646360, 9.4688950, -8.2819538, 9.4982128, -14.9148178, 14.9093971
14: -79.1411896, -46.7452774, -79.0506897, -46.6875610, -25.4645996, 25.2966919
15: -11.0540209, 7.8008413, -11.0560389, 7.8584013, -13.2215118, 13.1447296
16: -49.1847801, -29.5161247, -49.1468430, -29.5046082, -12.8564873, 12.7780819
17: -79.7503357, -45.0342712, -79.6465302, -44.9869080, -22.3472748, 22.1666145
18: -11.6330242, 9.5344324, -11.6151400, 9.5396175, -19.2984161, 19.2724609
19: -27.6121655, -12.5077648, -27.6212273, -12.5066175, -10.6334782, 10.6469002
20: -17.4772682, -4.3151007, -17.4861660, -4.3162184, -12.4558907, 12.4706116
21: -39.8518829, -20.1236210, -39.8260460, -20.1230907, -14.1726875, 14.1459351
22: -17.7116928, 0.3112378, -17.6861534, 0.3160415, -12.8372536, 12.8063297
23: -22.3003330, -7.3995724, -22.3284569, -7.3960695, -11.1869125, 11.2240067
24: -8.7316065, 4.7524405, -8.7470570, 4.7528057, -9.3654118, 9.3851604
25: -5.7866998, 8.8890266, -5.8085670, 8.8970737, -12.0362549, 12.0548210
26: -19.2292404, 2.4262612, -19.2159634, 2.4321284, -18.8692398, 18.8431549
27: -23.7464676, -7.1749973, -23.7355690, -7.1713982, -11.9922600, 11.9786034
28: -18.3417130, -0.1330619, -18.3826294, -0.1299648, -12.5151196, 12.5698929
29: -36.4779816, -18.7436295, -36.4340363, -18.7357903, -11.6978092, 11.6436672
30: -27.9381142, -7.8031311, -27.9221611, -7.8000650, -12.6382942, 12.6198635
31: -23.2141705, -7.2165742, -23.2285728, -7.2168760, -12.2436714, 12.2624969
32: -4.6371360, 10.3790932, -4.6679597, 10.3715420, -11.9924469, 12.0424881
33: 17.4490929, 40.2947388, 17.3834362, 40.2520332, -17.7277794, 17.8459244
34: 6.0324354, 25.5422821, 5.9878101, 25.5055466, -13.5358276, 13.6263618
35: 10.2625351, 29.9788208, 10.2087955, 29.9224777, -15.6183510, 15.7376671
36: 5.8040233, 23.6652164, 5.7511873, 23.6279068, -12.5839596, 12.6852188
37: 13.6622725, 32.5466576, 13.6239548, 32.5369263, -13.6311207, 13.6852417
38: 4.0618782, 25.2586498, 4.0183287, 25.2228394, -17.1534500, 17.2503433
39: 8.9871626, 31.0084305, 8.9237003, 30.9774837, -15.0147934, 15.1198540
40: 6.9149251, 25.1896286, 6.8791399, 25.1678944, -12.7235794, 12.7932434
41: -5.1098509, 10.3739662, -5.1436424, 10.3608265, -11.4131584, 11.4661808
42: -16.9200172, 0.0331984, -16.9384823, 0.0349202, -11.3824463, 11.4075851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=102, inp2_unstable=102, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=141, inp2_unstable=141, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1005
type: A, layer: 1, pos: 1491
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1493
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1422
type: A, layer: 1, pos: 1021
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 1413
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1487
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1438
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1475
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1471
type: A, layer: 1, pos: 1405

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 573

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4716808, upper bound: 12.5123138
time: 24.80 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5169923, upper bound: 12.5169930
time: 6.57 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 33.63 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 33.63
Output dim: 33, lower bound: -12.4114494, upper bound: 12.5123566
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.63
Output dim: 33, lower bound: -12.4575234, upper bound: 12.5173207
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 33.63
Output dim: 33, lower bound: -12.4317174, upper bound: 12.5119171
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.63
Output dim: 33, lower bound: -12.4777135, upper bound: 12.5168605
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 33.63
Output dim: 33, lower bound: -12.4514734, upper bound: 12.5127534
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.63
Output dim: 33, lower bound: -12.4968206, upper bound: 12.5174349
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 33.63
Output dim: 33, lower bound: -12.4716808, upper bound: 12.5123138
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.63
Output dim: 33, lower bound: -12.5169923, upper bound: 12.5169930

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -42.6619263, -18.9562836, -42.6243820, -18.9457970, -15.8325119, 15.7746925
1: -30.4583130, -12.9899931, -30.4325981, -12.9708548, -11.7973061, 11.7461166
2: -21.5722122, -5.9994974, -21.5550308, -5.9955416, -10.6786804, 10.6714973
3: -30.5614853, -12.4542637, -30.5538139, -12.4494400, -12.7532616, 12.7712631
4: -18.8166733, 0.6005006, -18.7869320, 0.6054590, -12.6815567, 12.6102753
5: -30.7094097, -9.9078598, -30.6913605, -9.9027166, -14.1780930, 14.1991081
6: -14.2736235, 1.3175290, -14.2824259, 1.2998533, -10.9977665, 11.0321293
7: -47.4766693, -26.0155830, -47.4579239, -25.9991264, -13.0921211, 13.0254135
8: -33.6464424, -12.1478338, -33.6109581, -12.1416759, -13.9796104, 13.9164391
9: -18.5868530, -7.0144506, -18.5826073, -6.9973040, -9.7491379, 9.7182026
10: -38.3383789, -15.4793587, -38.3283615, -15.4383774, -18.2215042, 18.1674271
11: -57.3754196, -34.7636528, -57.3753738, -34.7346802, -13.3185692, 13.2661438
12: -1.9317746, 17.8971729, -1.9359899, 17.9059086, -16.0587158, 16.0022507
13: -8.2234879, 9.4420300, -8.2399969, 9.4496269, -14.8051834, 14.8424492
14: -79.0314331, -46.8093910, -79.0099716, -46.7639923, -25.2920990, 25.2138672
15: -11.0373993, 7.7793159, -11.0097647, 7.7925816, -13.1333847, 13.0852909
16: -49.1323204, -29.5654259, -49.1349106, -29.5378227, -12.7662888, 12.7270451
17: -79.6302948, -45.0984840, -79.6170425, -45.0500488, -22.1511002, 22.0872536
18: -11.5657930, 9.4941072, -11.5731916, 9.5109825, -19.2025375, 19.1889343
19: -27.5773926, -12.5292730, -27.5813656, -12.5208378, -10.5871429, 10.5864563
20: -17.4473305, -4.3365612, -17.4515648, -4.3320837, -12.4298325, 12.4149475
21: -39.7861328, -20.1671715, -39.7889175, -20.1482124, -14.0798454, 14.0624428
22: -17.6663208, 0.2824273, -17.6686954, 0.3012562, -12.7748604, 12.7462158
23: -22.2838268, -7.4275422, -22.2868576, -7.4281549, -11.1483803, 11.1477051
24: -8.7076416, 4.7322664, -8.7111416, 4.7367015, -9.3320923, 9.3261070
25: -5.7605052, 8.8585987, -5.7656336, 8.8680992, -11.9866753, 11.9822273
26: -19.1681442, 2.3855023, -19.1742973, 2.4058990, -18.7829514, 18.7414856
27: -23.6974220, -7.2117305, -23.7009277, -7.2026534, -11.9148941, 11.8904114
28: -18.3275166, -0.1522346, -18.3337250, -0.1632133, -12.4738274, 12.4970245
29: -36.4285545, -18.7831879, -36.4287338, -18.7581081, -11.6304722, 11.5783043
30: -27.8831005, -7.8510885, -27.8848381, -7.8459368, -12.5428352, 12.5304298
31: -23.1752968, -7.2400494, -23.1813622, -7.2344556, -12.1918659, 12.1914597
32: -4.6158924, 10.3581171, -4.6275072, 10.3493977, -11.9550056, 11.9857864
33: 17.5166817, 40.2453842, 17.4768314, 40.2342720, -17.6429520, 17.7041473
34: 6.0734997, 25.4959030, 6.0477057, 25.4784985, -13.4747963, 13.5213051
35: 10.3200397, 29.9125271, 10.2804089, 29.8974648, -15.5409470, 15.6000557
36: 5.8476319, 23.6196404, 5.8186398, 23.6028290, -12.5219955, 12.5729485
37: 13.7052498, 32.5285072, 13.6895733, 32.5143166, -13.5695343, 13.6183929
38: 4.1257253, 25.2120113, 4.0878005, 25.2179184, -17.0885162, 17.1266632
39: 9.0454597, 30.9739647, 9.0117464, 30.9670486, -14.9383392, 14.9963112
40: 6.9565787, 25.1598225, 6.9387693, 25.1460800, -12.6633873, 12.6970177
41: -5.0854306, 10.3480873, -5.0979986, 10.3355055, -11.3640194, 11.4184952
42: -16.9111214, 0.0164814, -16.9109097, 0.0182648, -11.3562908, 11.3812675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=101, inp2_unstable=102, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1405

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 667

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.3994916, upper bound: 12.5054413
time: 6.38 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4392382, upper bound: 12.4990313
time: 12.85 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -42.6799545, -18.9556885, -42.6836243, -18.8827095, -15.9149170, 15.8231392
1: -30.4701824, -12.9894094, -30.4709854, -12.9206724, -11.8610649, 11.7763863
2: -21.5831928, -5.9989853, -21.5914898, -5.9582071, -10.7283058, 10.7013969
3: -30.5703926, -12.4536858, -30.5829124, -12.4104233, -12.8033676, 12.7940102
4: -18.8328285, 0.6009359, -18.8432865, 0.6580873, -12.7528801, 12.6564522
5: -30.7203808, -9.9067287, -30.7280331, -9.8531609, -14.2396240, 14.2282143
6: -14.2744360, 1.3267202, -14.3253527, 1.3311753, -11.0254688, 11.0865002
7: -47.4877205, -26.0152454, -47.4950905, -25.9531021, -13.1510048, 13.0536079
8: -33.6647339, -12.1471205, -33.6711426, -12.0703468, -14.0725098, 13.9630089
9: -18.5914459, -7.0135632, -18.5983162, -6.9605217, -9.7912292, 9.7319279
10: -38.3436584, -15.4780045, -38.3480492, -15.3880749, -18.2783966, 18.1864662
11: -57.3765907, -34.7586365, -57.3948822, -34.7136269, -13.3387318, 13.2913799
12: -1.9325750, 17.9045105, -1.9623258, 17.9351406, -16.0855103, 16.0372772
13: -8.2283058, 9.4431763, -8.2749310, 9.4955893, -14.8539925, 14.8825836
14: -79.0441742, -46.8078690, -79.0494232, -46.7004089, -25.3631210, 25.2428360
15: -11.0504265, 7.7800732, -11.0565271, 7.8547182, -13.2097855, 13.1244049
16: -49.1333771, -29.5627060, -49.1457443, -29.5133343, -12.8014641, 12.7368450
17: -79.6389313, -45.0973434, -79.6457520, -45.0004616, -22.2115021, 22.1103973
18: -11.5667391, 9.4986992, -11.6128139, 9.5324631, -19.2236328, 19.2337494
19: -27.5790691, -12.5271368, -27.6192322, -12.5102901, -10.5961075, 10.6280556
20: -17.4489040, -4.3332629, -17.4841442, -4.3192773, -12.4428864, 12.4507828
21: -39.7880554, -20.1635590, -39.8243294, -20.1306133, -14.0991936, 14.1044941
22: -17.6676292, 0.2837310, -17.6850891, 0.3102584, -12.7852211, 12.7677155
23: -22.2852516, -7.4204969, -22.3268223, -7.4000664, -11.1724548, 11.1965485
24: -8.7087059, 4.7360220, -8.7455320, 4.7493792, -9.3421307, 9.3650036
25: -5.7618675, 8.8630829, -5.8064547, 8.8917913, -12.0091400, 12.0281143
26: -19.1693535, 2.3895769, -19.2138062, 2.4246702, -18.8025894, 18.7837143
27: -23.6982651, -7.2052002, -23.7343445, -7.1773410, -11.9382553, 11.9322166
28: -18.3283138, -0.1440649, -18.3805866, -0.1318493, -12.4997673, 12.5541840
29: -36.4291000, -18.7811069, -36.4337654, -18.7437859, -11.6445332, 11.5847435
30: -27.8840408, -7.8411059, -27.9216251, -7.8075681, -12.5750427, 12.5790195
31: -23.1772366, -7.2364130, -23.2258797, -7.2207384, -12.2058849, 12.2418518
32: -4.6170387, 10.3641453, -4.6647992, 10.3705416, -11.9739723, 12.0299454
33: 17.5131378, 40.2509003, 17.3952427, 40.2516632, -17.6573029, 17.7908821
34: 6.0719647, 25.5041351, 5.9940724, 25.5051250, -13.4974594, 13.5831604
35: 10.3187027, 29.9199390, 10.2191029, 29.9219341, -15.5609093, 15.6686172
36: 5.8469076, 23.6272449, 5.7592478, 23.6277447, -12.5433121, 12.6410751
37: 13.7017632, 32.5354958, 13.6297216, 32.5364304, -13.5886993, 13.6854019
38: 4.1238580, 25.2127151, 4.0294790, 25.2212448, -17.0937767, 17.1949806
39: 9.0418186, 30.9772148, 8.9344387, 30.9775352, -14.9492340, 15.0785179
40: 6.9533010, 25.1662064, 6.8852267, 25.1677208, -12.6856308, 12.7592030
41: -5.0870028, 10.3554535, -5.1394606, 10.3598022, -11.3859940, 11.4686604
42: -16.9126415, 0.0202055, -16.9376278, 0.0352085, -11.3736172, 11.4123344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=101, inp2_unstable=102, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=141, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1405

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 667

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4196514, upper bound: 12.5049659
time: 5.81 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4594218, upper bound: 12.4985685
time: 24.13 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -42.6715164, -18.9414215, -42.6251831, -18.9437065, -15.8475266, 15.7901268
1: -30.4842949, -12.9662743, -30.4329491, -12.9665432, -11.8302002, 11.7683334
2: -21.5783367, -5.9890642, -21.5552940, -5.9946489, -10.6876030, 10.6822586
3: -30.5758400, -12.4196272, -30.5568085, -12.4480305, -12.7666779, 12.8099251
4: -18.8272972, 0.6100125, -18.7880611, 0.6067026, -12.7064209, 12.6183605
5: -30.7118988, -9.8939447, -30.6903534, -9.9011583, -14.1884613, 14.2128677
6: -14.2908792, 1.3259902, -14.2847271, 1.3003888, -11.0167103, 11.0471516
7: -47.5035629, -25.9933815, -47.4581795, -25.9957771, -13.1254425, 13.0468254
8: -33.6515541, -12.1356363, -33.6111946, -12.1401291, -13.9890671, 13.9303513
9: -18.6068516, -6.9843011, -18.5840588, -6.9933510, -9.7773933, 9.7467537
10: -38.3977737, -15.4202080, -38.3287430, -15.4272385, -18.2943039, 18.2195129
11: -57.4533768, -34.7124138, -57.3758240, -34.7241211, -13.4054909, 13.3073349
12: -1.9913034, 17.9273586, -1.9369104, 17.9122009, -16.1257248, 16.0276794
13: -8.2567892, 9.4668274, -8.2464676, 9.4520798, -14.8389397, 14.8676910
14: -79.1271744, -46.7494812, -79.0109863, -46.7516785, -25.4019699, 25.2632980
15: -11.0401897, 7.7976098, -11.0091105, 7.7957759, -13.1459694, 13.1025448
16: -49.1817703, -29.5200329, -49.1356392, -29.5293560, -12.8278427, 12.7657528
17: -79.7402954, -45.0380096, -79.6175766, -45.0369682, -22.2843094, 22.1380157
18: -11.6308937, 9.5271683, -11.5753241, 9.5176573, -19.2755661, 19.2223053
19: -27.6086102, -12.5104704, -27.5830021, -12.5172272, -10.6225891, 10.6040726
20: -17.4741936, -4.3193526, -17.4532967, -4.3292241, -12.4561768, 12.4322891
21: -39.8491058, -20.1285648, -39.7904587, -20.1409798, -14.1520233, 14.0965919
22: -17.7093716, 0.3084021, -17.6695671, 0.3067799, -12.8255043, 12.7702961
23: -22.2982292, -7.4078598, -22.2883701, -7.4244061, -11.1618156, 11.1661949
24: -8.7298193, 4.7475781, -8.7125378, 4.7399192, -9.3543434, 9.3393555
25: -5.7844529, 8.8833771, -5.7675929, 8.8731642, -12.0134468, 12.0073624
26: -19.2269955, 2.4192643, -19.1762505, 2.4128046, -18.8479538, 18.7725220
27: -23.7446537, -7.1837177, -23.7019596, -7.1971092, -11.9674339, 11.9147720
28: -18.3401966, -0.1419640, -18.3356056, -0.1614752, -12.4873619, 12.5102386
29: -36.4768562, -18.7474365, -36.4289246, -18.7504463, -11.6827087, 11.6072960
30: -27.9360123, -7.8143578, -27.8851585, -7.8386722, -12.6046219, 12.5599041
31: -23.2078590, -7.2207651, -23.1832237, -7.2306652, -12.2305622, 12.2105904
32: -4.6345425, 10.3688784, -4.6303639, 10.3495922, -11.9704170, 12.0027771
33: 17.4572334, 40.2889862, 17.4659119, 40.2345657, -17.6991882, 17.7580414
34: 6.0376902, 25.5331154, 6.0421238, 25.4787445, -13.5086594, 13.5657387
35: 10.2680702, 29.9711571, 10.2709093, 29.8979263, -15.5905685, 15.6681137
36: 5.8070555, 23.6573849, 5.8109908, 23.6029530, -12.5591297, 12.6197662
37: 13.6683750, 32.5385895, 13.6843090, 32.5145912, -13.6071243, 13.6335602
38: 4.0664597, 25.2505302, 4.0771918, 25.2180634, -17.1435394, 17.1769867
39: 8.9949265, 31.0046844, 9.0017796, 30.9669418, -14.9850769, 15.0363846
40: 6.9208984, 25.1828823, 6.9332452, 25.1461678, -12.6967659, 12.7300320
41: -5.1070137, 10.3637161, -5.1019602, 10.3359699, -11.3860817, 11.4391785
42: -16.9179192, 0.0218956, -16.9116554, 0.0165708, -11.3613987, 11.3922863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=101, inp2_unstable=102, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=141, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1405

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 667

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4387753, upper bound: 12.5055593
time: 28.50 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4785320, upper bound: 12.4991457
time: 14.91 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -42.6894836, -18.9407883, -42.6844482, -18.8805981, -15.9298935, 15.8385544
1: -30.4961739, -12.9656649, -30.4713440, -12.9163485, -11.8939705, 11.7986050
2: -21.5893173, -5.9885845, -21.5917511, -5.9573374, -10.7372284, 10.7121849
3: -30.5847397, -12.4189787, -30.5859261, -12.4088879, -12.8167534, 12.8326797
4: -18.8434753, 0.6104360, -18.8444347, 0.6593122, -12.7777672, 12.6645222
5: -30.7228699, -9.8928270, -30.7270527, -9.8515720, -14.2500000, 14.2420235
6: -14.2916718, 1.3351524, -14.3276939, 1.3317358, -11.0444221, 11.1015472
7: -47.5146294, -25.9930038, -47.4953842, -25.9497471, -13.1843262, 13.0750580
8: -33.6698494, -12.1349392, -33.6713676, -12.0687904, -14.0819664, 13.9769249
9: -18.6114235, -6.9834166, -18.5997505, -6.9566202, -9.8194771, 9.7604942
10: -38.4031143, -15.4188213, -38.3484268, -15.3769398, -18.3511772, 18.2385941
11: -57.4545517, -34.7074165, -57.3953629, -34.7030296, -13.4256344, 13.3325920
12: -1.9921412, 17.9347019, -1.9632647, 17.9414272, -16.1525192, 16.0627174
13: -8.2615967, 9.4680080, -8.2813625, 9.4980545, -14.8877335, 14.9078255
14: -79.1398773, -46.7479858, -79.0504456, -46.6880989, -25.4729691, 25.2922897
15: -11.0532560, 7.7983799, -11.0559015, 7.8579416, -13.2223969, 13.1416378
16: -49.1827621, -29.5173588, -49.1464729, -29.5048714, -12.8629990, 12.7755413
17: -79.7488403, -45.0367584, -79.6462479, -44.9873657, -22.3447189, 22.1611938
18: -11.6318569, 9.5317755, -11.6149321, 9.5390720, -19.2966766, 19.2671204
19: -27.6103325, -12.5083437, -27.6208839, -12.5067167, -10.6315937, 10.6456814
20: -17.4757614, -4.3160458, -17.4858704, -4.3163981, -12.4692307, 12.4681168
21: -39.8510170, -20.1249352, -39.8258820, -20.1233501, -14.1714020, 14.1386299
22: -17.7106819, 0.3097048, -17.6859493, 0.3157568, -12.8358917, 12.7918110
23: -22.2996635, -7.4008217, -22.3283405, -7.3963022, -11.1858940, 11.2150383
24: -8.7309074, 4.7513556, -8.7469091, 4.7525902, -9.3644142, 9.3782673
25: -5.7858615, 8.8878784, -5.8084183, 8.8968716, -12.0359344, 12.0532417
26: -19.2281513, 2.4233675, -19.2157707, 2.4316213, -18.8676300, 18.8146973
27: -23.7455006, -7.1771536, -23.7353859, -7.1718216, -11.9907990, 11.9565582
28: -18.3409481, -0.1338258, -18.3824806, -0.1301217, -12.5133018, 12.5674210
29: -36.4774361, -18.7453461, -36.4339371, -18.7360992, -11.6967945, 11.6137314
30: -27.9369526, -7.8043332, -27.9219456, -7.8002968, -12.6368160, 12.6084805
31: -23.2098351, -7.2171445, -23.2277431, -7.2169666, -12.2446232, 12.2610016
32: -4.6356969, 10.3749294, -4.6676645, 10.3707247, -11.9893570, 12.0469131
33: 17.4536514, 40.2944489, 17.3842907, 40.2519684, -17.7135620, 17.8447800
34: 6.0361390, 25.5413361, 5.9885225, 25.5053825, -13.5313377, 13.6275940
35: 10.2666817, 29.9785404, 10.2096024, 29.9224167, -15.6105423, 15.7366791
36: 5.8062897, 23.6650009, 5.7516007, 23.6278572, -12.5804768, 12.6878700
37: 13.6648855, 32.5455627, 13.6244621, 32.5367126, -13.6263008, 13.7005615
38: 4.0646143, 25.2512550, 4.0188556, 25.2214012, -17.1488190, 17.2453079
39: 8.9912281, 31.0079384, 8.9244556, 30.9774036, -14.9959621, 15.1186104
40: 6.9176521, 25.1892757, 6.8797150, 25.1678352, -12.7190266, 12.7922230
41: -5.1085939, 10.3710871, -5.1433988, 10.3602829, -11.4080505, 11.4893398
42: -16.9194336, 0.0256031, -16.9383640, 0.0335014, -11.3787384, 11.4233513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=101, inp2_unstable=102, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=141, inp2_unstable=141, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1005
type: B, layer: 1, pos: 1491
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1493
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1422
type: B, layer: 1, pos: 1021
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 1413
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1487
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1438
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1475
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1471
type: B, layer: 1, pos: 1405

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 667

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4589301, upper bound: 12.5051053
time: 16.38 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4987006, upper bound: 12.4987018
time: 15.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 34.05 seconds
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 34.05
Output dim: 33, lower bound: -12.3994916, upper bound: 12.5054413
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 34.05
Output dim: 33, lower bound: -12.4392382, upper bound: 12.4990313
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 34.05
Output dim: 33, lower bound: -12.4196514, upper bound: 12.5049659
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 34.05
Output dim: 33, lower bound: -12.4594218, upper bound: 12.4985685
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 34.05
Output dim: 33, lower bound: -12.4387753, upper bound: 12.5055593
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 34.05
Output dim: 33, lower bound: -12.4785320, upper bound: 12.4991457
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 34.05
Output dim: 33, lower bound: -12.4589301, upper bound: 12.5051053
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 34.05
Output dim: 33, lower bound: -12.4987006, upper bound: 12.4987018

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 20.22 + 315.84 = 336.06 seconds
