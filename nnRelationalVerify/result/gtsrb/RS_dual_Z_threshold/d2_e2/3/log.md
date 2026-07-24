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
execution time: IAR + RelationalAnalysis = 2.81 + 16.86 = 19.67 seconds
status: Status.UNKNOWN
relational distance
Output dim: 33, lower bound: -12.5279291, upper bound: 12.5279291

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5268719, upper bound: 12.4865129
time: 17.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4865130, upper bound: 12.5268719
time: 6.11 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 23.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 23.66
Output dim: 33, lower bound: -12.5268719, upper bound: 12.4865129
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 23.66
Output dim: 33, lower bound: -12.4865130, upper bound: 12.5268719

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8491402, 15.8489456
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8109207, 11.8119335
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7214737, 10.7214127
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.8091736, 12.8069572
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6792183, 12.6785927
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2494888, 14.2480392
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0253735, 11.0198441
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0826797, 13.0825882
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9926872, 13.9923935
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7718163, 9.7714138
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2315407, 18.2320251
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3310585, 13.3370171
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0972672, 16.0975571
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8745041, 14.8735809
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2581558, 25.2719421
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1417503, 13.1436272
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7797432, 12.7796898
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1296310, 22.1437302
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2218628, 19.2261276
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6058407, 10.6091881
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4420547, 12.4420547
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0988255, 14.1036072
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7736778, 12.7789497
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1800194, 11.1830521
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3344383, 9.3403835
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0086155, 12.0118713
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7973862, 18.8019638
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9419098, 11.9446716
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5220795, 12.5235443
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6219349, 11.6283493
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5677681, 12.5736046
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2171421, 12.2205696
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9699173, 11.9659805
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7329826, 17.7315369
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5289688, 13.5225773
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6170998, 15.6131172
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5668335, 12.5598717
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6232529, 13.6213989
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1384583, 17.1296425
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0125999, 15.0090065
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7035904, 12.6972065
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3875065, 11.3841896
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3610210, 11.3561382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4872155, upper bound: 12.4822454
time: 15.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5261898, upper bound: 12.4701641
time: 17.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8489418, 15.8491402
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8119354, 11.8109226
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7214127, 10.7214737
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.8069611, 12.8091698
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6785965, 12.6792183
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2480392, 14.2494888
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0198421, 11.0253696
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0825920, 13.0826797
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9923935, 13.9926872
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7714119, 9.7718163
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2320213, 18.2315369
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3370171, 13.3310604
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0975571, 16.0972672
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8735809, 14.8745041
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2719421, 25.2581558
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1436272, 13.1417503
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7796898, 12.7797413
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1437302, 22.1296310
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2261276, 19.2218628
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6091900, 10.6058388
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4420547, 12.4420547
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1036091, 14.0988274
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7789497, 12.7736778
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1830521, 11.1800194
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3403816, 9.3344383
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0118732, 12.0086136
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8019714, 18.7973862
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9446716, 11.9419098
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5235443, 12.5220795
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6283474, 11.6219368
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5736046, 12.5677681
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2205677, 12.2171402
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9659805, 11.9699173
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7315407, 17.7329865
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5225754, 13.5289688
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6131172, 15.6170998
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5598717, 12.5668335
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6213951, 13.6232529
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1296387, 17.1384621
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0090027, 15.0125999
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6972046, 12.7035942
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3841915, 11.3875084
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3561382, 11.3610210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4701641, upper bound: 12.5261898
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4822454, upper bound: 12.4872155
time: 5.47 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.87 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 14.87
Output dim: 33, lower bound: -12.4872155, upper bound: 12.4822454
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.87
Output dim: 33, lower bound: -12.5261898, upper bound: 12.4701641
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.87
Output dim: 33, lower bound: -12.4701641, upper bound: 12.5261898
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 14.87
Output dim: 33, lower bound: -12.4822454, upper bound: 12.4872155

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8571167, 15.8552513
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8037186, 11.8053493
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7066612, 10.6980438
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7809410, 12.7682915
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.7056770, 12.6982994
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2090912, 14.1918488
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0301380, 11.0234241
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.1052780, 13.0982437
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -14.0101662, 14.0028343
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7664680, 9.7650261
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2294350, 18.2297363
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2973576, 13.3116131
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0294418, 16.0458488
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8501167, 14.8413200
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2682800, 25.2841949
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1439743, 13.1458187
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7881737, 12.7857208
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1267014, 22.1414490
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2187500, 19.2237778
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6055012, 10.6091766
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4562340, 12.4521408
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0910568, 14.0977497
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7563477, 12.7658882
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1706429, 11.1764927
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3268585, 9.3346691
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0094337, 12.0131302
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7633133, 18.7762833
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9154816, 11.9247551
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5179710, 12.5226898
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5800819, 11.5970535
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5553379, 12.5636559
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2166195, 12.2237930
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9742088, 11.9709663
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7232208, 17.7185822
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5320129, 13.5243320
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6132622, 15.6083717
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5718689, 12.5639191
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6390953, 13.6377296
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1398659, 17.1296082
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9977570, 14.9893188
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7019730, 12.6951160
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4098969, 11.4099522
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3808517, 11.3744850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 574

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4830746, upper bound: 12.4685814
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4812406, upper bound: 12.4366912
time: 7.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8552551, 15.8571205
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8053513, 11.8037167
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6980438, 10.7066612
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7682915, 12.7809410
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6982994, 12.7056770
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1918488, 14.2090874
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0234241, 11.0301361
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0982437, 13.1052780
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -14.0028343, 14.0101662
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7650261, 9.7664661
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2297401, 18.2294350
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3116131, 13.2973576
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0458527, 16.0294418
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8413200, 14.8501205
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2841949, 25.2682800
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1458168, 13.1439724
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7857208, 12.7881737
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1414413, 22.1266937
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2237778, 19.2187500
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6091785, 10.6055012
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4521408, 12.4562340
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0977478, 14.0910568
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7658882, 12.7563496
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1764946, 11.1706409
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3346710, 9.3268566
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0131302, 12.0094337
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7762833, 18.7633133
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9247551, 11.9154816
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5226898, 12.5179749
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5970535, 11.5800838
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5636539, 12.5553360
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2237911, 12.2166214
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9709663, 11.9742088
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7185822, 17.7232208
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5243340, 13.5320129
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6083717, 15.6132622
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5639191, 12.5718689
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6377296, 13.6390991
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1296043, 17.1398659
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9893188, 14.9977608
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6951180, 12.7019730
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4099541, 11.4098969
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3744850, 11.3808517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 574

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4366912, upper bound: 12.5254526
time: 18.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4685814, upper bound: 12.4830746
time: 27.14 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 47.57 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 47.57
Output dim: 33, lower bound: -12.4830746, upper bound: 12.4685814
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 47.57
Output dim: 33, lower bound: -12.4812406, upper bound: 12.4366912
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 47.57
Output dim: 33, lower bound: -12.4366912, upper bound: 12.5254526
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 47.57
Output dim: 33, lower bound: -12.4685814, upper bound: 12.4830746

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8290939, 15.8247719
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7793427, 11.7700710
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6918983, 10.6986809
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7690659, 12.7810974
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6876068, 12.6924629
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1914825, 14.2083588
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0083237, 11.0187492
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0818062, 13.0850601
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9851875, 13.9898949
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7557869, 9.7574425
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2108078, 18.2056541
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2925720, 13.2725811
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0341492, 16.0185318
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8404541, 14.8490791
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2506485, 25.2238007
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1282463, 13.1212654
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7804337, 12.7836990
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1045723, 22.0777969
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2170410, 19.2099838
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6091995, 10.6044464
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4523392, 12.4569740
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0888443, 14.0792446
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7569656, 12.7445164
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1763515, 11.1704769
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3297310, 9.3203239
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0133038, 12.0093117
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7623520, 18.7462006
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9229851, 11.9138641
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5211411, 12.5173645
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5802574, 11.5578136
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5524731, 12.5448456
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2238922, 12.2153816
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9477348, 11.9565926
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7042046, 17.7123718
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5055504, 13.5178509
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5859070, 15.5963249
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5362892, 12.5510292
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6316986, 13.6345482
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1191063, 17.1319847
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9627724, 14.9777451
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6710033, 12.6837883
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3962975, 11.3996220
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3770542, 11.3845978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 575

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4149288, upper bound: 12.5248296
time: 15.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4351489, upper bound: 12.4992977
time: 13.37 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 31.20 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.20
Output dim: 33, lower bound: -12.4149288, upper bound: 12.5248296
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 31.20
Output dim: 33, lower bound: -12.4351489, upper bound: 12.4992977

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8245316, 15.8145638
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7741241, 11.7584171
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6911240, 10.6969604
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7686920, 12.7802582
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6862755, 12.6894913
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1911316, 14.2077255
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0052834, 11.0173893
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0790024, 13.0788040
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9822388, 13.9833069
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7530060, 9.7533627
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2052841, 18.1949196
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2866898, 13.2621994
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0265961, 16.0115471
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8394928, 14.8475876
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2446671, 25.2111664
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1254997, 13.1151314
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7766800, 12.7804680
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0979233, 22.0629425
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2161865, 19.2082367
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6076088, 10.6022682
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4509048, 12.4560890
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0862198, 14.0743198
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7557602, 12.7418213
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1757927, 11.1700306
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3289394, 9.3189468
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0132084, 12.0091286
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7584991, 18.7404480
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9228363, 11.9137688
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5165482, 12.5152664
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5769157, 11.5504303
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5482655, 12.5411263
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2190781, 12.2096786
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9409065, 11.9534683
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.6994438, 17.7102432
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.4985352, 13.5147114
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5772438, 15.5924454
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5266991, 12.5467339
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6316528, 13.6345444
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1160088, 17.1306038
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9550743, 14.9742661
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6643124, 12.6807899
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3939781, 11.3986149
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3730736, 11.3837547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 541

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.3736257, upper bound: 12.5240963
time: 5.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4141556, upper bound: 12.4857886
time: 16.47 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.40 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.40
Output dim: 33, lower bound: -12.3736257, upper bound: 12.5240963
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 24.40
Output dim: 33, lower bound: -12.4141556, upper bound: 12.4857886

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8227615, 15.8127098
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7708054, 11.7518291
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6861191, 10.6922951
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7644653, 12.7770424
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6853600, 12.6878700
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1855316, 14.2029037
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0044250, 11.0170937
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0747490, 13.0711517
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9811172, 13.9821396
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7472610, 9.7479553
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.1970253, 18.1815643
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2749729, 13.2308369
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0208893, 16.0044289
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8306656, 14.8443451
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2327347, 25.1886520
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1248322, 13.1135674
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7664909, 12.7605057
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0855980, 22.0361786
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2140732, 19.2035446
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6029377, 10.5906506
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4485779, 12.4523849
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0777702, 14.0526943
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7523842, 12.7340870
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1709251, 11.1596050
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3245621, 9.3110962
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0102119, 12.0045967
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7529831, 18.7287598
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9173050, 11.8992691
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5143585, 12.5095673
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5706139, 11.5315304
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5380745, 12.5184269
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2159081, 12.2020416
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9338989, 11.9480743
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.6897049, 17.7067871
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.4893150, 13.5111198
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5682755, 15.5892601
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5166702, 12.5419312
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6315308, 13.6344757
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.0988083, 17.1237450
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9450874, 14.9702759
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6576824, 12.6762829
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3938484, 11.3985157
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3709126, 11.3826561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.3456217, upper bound: 12.5235208
time: 16.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.3721018, upper bound: 12.4882120
time: 14.98 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 33.58 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 33.58
Output dim: 33, lower bound: -12.3456217, upper bound: 12.5235208
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 33.58
Output dim: 33, lower bound: -12.3721018, upper bound: 12.4882120

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8097000, 15.7964668
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7547035, 11.7305870
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6851540, 10.6906471
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7674332, 12.7794609
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6839561, 12.6860981
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1818008, 14.1980057
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0054550, 11.0183334
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0520706, 13.0415573
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9769897, 13.9755745
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7366180, 9.7347202
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.1687126, 18.1443214
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2305679, 13.1740971
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0163193, 16.0000381
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8305626, 14.8435402
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.1916046, 25.1372375
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1260490, 13.1144810
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7327881, 12.7177544
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0420990, 21.9795685
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2093048, 19.1980591
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5979862, 10.5816326
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4485893, 12.4507370
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0471783, 14.0135231
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7464066, 12.7262783
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1733799, 11.1600304
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3249226, 9.3101845
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0109024, 12.0039139
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7474899, 18.7215729
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9040451, 11.8837967
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5145969, 12.5098763
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5495796, 11.5037899
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5076981, 12.4820480
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2087135, 12.1917534
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9151344, 11.9318695
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.6715584, 17.6931114
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.4640884, 13.4921036
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5442352, 15.5711479
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.4885483, 12.5206680
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6262455, 13.6300621
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.0672073, 17.0998726
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9290752, 14.9580956
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6430702, 12.6648560
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3854218, 11.3914089
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3717613, 11.3850079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 543

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.3290069, upper bound: 12.5200494
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.3423795, upper bound: 12.5044412
time: 25.95 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 34.12 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 34.12
Output dim: 33, lower bound: -12.3290069, upper bound: 12.5200494
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 34.12
Output dim: 33, lower bound: -12.3423795, upper bound: 12.5044412

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8099518, 15.7945595
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7552223, 11.7276878
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6850204, 10.6902695
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7675552, 12.7793159
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6839256, 12.6853313
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1817360, 14.1973648
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0039330, 11.0174656
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0519333, 13.0370140
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9769897, 13.9736938
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7361107, 9.7322102
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.1683884, 18.1375847
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2291870, 13.1642876
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0139389, 15.9986191
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8294067, 14.8414497
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.1914673, 25.1310196
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1259766, 13.1128426
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7315750, 12.7096977
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0425682, 21.9723701
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2093124, 19.1980667
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5979462, 10.5785599
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4482727, 12.4490623
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0463066, 14.0068035
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7464790, 12.7260132
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1730080, 11.1586456
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3248959, 9.3093395
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0109482, 12.0029907
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7464600, 18.7199097
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9029083, 11.8815460
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5144138, 12.5099182
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5499420, 11.4999008
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5065422, 12.4762173
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2087975, 12.1893921
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9131927, 11.9320679
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.6687698, 17.6932297
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.4599190, 13.4927902
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5402298, 15.5717163
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.4844933, 12.5212669
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6260910, 13.6300468
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.0624504, 17.1001358
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9275036, 14.9582825
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6419792, 12.6650887
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3854103, 11.3914089
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3684273, 11.3827534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 606

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.3113714, upper bound: 12.5187695
time: 6.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.3277043, upper bound: 12.5025285
time: 5.80 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 14.56 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 14.56
Output dim: 33, lower bound: -12.3113714, upper bound: 12.5187695
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 14.56
Output dim: 33, lower bound: -12.3277043, upper bound: 12.5025285

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8058052, 15.7912483
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7522163, 11.7255669
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6851501, 10.6900940
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7666664, 12.7792740
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6826744, 12.6847477
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1806335, 14.1973076
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0014610, 11.0174408
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0512009, 13.0356293
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9732361, 13.9720955
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7327461, 9.7302589
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.1642418, 18.1326332
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2294350, 13.1576576
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0138702, 15.9988174
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8293953, 14.8414536
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.1880951, 25.1186447
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1249580, 13.1120453
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7286530, 12.7053280
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0391312, 21.9630699
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2093430, 19.1976471
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5974293, 10.5755768
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4480209, 12.4477539
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0461159, 14.0027752
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7461510, 12.7247810
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1728497, 11.1556511
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3251114, 9.3071098
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0110264, 12.0021667
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7465515, 18.7189178
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9030113, 11.8802223
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5142288, 12.5081520
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5495682, 11.4954834
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5069160, 12.4723263
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2084045, 12.1875229
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9102364, 11.9321404
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.6649628, 17.6911659
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.4583549, 13.4933128
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5369568, 15.5696411
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.4797745, 12.5195732
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6235123, 13.6285934
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.0583458, 17.1001511
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9234753, 14.9576416
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6381378, 12.6636028
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3822632, 11.3903904
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3672981, 11.3825836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1762

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.3106131, upper bound: 12.4948713
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.2872240, upper bound: 12.5179980
time: 6.36 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 15.18 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 15.18
Output dim: 33, lower bound: -12.3106131, upper bound: 12.4948713
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 15.18
Output dim: 33, lower bound: -12.2872240, upper bound: 12.5179980

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.7864304, 15.7654114
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7386017, 11.7073803
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6775475, 10.6802063
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7613564, 12.7721214
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6726456, 12.6715927
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1697121, 14.1830139
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0004158, 11.0165710
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0329819, 13.0113831
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9543076, 13.9468575
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7219505, 9.7159348
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.1445923, 18.1061020
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2329731, 13.1579742
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0104980, 15.9960060
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8287697, 14.8396378
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.1640854, 25.0865021
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1200409, 13.1056652
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7188263, 12.6915646
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0254517, 21.9466248
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2064056, 19.1977310
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5997810, 10.5769310
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4489441, 12.4481735
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0480728, 14.0020218
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7458954, 12.7245007
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1721516, 11.1550827
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3226776, 9.3057861
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0104389, 12.0017853
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7416916, 18.7162552
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.8984985, 11.8768158
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5059776, 12.5017548
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5462685, 11.4914150
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5069847, 12.4724121
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2116966, 12.1901093
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9046707, 11.9280090
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.6478577, 17.6783447
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.4345131, 13.4748821
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5159264, 15.5537529
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.4603882, 12.5050125
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6170731, 13.6235046
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.0498238, 17.0929642
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9177799, 14.9529800
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6320648, 12.6585655
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3765717, 11.3860893
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3716965, 11.3864479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1630

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.2856857, upper bound: 12.4997720
time: 15.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.2772707, upper bound: 12.5166661
time: 17.64 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 35.82 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 35.82
Output dim: 33, lower bound: -12.2856857, upper bound: 12.4997720
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 35.82
Output dim: 33, lower bound: -12.2772707, upper bound: 12.5166661

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.7866096, 15.7659416
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7385941, 11.7073898
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6726570, 10.6760292
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7453232, 12.7600555
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6710587, 12.6720428
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1526642, 14.1696930
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -10.9897633, 11.0085220
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0295296, 13.0085831
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9507828, 13.9457092
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7219238, 9.7159100
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.1381989, 18.0989380
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2168655, 13.1366539
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0041275, 15.9889755
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8210258, 14.8337898
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.1457748, 25.0624542
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1203995, 13.1063843
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7148666, 12.6866455
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0133514, 21.9306221
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2007370, 19.1902237
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5917892, 10.5667191
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4468498, 12.4455795
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0395927, 13.9907932
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7412872, 12.7185192
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1618080, 11.1414967
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3113499, 9.2907906
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0050087, 11.9945755
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7342911, 18.7064590
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.8917351, 11.8678665
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4959717, 12.4887772
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5346603, 11.4761639
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.4916687, 12.4521427
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2074757, 12.1847076
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9004822, 11.9264908
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.6437569, 17.6748886
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.4363823, 13.4784451
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5126991, 15.5510025
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.4509945, 12.4981880
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6175537, 13.6239052
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.0452614, 17.0923271
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9090347, 14.9467049
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6251793, 12.6536865
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3698883, 11.3807297
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3647041, 11.3808060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 605

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.2615319, upper bound: 12.5141397
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.2748424, upper bound: 12.4986280
time: 6.59 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 14.65 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 14.65
Output dim: 33, lower bound: -12.2615319, upper bound: 12.5141397
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 14.65
Output dim: 33, lower bound: -12.2748424, upper bound: 12.4986280

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 19.67 + 341.82 = 361.48 seconds
