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
execution time: IAR + RelationalAnalysis = 2.85 + 17.49 = 20.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 33, lower bound: -12.5279291, upper bound: 12.5279291

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1786

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 581

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5274363, upper bound: 12.5080013
time: 23.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5080013, upper bound: 12.5274363
time: 12.91 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 36.24 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 36.24
Output dim: 33, lower bound: -12.5274363, upper bound: 12.5080013
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 36.24
Output dim: 33, lower bound: -12.5080013, upper bound: 12.5274363

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8461304, 15.8534813
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8153801, 11.8188248
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7113991, 10.7162361
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.8064575, 12.8118248
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6830826, 12.6875687
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2469673, 14.2518730
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0487976, 11.0474052
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0835762, 13.0869370
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9932709, 13.9955177
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7696915, 9.7696609
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2481117, 18.2466240
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3568401, 13.3547955
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0987625, 16.0960312
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8737755, 14.8767509
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3109589, 25.3160934
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1522522, 13.1577072
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7838593, 12.7826729
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1673126, 22.1776886
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2384644, 19.2385406
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6112499, 10.6107121
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4438171, 12.4414368
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1183872, 14.1157112
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7886581, 12.7933578
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1916313, 11.1915531
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3572960, 9.3566418
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0174847, 12.0146027
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8140869, 18.8136902
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9529839, 11.9529457
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5220661, 12.5227242
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6355534, 11.6405678
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5930290, 12.5858498
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2236404, 12.2230721
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9945259, 11.9937668
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7305603, 17.7254028
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5432682, 13.5430412
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6258125, 15.6254539
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5895329, 12.5896683
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6239815, 13.6157761
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1566582, 17.1571503
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0177841, 15.0128136
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7194824, 12.7132702
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4153786, 11.4145737
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3800697, 11.3780594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 590

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1444

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5218290, upper bound: 12.5077632
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5271983, upper bound: 12.5023904
time: 6.64 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8534851, 15.8461304
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8188248, 11.8153801
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7162361, 10.7113991
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.8118286, 12.8064575
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6875687, 12.6830788
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2518730, 14.2469673
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0474052, 11.0487976
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0869370, 13.0835762
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9955215, 13.9932709
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7696609, 9.7696915
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2466240, 18.2481079
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3547955, 13.3568401
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0960312, 16.0987625
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8767509, 14.8737755
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3160934, 25.3109589
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1577072, 13.1522522
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7826729, 12.7838593
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1776886, 22.1673126
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2385406, 19.2384644
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6107121, 10.6112499
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4414368, 12.4438171
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1157131, 14.1183853
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7933578, 12.7886600
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1915512, 11.1916313
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3566399, 9.3572979
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0146046, 12.0174866
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8136902, 18.8140869
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9529457, 11.9529839
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5227222, 12.5220642
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6405659, 11.6355515
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5858498, 12.5930290
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2230721, 12.2236404
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9937668, 11.9945259
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7254028, 17.7305603
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5430393, 13.5432682
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6254539, 15.6258125
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5896702, 12.5895309
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6157761, 13.6239815
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1571541, 17.1566582
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0128136, 15.0177841
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7132721, 12.7194843
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4145737, 11.4153767
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3780594, 11.3800678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1713

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 616

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5079109, upper bound: 12.5207152
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5012835, upper bound: 12.5273459
time: 5.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.53 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.53
Output dim: 33, lower bound: -12.5218290, upper bound: 12.5077632
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.53
Output dim: 33, lower bound: -12.5271983, upper bound: 12.5023904
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.53
Output dim: 33, lower bound: -12.5079109, upper bound: 12.5207152
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.53
Output dim: 33, lower bound: -12.5012835, upper bound: 12.5273459

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8464890, 15.8535004
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8153229, 11.8187332
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7094727, 10.7148132
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.8071747, 12.8124313
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6825409, 12.6876507
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2463989, 14.2512054
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0493584, 11.0478725
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0832100, 13.0863533
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9889679, 13.9920998
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7689819, 9.7687874
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2508087, 18.2490387
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3564091, 13.3539734
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0986557, 16.0956726
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8722801, 14.8756218
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3098145, 25.3146591
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1523895, 13.1577148
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7824364, 12.7808113
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1666336, 22.1758881
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2361069, 19.2356491
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6108570, 10.6097527
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4438248, 12.4414482
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1182785, 14.1149597
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7854156, 12.7888069
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1904793, 11.1895103
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3566837, 9.3559971
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0174751, 12.0141525
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8117981, 18.8109131
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9451027, 11.9442101
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5196323, 12.5187836
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6344872, 11.6379566
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5921478, 12.5841866
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2235088, 12.2227859
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9945259, 11.9937172
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7284889, 17.7237549
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5406628, 13.5410347
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6270485, 15.6273880
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5897675, 12.5894928
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6229877, 13.6148529
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1566200, 17.1572838
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0128021, 15.0084610
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7186470, 12.7130928
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4166584, 11.4154091
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3797779, 11.3776512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1628

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1487

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5215975, upper bound: 12.5036318
time: 25.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5176979, upper bound: 12.5075317
time: 16.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8461533, 15.8538399
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8152847, 11.8187675
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7099762, 10.7143097
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.8070602, 12.8125420
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6831589, 12.6870327
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2462997, 14.2513008
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0492630, 11.0479660
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0829926, 13.0865707
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9898529, 13.9912186
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7688179, 9.7689514
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2505264, 18.2493248
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3560200, 13.3543663
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0984039, 16.0959244
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8726463, 14.8752556
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3095245, 25.3149490
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1522598, 13.1578445
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7820015, 12.7812519
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1655121, 22.1770096
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2355728, 19.2361832
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6102924, 10.6103191
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4438286, 12.4414444
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1176338, 14.1156044
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7841110, 12.7901134
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1895866, 11.1904011
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3566532, 9.3560295
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0170326, 12.0145950
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8113098, 18.8114014
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9442482, 11.9450684
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5181255, 12.5202904
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6329422, 11.6395035
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5913658, 12.5849686
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2233524, 12.2229385
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9944801, 11.9937630
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7289162, 17.7233315
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5412617, 13.5404377
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6277504, 15.6266899
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5893555, 12.5899048
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6230602, 13.6147842
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1567879, 17.1571083
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0134354, 15.0078278
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7193069, 12.7124348
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4162159, 11.4158554
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3796597, 11.3777695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 516

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1764

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5238212, upper bound: 12.4899506
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5147890, upper bound: 12.4990255
time: 6.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8425522, 15.8331223
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8137779, 11.8107262
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7121162, 10.7047882
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.8056374, 12.7960930
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6653519, 12.6578503
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2422638, 14.2316246
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0483322, 11.0479736
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0838509, 13.0805740
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9668617, 13.9582138
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7661209, 9.7648010
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2462158, 18.2459335
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3100891, 13.3263683
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0870132, 16.0929108
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8742065, 14.8691521
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3108521, 25.3099442
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1419945, 13.1363621
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7795868, 12.7870464
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1356010, 22.1390457
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2344513, 19.2385483
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5755138, 10.5838814
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4398651, 12.4420738
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0837975, 14.0943413
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7729721, 12.7743721
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1585922, 11.1646061
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3433075, 9.3481731
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0059280, 12.0107956
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8088608, 18.8118515
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9433060, 11.9486427
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4920540, 12.4992485
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6112614, 11.6182384
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5576172, 12.5723896
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.1973991, 12.2036285
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9933052, 11.9934082
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7259979, 17.7280426
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5395203, 13.5382099
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6255264, 15.6255836
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5853558, 12.5844917
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6146927, 13.6218681
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1518478, 17.1411819
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9996185, 14.9993172
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7061195, 12.7073021
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4150219, 11.4158878
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3766441, 11.3778305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1726

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1713

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5065109, upper bound: 12.5059855
time: 14.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4931758, upper bound: 12.5193116
time: 15.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8404770, 15.8352013
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8141747, 11.8103333
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7096252, 10.7072830
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.8014641, 12.8002701
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6623459, 12.6608601
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2365265, 14.2373581
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0465813, 11.0497227
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0839348, 13.0804901
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9604645, 13.9646111
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7647705, 9.7661495
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2444458, 18.2477036
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3243256, 13.3121338
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0901794, 16.0897446
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8721237, 14.8712349
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3150787, 25.3057175
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1418190, 13.1365414
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7858582, 12.7807713
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1494255, 22.1252327
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2386246, 19.2343674
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5833454, 10.5760517
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4396935, 12.4422455
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0916672, 14.0864716
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7790680, 12.7682762
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1645279, 11.1586704
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3475189, 9.3439617
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0079117, 12.0088120
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8114548, 18.8092575
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9486046, 11.9433403
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4999084, 12.4913940
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6232548, 11.6062450
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5652084, 12.5647945
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2030602, 12.1979675
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9926491, 11.9940643
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7228851, 17.7311554
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5379829, 13.5397491
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6252213, 15.6258850
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5846310, 12.5852203
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6136665, 13.6228981
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1416779, 17.1513519
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9943466, 15.0045929
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7010918, 12.7123299
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4150829, 11.4158268
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3758202, 11.3786545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1591

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 516

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4917175, upper bound: 12.5216381
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4955767, upper bound: 12.5177789
time: 17.33 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.35 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.35
Output dim: 33, lower bound: -12.5215975, upper bound: 12.5036318
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.35
Output dim: 33, lower bound: -12.5176979, upper bound: 12.5075317
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.35
Output dim: 33, lower bound: -12.5238212, upper bound: 12.4899506
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 25.35
Output dim: 33, lower bound: -12.5147890, upper bound: 12.4990255
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 25.35
Output dim: 33, lower bound: -12.5065109, upper bound: 12.5059855
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.35
Output dim: 33, lower bound: -12.4931758, upper bound: 12.5193116
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.35
Output dim: 33, lower bound: -12.4917175, upper bound: 12.5216381
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.35
Output dim: 33, lower bound: -12.4955767, upper bound: 12.5177789

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8441544, 15.8517303
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8135033, 11.8174000
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7076988, 10.7135277
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.8054619, 12.8111954
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6800461, 12.6858387
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2443466, 14.2497063
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0486870, 11.0469894
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0807152, 13.0845032
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9866676, 13.9904594
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7684174, 9.7683315
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2497368, 18.2483482
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3563957, 13.3539677
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0974503, 16.0941544
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8710785, 14.8747368
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3091049, 25.3141861
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1512604, 13.1568928
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7820969, 12.7805710
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1652145, 22.1747971
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2358170, 19.2352600
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6103706, 10.6092815
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4438553, 12.4415207
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1178169, 14.1144829
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7851067, 12.7883930
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1898041, 11.1885490
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3564281, 9.3556442
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0170822, 12.0136223
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8112946, 18.8102798
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9442024, 11.9429245
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5187492, 12.5174789
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6342163, 11.6375027
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5917778, 12.5836315
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2228813, 12.2221069
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9938660, 11.9927368
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7278976, 17.7230263
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5398846, 13.5400105
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6261978, 15.6262512
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5889511, 12.5883789
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6222820, 13.6138992
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1566925, 17.1573448
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0126152, 15.0082321
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7180061, 12.7122688
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4160080, 11.4144325
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3794346, 11.3772125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 543

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1742

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5190405, upper bound: 12.4857123
time: 13.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5023316, upper bound: 12.5009923
time: 6.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8447189, 15.8511658
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8139915, 11.8169193
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7081871, 10.7130394
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.8059349, 12.8107262
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6807251, 12.6851559
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2448959, 14.2491608
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0484772, 11.0472031
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0813560, 13.0838585
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9873276, 13.9897995
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7685280, 9.7682228
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2501183, 18.2479668
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3564034, 13.3539562
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0971375, 16.0944672
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8713951, 14.8744202
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3093414, 25.3139496
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1515656, 13.1565914
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7821960, 12.7804699
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1655426, 22.1744690
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2357178, 19.2353668
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6103859, 10.6092682
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4438972, 12.4414787
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1178017, 14.1144981
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7849998, 12.7885017
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1895180, 11.1888351
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3563328, 9.3557396
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0169449, 12.0137558
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8111649, 18.8104019
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9438171, 11.9433098
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5183258, 12.5178986
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6340332, 11.6376858
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5915947, 12.5838127
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2228279, 12.2221565
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9935379, 11.9930611
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7277603, 17.7231636
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5396404, 13.5402546
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6259079, 15.6265411
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5886536, 12.5886765
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6220341, 13.6141434
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1566772, 17.1573563
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0125694, 15.0082741
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7178230, 12.7124519
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4156799, 11.4147587
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3793392, 11.3773098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 620

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1730

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5099880, upper bound: 12.4921303
time: 19.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5023017, upper bound: 12.4998069
time: 30.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8240509, 15.8362160
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7923241, 11.8007355
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6888161, 10.6982956
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7876854, 12.7978783
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6476326, 12.6602688
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2258911, 14.2358589
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0398483, 11.0367203
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0619431, 13.0705795
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9481277, 13.9599075
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7686539, 9.7687874
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2427979, 18.2432098
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3411579, 13.3365974
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0962296, 16.0934029
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8644066, 14.8691216
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2817841, 25.2923660
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1253624, 13.1369190
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7791977, 12.7728348
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1586914, 22.1713028
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2321777, 19.2317276
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5957699, 10.5910988
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4399071, 12.4368095
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1061954, 14.1009750
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7806473, 12.7856026
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1726227, 11.1679649
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3508682, 9.3482475
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0112801, 12.0060158
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8106613, 18.8106766
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9342957, 11.9319153
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4965134, 12.4917336
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6225510, 11.6265335
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5751534, 12.5631332
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2110863, 12.2072678
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9875374, 11.9856911
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7263641, 17.7192650
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5319328, 13.5287323
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6180878, 15.6128311
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5778751, 12.5744133
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6104431, 13.5984688
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1554489, 17.1572380
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0123501, 15.0066910
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7139320, 12.7069473
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4036980, 11.3990688
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3755245, 11.3730965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1478

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1398

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5204471, upper bound: 12.4878224
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5216731, upper bound: 12.4866135
time: 30.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.7878761, 15.7640076
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7840500, 11.7709675
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6950340, 10.6833687
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7750092, 12.7575951
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6318550, 12.6089249
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2047958, 14.1843948
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0268593, 11.0337372
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0498619, 13.0392456
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9195976, 13.8975449
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7628784, 9.7595196
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2133865, 18.2163391
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2607422, 13.2842712
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0443192, 16.0588493
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8705940, 14.8638191
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3180466, 25.3136292
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0872765, 13.0666389
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7628899, 12.7719707
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1315994, 22.1337433
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2287903, 19.2338715
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5759106, 10.5844078
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4349518, 12.4372368
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0664978, 14.0772190
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7727623, 12.7674789
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1339493, 11.1472092
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3239594, 9.3331394
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -11.9974461, 12.0036011
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7967911, 18.8025360
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9439087, 11.9491234
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4919319, 12.5008240
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6024208, 11.6080132
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5214386, 12.5429306
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.1830368, 12.1895866
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9850426, 11.9865723
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7046471, 17.7122269
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5298157, 13.5306072
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6280937, 15.6309319
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5847588, 12.5844193
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.5948906, 13.6073074
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1623421, 17.1519699
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9659920, 14.9729271
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6804543, 12.6868954
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4154091, 11.4222984
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3470879, 11.3561878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1761

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 543

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4741717, upper bound: 12.5158435
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4897169, upper bound: 12.5003530
time: 24.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8405037, 15.8351173
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8140488, 11.8102875
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7108078, 10.7071114
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.8023567, 12.8001709
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6625862, 12.6606617
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2381172, 14.2363892
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0464287, 11.0492573
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0859756, 13.0789795
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9601784, 13.9648209
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7647476, 9.7662659
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2444229, 18.2476654
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3262310, 13.3107166
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0890732, 16.0926437
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8725395, 14.8710632
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3169250, 25.3043365
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1412544, 13.1370964
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7881775, 12.7790527
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1509590, 22.1240845
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2391586, 19.2338181
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5848064, 10.5749664
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4412880, 12.4410629
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0934639, 14.0851440
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7790565, 12.7681255
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1653500, 11.1579552
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3479767, 9.3434238
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0086231, 12.0081940
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8113403, 18.8102875
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9489975, 11.9429779
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5011101, 12.4905014
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6234550, 11.6060181
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5672989, 12.5631256
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2049255, 12.1965866
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9921303, 11.9944916
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7226028, 17.7315483
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5371513, 13.5408802
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6250229, 15.6261559
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5839405, 12.5861435
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6131248, 13.6236267
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1399841, 17.1536255
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9937096, 15.0054474
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7006950, 12.7126770
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4139843, 11.4173069
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3746567, 11.3802319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 514

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1590

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4849002, upper bound: 12.5209467
time: 24.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4909300, upper bound: 12.5142189
time: 26.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8403893, 15.8352013
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8141747, 11.8102074
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7094536, 10.7072830
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.8013573, 12.8002701
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6621437, 12.6608601
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2355614, 14.2373581
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0465813, 11.0495682
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0824242, 13.0804901
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9604645, 13.9643250
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7647705, 9.7661247
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2444077, 18.2477036
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3229046, 13.3121338
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0901794, 16.0886345
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8719559, 14.8712349
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3137054, 25.3057175
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1418190, 13.1359787
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7841377, 12.7807713
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1482735, 22.1252327
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2380753, 19.2343674
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5822582, 10.5760517
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4385109, 12.4422455
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0903397, 14.0864716
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7789192, 12.7682762
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1638126, 11.1586704
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3469772, 9.3439617
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0072956, 12.0088120
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8114548, 18.8091431
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9482422, 11.9433403
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4990158, 12.4913940
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6230278, 11.6062450
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5635452, 12.5647945
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2016792, 12.1979675
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9926491, 11.9935493
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7228851, 17.7308731
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5379829, 13.5389156
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6252213, 15.6256866
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5846310, 12.5845337
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6136665, 13.6223602
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1416779, 17.1496582
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9943466, 15.0039597
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7010918, 12.7119370
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4150829, 11.4147282
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3758202, 11.3774872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1372

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1744

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4583496, upper bound: 12.4793938
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4571791, upper bound: 12.4805638
time: 17.44 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 33, lower bound: -12.5190405, upper bound: 12.4857123
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.30
Output dim: 33, lower bound: -12.5023316, upper bound: 12.5009923
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.30
Output dim: 33, lower bound: -12.5099880, upper bound: 12.4921303
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.30
Output dim: 33, lower bound: -12.5023017, upper bound: 12.4998069
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 33, lower bound: -12.5204471, upper bound: 12.4878224
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 33, lower bound: -12.5216731, upper bound: 12.4866135
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 33, lower bound: -12.4741717, upper bound: 12.5158435
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.30
Output dim: 33, lower bound: -12.4897169, upper bound: 12.5003530
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.30
Output dim: 33, lower bound: -12.4849002, upper bound: 12.5209467
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.30
Output dim: 33, lower bound: -12.4909300, upper bound: 12.5142189
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 26.30
Output dim: 33, lower bound: -12.4583496, upper bound: 12.4793938
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 26.30
Output dim: 33, lower bound: -12.4571791, upper bound: 12.4805638

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8353348, 15.8484917
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8067703, 11.8152046
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7009468, 10.7112083
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7975731, 12.8087387
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6702042, 12.6827049
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2347794, 14.2466469
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0486794, 11.0469532
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0696106, 13.0808411
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9747620, 13.9865494
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7658119, 9.7671547
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2466393, 18.2481079
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3562241, 13.3540821
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0952759, 16.0884094
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8674622, 14.8736382
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3043518, 25.3123474
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1475563, 13.1556702
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7801285, 12.7801208
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1632233, 22.1739273
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2346115, 19.2313004
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6094761, 10.6079445
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4437485, 12.4422073
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1175194, 14.1140728
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7842369, 12.7853222
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1883316, 11.1839275
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3551750, 9.3517151
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0159016, 12.0099144
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8099976, 18.8060150
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9430885, 11.9394188
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5170441, 12.5121002
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6333580, 11.6349716
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5916195, 12.5808868
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2219772, 12.2209702
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9937248, 11.9925079
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7257385, 17.7176247
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5378666, 13.5351372
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6240692, 15.6205139
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5872841, 12.5846443
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6201553, 13.6086655
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1565933, 17.1575203
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0114403, 15.0068970
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7175655, 12.7115784
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4147873, 11.4120750
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3794270, 11.3772030

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 910

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1628

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5184123, upper bound: 12.4853002
time: 13.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5186456, upper bound: 12.4850661
time: 16.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8231812, 15.8355675
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7919922, 11.8001442
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6881638, 10.6975365
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7855721, 12.7966690
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6475372, 12.6599522
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2246780, 14.2353363
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0388603, 11.0362587
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0616951, 13.0703735
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9477959, 13.9592590
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7676811, 9.7678528
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2404366, 18.2395172
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3417473, 13.3358707
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0958939, 16.0927010
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8643837, 14.8690605
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2817230, 25.2921524
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1245270, 13.1360970
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7779427, 12.7707500
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1581650, 22.1695976
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2320938, 19.2316360
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5943794, 10.5890484
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4398651, 12.4368286
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1051254, 14.0992336
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7800903, 12.7848644
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1722641, 11.1676178
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3504982, 9.3470345
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0104923, 12.0046310
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8106079, 18.8105087
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9342117, 11.9319992
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4961853, 12.4918060
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6228504, 11.6262608
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5743256, 12.5618458
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2097836, 12.2052040
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9869614, 11.9859428
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7263298, 17.7190132
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5312080, 13.5288963
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6169891, 15.6124878
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5762844, 12.5734711
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6102276, 13.5979233
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1540718, 17.1569176
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0110931, 15.0041962
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7137489, 12.7069359
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4031525, 11.3987999
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3750286, 11.3728848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1459

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1630

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5200156, upper bound: 12.4731417
time: 6.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5058117, upper bound: 12.4873898
time: 22.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8240509, 15.8353500
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7923241, 11.8003998
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6888161, 10.6976433
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7876854, 12.7957573
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6473198, 12.6602688
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2258911, 14.2346497
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0398483, 11.0357361
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0619431, 13.0703316
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9481277, 13.9595718
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7677193, 9.7687874
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2391090, 18.2432098
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3404312, 13.3365974
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0955276, 16.0934029
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8643456, 14.8691216
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2817841, 25.2922974
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1253624, 13.1360817
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7771111, 12.7728348
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1569901, 22.1713028
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2321777, 19.2316437
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5937195, 10.5910988
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4399071, 12.4367676
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1044579, 14.1009750
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7799072, 12.7856026
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1722717, 11.1679649
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3496590, 9.3482475
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0098934, 12.0060158
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8104935, 18.8106766
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9342957, 11.9318314
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4965134, 12.4914055
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6222782, 11.6265335
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5738678, 12.5631332
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2090244, 12.2072678
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9875374, 11.9851151
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7261086, 17.7192650
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5319328, 13.5280075
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6180878, 15.6117325
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5778751, 12.5728226
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6098995, 13.5984688
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1554489, 17.1558609
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0098495, 15.0066910
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7139320, 12.7067642
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4036980, 11.3985252
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3755245, 11.3726025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1649

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 573

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4839065, upper bound: 12.4859365
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5210002, upper bound: 12.4488100
time: 17.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.7880745, 15.7620964
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7845688, 11.7680645
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6949005, 10.6829948
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7751122, 12.7574539
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6318283, 12.6081619
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2047119, 14.1837463
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0253296, 11.0328693
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0496674, 13.0347023
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9194908, 13.8956718
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7620468, 9.7570038
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2128067, 18.2095909
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2580872, 13.2744637
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0419388, 16.0574303
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8694344, 14.8616028
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3171997, 25.3074112
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0872040, 13.0649986
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7605667, 12.7639160
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1316223, 22.1265450
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2287750, 19.2338486
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5756779, 10.5813370
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4344292, 12.4355698
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0647316, 14.0705013
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7727966, 12.7672024
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1331520, 11.1457462
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3238239, 9.3322926
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -11.9973812, 12.0026817
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7955933, 18.8008575
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9427414, 11.9468727
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4917450, 12.5008621
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6026497, 11.6041164
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5194130, 12.5370960
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.1830864, 12.1872253
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9831009, 11.9865112
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7018433, 17.7123337
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5256577, 13.5313034
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6240845, 15.6315002
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5806999, 12.5850182
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.5947304, 13.6072922
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1575890, 17.1521797
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9644203, 14.9731140
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6793633, 12.6871281
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4153957, 11.4222965
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3433304, 11.3538742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1387

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 620

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4666030, upper bound: 12.5081830
time: 7.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4665976, upper bound: 12.5081870
time: 6.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8395004, 15.8267365
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.8114738, 11.8040504
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.7033005, 10.6942863
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.8082123, 12.8041306
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6486206, 12.6361008
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2423477, 14.2403603
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -10.9951210, 11.0121059
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0858650, 13.0788765
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9678307, 13.9675598
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7594070, 9.7615204
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2311478, 18.2375565
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3305397, 13.3227234
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0736160, 16.0827713
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8749123, 14.8733711
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2810669, 25.2551193
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1277580, 13.1107254
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7717628, 12.7747364
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1105347, 22.0693626
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2288055, 19.2195892
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5887985, 10.5798626
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4467239, 12.4505844
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0943718, 14.0917263
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7523804, 12.7325974
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1697598, 11.1634769
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3337936, 9.3246574
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0083885, 12.0079689
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7944489, 18.7878952
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9442558, 11.9376907
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4984188, 12.4874687
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6128464, 11.5908089
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5681190, 12.5657749
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2112083, 12.2047806
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9924126, 12.0024452
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7152786, 17.7257156
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5358238, 13.5385246
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6277199, 15.6282272
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5831032, 12.5867004
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.5834732, 13.6026001
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1396980, 17.1533737
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9801273, 14.9951210
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6670437, 12.6890011
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3992500, 11.4119625
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3199425, 11.3388042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1493

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1665

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4765874, upper bound: 12.5199822
time: 6.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4839334, upper bound: 12.5126089
time: 9.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 18.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.52
Output dim: 33, lower bound: -12.5184123, upper bound: 12.4853002
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.52
Output dim: 33, lower bound: -12.5186456, upper bound: 12.4850661
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.52
Output dim: 33, lower bound: -12.5200156, upper bound: 12.4731417
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.52
Output dim: 33, lower bound: -12.5058117, upper bound: 12.4873898
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.52
Output dim: 33, lower bound: -12.4839065, upper bound: 12.4859365
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 18.52
Output dim: 33, lower bound: -12.5210002, upper bound: 12.4488100
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.52
Output dim: 33, lower bound: -12.4666030, upper bound: 12.5081830
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.52
Output dim: 33, lower bound: -12.4665976, upper bound: 12.5081870
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 18.52
Output dim: 33, lower bound: -12.4765874, upper bound: 12.5199822
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.52
Output dim: 33, lower bound: -12.4839334, upper bound: 12.5126089

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8135796, 15.8200378
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7913399, 11.7949486
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6866837, 10.6926727
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7730865, 12.7769165
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6650238, 12.6714420
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2075195, 14.2112694
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0429726, 11.0415668
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0490799, 13.0539436
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9535866, 13.9583321
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7681580, 9.7685070
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2501869, 18.2512512
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3549252, 13.3548450
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0867157, 16.0815964
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8589249, 14.8624344
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3253784, 25.3299713
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1359024, 13.1384468
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7791977, 12.7793579
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1784019, 22.1829262
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2298203, 19.2269287
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6060562, 10.6058064
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4415092, 12.4400711
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1186390, 14.1153812
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7916756, 12.7919006
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1774044, 11.1757736
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3387184, 9.3381920
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0093994, 12.0040932
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8023376, 18.8002319
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9390793, 11.9363861
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5016346, 12.5006561
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6287231, 11.6304359
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5741749, 12.5671329
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2216225, 12.2206726
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9968033, 11.9956474
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7306976, 17.7234917
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5252266, 13.5254841
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6347313, 15.6346512
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5839787, 12.5822525
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6166382, 13.6097717
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1587753, 17.1594238
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0041962, 15.0016174
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7090645, 12.7056713
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4191628, 11.4173565
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3749199, 11.3729439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1634

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5104940, upper bound: 12.4612744
time: 19.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4944405, upper bound: 12.4774804
time: 19.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8068810, 15.8267288
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7865143, 11.7997742
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6824112, 10.6969452
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7657471, 12.7842522
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6589394, 12.6775265
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1993942, 14.2193909
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0432930, 11.0412445
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0427132, 13.0603104
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9465446, 13.9653740
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7671623, 9.7694988
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2497902, 18.2516556
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3569851, 13.3527851
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0884628, 16.0798492
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8562584, 14.8651009
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.3219757, 25.3333664
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1303329, 13.1440163
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7793617, 12.7791901
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1722221, 22.1891098
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2302399, 19.2265091
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.6073380, 10.6045227
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4416122, 12.4399681
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1188297, 14.1151905
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7908134, 12.7927628
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1801777, 11.1730003
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3416519, 9.3352566
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0100822, 12.0034142
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8042145, 18.7983551
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9400558, 11.9354095
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.5055943, 12.4966927
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6288223, 11.6303368
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5778675, 12.5634441
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2216835, 12.2206116
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9968643, 11.9955826
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7316132, 17.7225761
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5282135, 13.5225010
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6382027, 15.6311798
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5848942, 12.5813370
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6212654, 13.6051483
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1584930, 17.1597061
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0061607, 14.9996490
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7116585, 12.7030792
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4200668, 11.4164505
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3751678, 11.3726959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1665

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1763

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5107260, upper bound: 12.4610414
time: 18.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4946739, upper bound: 12.4772472
time: 12.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8238373, 15.8358765
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7919998, 11.8001404
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6839828, 10.6926422
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7735291, 12.7806549
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6479874, 12.6583652
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2113800, 14.2183037
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0305595, 11.0256157
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0589943, 13.0670204
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9468269, 13.9559135
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7676582, 9.7678261
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2332764, 18.2326736
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3204250, 13.3193741
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0888596, 16.0863304
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8585396, 14.8613167
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2576523, 25.2725143
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1253319, 13.1365452
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7730293, 12.7663784
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1421585, 22.1564980
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2245636, 19.2259445
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5841599, 10.5806274
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4372635, 12.4345970
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0938931, 14.0904007
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7741051, 12.7801189
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1586647, 11.1568718
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3354988, 9.3357010
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0032845, 11.9991989
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8008041, 18.8031082
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9252586, 11.9252090
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4832230, 12.4814873
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6075935, 11.6143990
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5540504, 12.5465279
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2043877, 12.2008286
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9845505, 11.9813385
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7227020, 17.7149048
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5340710, 13.5308304
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6139946, 15.6092682
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5689125, 12.5641785
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6107254, 13.5985031
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1529350, 17.1522560
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -15.0046921, 14.9954453
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7080822, 12.6998177
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3976746, 11.3921165
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3693123, 11.3658924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 616

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1590

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5125958, upper bound: 12.4723541
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5193245, upper bound: 12.4663279
time: 13.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8320389, 15.8416595
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7854424, 11.7938614
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6774521, 10.6777191
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7578583, 12.7556381
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6733170, 12.6795082
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1849937, 14.1779480
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0442982, 11.0389938
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0855904, 13.0870323
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9660149, 13.9704208
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7623711, 9.7624054
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2370148, 18.2406998
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3067284, 13.3111916
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0276871, 16.0416603
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8402557, 14.8371582
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2919083, 25.3045425
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1275864, 13.1382713
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7854080, 12.7787247
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1541443, 22.1690254
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2290573, 19.2292328
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5930653, 10.5907726
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4541664, 12.4469414
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0966949, 14.0951233
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7625656, 12.7725258
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1619225, 11.1603069
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3420792, 9.3425312
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0108147, 12.0073662
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7764130, 18.7849808
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9078751, 11.9119148
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4917145, 12.4898567
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5835304, 11.5983486
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5615463, 12.5531654
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2108994, 12.2128868
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9930954, 11.9913635
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7163353, 17.7063103
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5353966, 13.5301819
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6143379, 15.6069870
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5823307, 12.5762978
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6255741, 13.6146355
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1569023, 17.1558685
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9949970, 14.9869881
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7124462, 12.7049332
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4245148, 11.4227085
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3936672, 11.3892651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1459

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1664

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5184749, upper bound: 12.4462594
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5184706, upper bound: 12.4462636
time: 6.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8154221, 15.7976875
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7873688, 11.7730274
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6924438, 10.6813164
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7961578, 12.7879219
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6279182, 12.6095772
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2300644, 14.2271500
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -10.9947968, 11.0132637
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0693169, 13.0605392
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9461098, 13.9395103
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7558174, 9.7579632
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2265015, 18.2323074
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3304958, 13.3227558
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0731964, 16.0801392
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8743629, 14.8726463
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2784500, 25.2505493
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1027298, 13.0787945
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7724075, 12.7773781
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0920372, 22.0422935
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2293930, 19.2199173
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5782490, 10.5714073
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4374275, 12.4439240
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0932961, 14.0909805
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7535286, 12.7327156
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1570816, 11.1534348
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3273335, 9.3197479
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0000496, 12.0017319
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7996902, 18.7905045
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9442711, 11.9377060
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4794464, 12.4730530
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6041603, 11.5815411
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5562134, 12.5575638
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.1977882, 12.1941071
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9886398, 11.9990540
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7020988, 17.7135506
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5261459, 13.5292377
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6123581, 15.6164360
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5714893, 12.5772018
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.5715580, 13.5933266
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1412010, 17.1545296
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9644508, 14.9832687
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6499290, 12.6755848
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4003010, 11.4130421
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3191185, 11.3362331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1405

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1422

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.4715629, upper bound: 12.5193530
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4715629, upper bound: 12.5149729
time: 28.95 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 37.35 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 37.35
Output dim: 33, lower bound: -12.5104940, upper bound: 12.4612744
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 37.35
Output dim: 33, lower bound: -12.4944405, upper bound: 12.4774804
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 37.35
Output dim: 33, lower bound: -12.5107260, upper bound: 12.4610414
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 37.35
Output dim: 33, lower bound: -12.4946739, upper bound: 12.4772472
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 37.35
Output dim: 33, lower bound: -12.5125958, upper bound: 12.4723541
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.35
Output dim: 33, lower bound: -12.5193245, upper bound: 12.4663279
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.35
Output dim: 33, lower bound: -12.5184749, upper bound: 12.4462594
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.35
Output dim: 33, lower bound: -12.5184706, upper bound: 12.4462636
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.35
Output dim: 33, lower bound: -12.4715629, upper bound: 12.5193530
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 37.35
Output dim: 33, lower bound: -12.4715629, upper bound: 12.5149729

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8154449, 15.8348694
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7857552, 11.7975559
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6711578, 10.6851349
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7774811, 12.7865067
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6234283, 12.6444054
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2153473, 14.2225266
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -10.9934044, 10.9743080
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0589027, 13.0669212
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9495621, 13.9635620
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7629166, 9.7624950
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2231750, 18.2194061
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3324318, 13.3236771
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0789719, 16.0708580
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8608475, 14.8636932
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2084579, 25.2366791
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0989532, 13.1230450
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7687073, 12.7499619
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0874252, 22.1160583
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2103348, 19.2155914
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5890560, 10.5846195
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4467773, 12.4400291
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.1004734, 14.0913048
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7385750, 12.7534466
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1641808, 11.1612740
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3167324, 9.3215199
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0030594, 11.9989662
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7784042, 18.7862091
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9199753, 11.9204712
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4801884, 12.4787903
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5923862, 11.6037903
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5567017, 12.5473461
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2125740, 12.2071075
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9925041, 11.9816170
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7168617, 17.7075729
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5317230, 13.5295105
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6160583, 15.6119576
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5694695, 12.5633392
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.5896912, 13.5688438
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1526756, 17.1519699
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9943676, 14.9818649
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6844120, 12.6661682
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3923187, 11.3773746
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3278866, 11.3111801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1761

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 773

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5072015, upper bound: 12.4647520
time: 15.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5177431, upper bound: 12.4542065
time: 5.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8318024, 15.8425674
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7853889, 11.7937927
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6774559, 10.6777306
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7578125, 12.7560730
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6733017, 12.6795254
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1849098, 14.1789246
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0442848, 11.0388489
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0855446, 13.0874176
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9659615, 13.9707832
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7619286, 9.7633705
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2369080, 18.2408409
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3066597, 13.3111572
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0300903, 16.0414200
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8402481, 14.8371696
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2914200, 25.3050385
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1275749, 13.1382408
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7850266, 12.7798996
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1540756, 22.1689949
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2295990, 19.2292175
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5927563, 10.5909939
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4539413, 12.4474068
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0965195, 14.0955849
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7625618, 12.7725296
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1616440, 11.1606941
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3420105, 9.3425312
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0105934, 12.0078125
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7767487, 18.7849655
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9077339, 11.9118423
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4913025, 12.4903641
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5835228, 11.5983849
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5613708, 12.5538082
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2106934, 12.2129974
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9934196, 11.9913177
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7163506, 17.7062683
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5351791, 13.5299377
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6142426, 15.6071472
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5822697, 12.5762558
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6255589, 13.6145859
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1569099, 17.1558685
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9948902, 14.9871750
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7124596, 12.7049198
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4247894, 11.4225330
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3939323, 11.3890457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 590

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1492

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5183099, upper bound: 12.4446188
time: 14.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5173447, upper bound: 12.4460968
time: 5.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8320389, 15.8414154
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7853737, 11.7938614
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6774521, 10.6777191
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7578583, 12.7555962
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6733170, 12.6794987
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1849937, 14.1778641
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0441551, 11.0389938
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0855904, 13.0869865
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9660149, 13.9703636
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7623711, 9.7619629
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2370148, 18.2405891
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3066940, 13.3111916
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0274506, 16.0416603
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8402557, 14.8371506
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2919083, 25.3040543
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1275558, 13.1382713
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7854080, 12.7783432
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1541214, 22.1690254
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2290421, 19.2292328
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5930653, 10.5904636
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4541664, 12.4467125
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0966949, 14.0949478
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7625656, 12.7725258
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1619225, 11.1600266
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3420792, 9.3424644
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0108147, 12.0071449
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7763977, 18.7849808
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9078751, 11.9117775
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4917145, 12.4894524
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5835304, 11.5983410
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5615463, 12.5529900
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2108994, 12.2126808
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9930496, 11.9913635
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7162895, 17.7063103
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5353966, 13.5299644
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6143379, 15.6068916
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5823307, 12.5762405
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6255741, 13.6146164
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1569023, 17.1558685
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9949970, 14.9868774
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7124367, 12.7049332
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4243393, 11.4227085
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3934479, 11.3892651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1725

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1478

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5110208, upper bound: 12.4391452
time: 18.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5112818, upper bound: 12.4388843
time: 18.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8128052, 15.7943764
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7850075, 11.7700443
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6914024, 10.6799507
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7934456, 12.7843513
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6254082, 12.6062126
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2280579, 14.2245178
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -10.9945297, 11.0129623
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0678787, 13.0587273
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9412079, 13.9334145
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7550125, 9.7569809
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2265320, 18.2323112
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3273544, 13.3198433
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0729599, 16.0798569
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8743515, 14.8726196
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2786713, 25.2503052
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0997543, 13.0748348
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7714081, 12.7766037
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0920067, 22.0420799
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2290878, 19.2196503
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5747585, 10.5686264
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4365463, 12.4432411
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0927658, 14.0904732
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7534218, 12.7326202
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1546555, 11.1515942
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3260880, 9.3188190
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -11.9989166, 12.0008774
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.8001099, 18.7907028
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9442482, 11.9377060
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4761124, 12.4704628
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.6041813, 11.5816803
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5540142, 12.5556793
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.1945019, 12.1915169
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9875793, 11.9981308
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7017822, 17.7133484
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5261822, 13.5292892
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6115150, 15.6158066
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5707188, 12.5765915
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.5703068, 13.5923615
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1421318, 17.1548233
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9637947, 14.9827576
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6496010, 12.6753044
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3990211, 11.4119511
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3192253, 11.3361034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1398
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1604

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4666194, upper bound: 12.5143935
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4669512, upper bound: 12.5140753
time: 23.16 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 31.77 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 31.77
Output dim: 33, lower bound: -12.5072015, upper bound: 12.4647520
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 31.77
Output dim: 33, lower bound: -12.5177431, upper bound: 12.4542065
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 31.77
Output dim: 33, lower bound: -12.5183099, upper bound: 12.4446188
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 31.77
Output dim: 33, lower bound: -12.5173447, upper bound: 12.4460968
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 31.77
Output dim: 33, lower bound: -12.5110208, upper bound: 12.4391452
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 31.77
Output dim: 33, lower bound: -12.5112818, upper bound: 12.4388843
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 31.77
Output dim: 33, lower bound: -12.4666194, upper bound: 12.5143935
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 31.77
Output dim: 33, lower bound: -12.4669512, upper bound: 12.5140753

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8147354, 15.8348694
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7852478, 11.7975559
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6702042, 10.6851349
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7763062, 12.7865067
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6218376, 12.6444054
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.2129250, 14.2225266
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -10.9934044, 10.9740295
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0555954, 13.0669212
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9492302, 13.9635620
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7629166, 9.7623558
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2231750, 18.2190933
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3303642, 13.3236771
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0789719, 16.0683174
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8605423, 14.8636932
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2052917, 25.2366791
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0987663, 13.1230450
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7664680, 12.7499619
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0840683, 22.1160583
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2096863, 19.2155914
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5870686, 10.5846195
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4452171, 12.4400291
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0984535, 14.0913048
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7373543, 12.7534466
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1636505, 11.1612740
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3163433, 9.3215199
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0027771, 11.9989662
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7784042, 18.7858582
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9186096, 11.9204712
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4787483, 12.4787903
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5909386, 11.6037903
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5549297, 12.5473461
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2101021, 12.2071075
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9925041, 11.9811783
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7168617, 17.7057266
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5317230, 13.5278530
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6160583, 15.6111221
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5694695, 12.5624886
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.5896912, 13.5665703
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1526756, 17.1491890
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9943676, 14.9800949
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6844120, 12.6649284
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3923187, 11.3757248
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3278866, 11.3087006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1779

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5167175, upper bound: 12.4269202
time: 17.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4906147, upper bound: 12.4531807
time: 6.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8291855, 15.8403053
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7824783, 11.7913513
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6717987, 10.6731224
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7530022, 12.7522697
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6639633, 12.6717472
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1793823, 14.1745262
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0443039, 11.0388851
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0806999, 13.0836563
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9553032, 13.9619064
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7620621, 9.7634354
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2346573, 18.2393570
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3022270, 13.3059139
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0292435, 16.0403976
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8371620, 14.8348656
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2908249, 25.3048096
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1239281, 13.1350784
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7839928, 12.7782192
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1541862, 22.1690369
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2275085, 19.2265167
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5897121, 10.5871391
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4531441, 12.4465218
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0934029, 14.0916214
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7598534, 12.7690506
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1574879, 11.1554260
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3396301, 9.3394718
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0077744, 12.0042343
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7749176, 18.7826233
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9040337, 11.9071312
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4852409, 12.4827805
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5797729, 11.5935078
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5557098, 12.5467548
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2090340, 12.2108879
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9941406, 11.9922371
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7164688, 17.7062035
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5323601, 13.5266190
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6137161, 15.6056061
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5798550, 12.5731392
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6229362, 13.6112671
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1585884, 17.1581955
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9942818, 14.9871712
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7129707, 12.7056389
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4239159, 11.4213371
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3938332, 11.3890724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 590

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 726

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5181374, upper bound: 12.4380160
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5116639, upper bound: 12.4444452
time: 6.47 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8294754, 15.8399506
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7830124, 11.7908821
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6728516, 10.6720695
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7540016, 12.7512741
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6655273, 12.6701870
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1804962, 14.1734047
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0443192, 11.0388718
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0819397, 13.0825729
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9570808, 13.9601250
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7619934, 9.7635040
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2355576, 18.2386055
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3014069, 13.3067341
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0290680, 16.0405731
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8379440, 14.8340874
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2912827, 25.3044434
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1244164, 13.1345901
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7833443, 12.7788677
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1541100, 22.1691093
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2269058, 19.2271194
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5889034, 10.5879478
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4530525, 12.4465790
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0925560, 14.0924683
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7590790, 12.7698250
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1563740, 11.1565418
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3389511, 9.3401508
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0070076, 12.0049973
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7743988, 18.7831421
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9030228, 11.9081421
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4837227, 12.4842987
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5786476, 11.5946331
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5543175, 12.5481453
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2085762, 12.2113419
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9943390, 11.9920387
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7162857, 17.7063904
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5318680, 13.5270309
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6127090, 15.6067848
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5791569, 12.5739784
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6222420, 13.6121674
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1592293, 17.1575546
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9948921, 14.9865646
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7131805, 12.7054291
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4235916, 11.4216576
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3939133, 11.3889580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1648

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1634

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5147640, upper bound: 12.4456457
time: 18.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5169141, upper bound: 12.4434944
time: 5.85 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 26.90 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 26.90
Output dim: 33, lower bound: -12.5167175, upper bound: 12.4269202
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 26.90
Output dim: 33, lower bound: -12.4906147, upper bound: 12.4531807
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 26.90
Output dim: 33, lower bound: -12.5181374, upper bound: 12.4380160
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 26.90
Output dim: 33, lower bound: -12.5116639, upper bound: 12.4444452
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 26.90
Output dim: 33, lower bound: -12.5147640, upper bound: 12.4456457
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 26.90
Output dim: 33, lower bound: -12.5169141, upper bound: 12.4434944

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.7624664, 15.7955780
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7423286, 11.7653770
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6336670, 10.6576767
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7382698, 12.7581139
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.5684662, 12.6042042
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1673851, 14.1884804
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -10.9878616, 10.9673691
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -12.9966431, 13.0228004
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.8823509, 13.9131584
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7424469, 9.7467365
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.1736031, 18.1811066
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3183861, 13.3159389
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0609741, 16.0453796
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8449898, 14.8531761
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.1652374, 25.2054138
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0774078, 13.1069508
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7394638, 12.7288666
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0452881, 22.0862389
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2017822, 19.2030334
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5851269, 10.5826340
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4441452, 12.4398308
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0973854, 14.0907249
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7312965, 12.7456017
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1517067, 11.1445370
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3128643, 9.3153820
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -11.9940987, 11.9870796
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7720795, 18.7764893
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9077187, 11.9044800
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4527893, 12.4444542
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5879745, 11.6005440
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5554714, 12.5460243
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2047291, 12.2013245
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9808311, 11.9659462
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.6959305, 17.6786652
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5075684, 13.4958630
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5886612, 15.5749397
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5455723, 12.5310898
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.5728836, 13.5443420
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1504211, 17.1468124
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9923630, 14.9779739
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6750088, 12.6524887
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3802280, 11.3603706
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3315754, 11.3124046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 606

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1432

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4982751, upper bound: 12.4087202
time: 14.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4986856, upper bound: 12.4083524
time: 32.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8074875, 15.8287010
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7731400, 11.7867889
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6596680, 10.6662407
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7392616, 12.7442932
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6572113, 12.6711197
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1602173, 14.1632576
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0443192, 11.0372181
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0660973, 13.0764580
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9450722, 13.9545898
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7608757, 9.7643566
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2341995, 18.2393227
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3021603, 13.3056374
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0218964, 16.0299911
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8295746, 14.8309669
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2835770, 25.3091736
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1099854, 13.1278973
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7775192, 12.7722683
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1358757, 22.1656189
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2248459, 19.2238693
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5798092, 10.5809479
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4527245, 12.4457932
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0877838, 14.0855370
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7521057, 12.7678185
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1551723, 11.1533566
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3365173, 9.3357735
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0076141, 12.0040398
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7674332, 18.7742004
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9037170, 11.9068871
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4834023, 12.4813766
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5638504, 11.5853558
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5555725, 12.5437241
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2012138, 12.2049789
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9941368, 11.9921951
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7195473, 17.7032204
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5239201, 13.5133877
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6150551, 15.6036644
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5758438, 12.5667725
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6200104, 13.6005630
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1508751, 17.1488953
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9915962, 14.9795036
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7047863, 12.6897564
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4224262, 11.4176884
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3927689, 11.3852253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1544

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1600

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5179804, upper bound: 12.4250351
time: 15.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5050984, upper bound: 12.4378586
time: 7.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.8199234, 15.8314590
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7763710, 11.7850723
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6668701, 10.6665192
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7447319, 12.7435188
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6602516, 12.6664829
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1757584, 14.1689224
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0420895, 11.0369282
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0825424, 13.0831375
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9382324, 13.9457054
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7620850, 9.7641983
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2368774, 18.2409439
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3013897, 13.3043003
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0324783, 16.0454178
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8350792, 14.8312340
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2955780, 25.3091507
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.1144524, 13.1258736
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7867470, 12.7798786
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1523590, 22.1675110
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2194977, 19.2173462
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5827751, 10.5798454
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4461403, 12.4374466
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0846786, 14.0817871
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7560730, 12.7665176
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1502533, 11.1484528
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3324394, 9.3315887
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -11.9996643, 11.9952888
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7745132, 18.7832489
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.8983612, 11.9028625
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4742413, 12.4719696
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5800209, 11.5958061
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5385933, 12.5272026
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2018852, 12.2026138
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9949608, 11.9942551
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7169113, 17.7060661
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5326729, 13.5282383
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6147881, 15.6082993
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5786800, 12.5737000
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6176281, 13.6064377
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1617393, 17.1634521
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9962006, 14.9883919
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.7131920, 12.7054539
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4183788, 11.4172573
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3871613, 11.3843021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2040
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1646

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 580

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5144665, upper bound: 12.4393975
time: 10.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5111752, upper bound: 12.4412918
time: 6.40 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 18.89 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 18.89
Output dim: 33, lower bound: -12.4982751, upper bound: 12.4087202
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 18.89
Output dim: 33, lower bound: -12.4986856, upper bound: 12.4083524
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 18.89
Output dim: 33, lower bound: -12.5179804, upper bound: 12.4250351
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 18.89
Output dim: 33, lower bound: -12.5050984, upper bound: 12.4378586
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 18.89
Output dim: 33, lower bound: -12.5144665, upper bound: 12.4393975
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 18.89
Output dim: 33, lower bound: -12.5111752, upper bound: 12.4412918

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.7868156, 15.8132706
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7610054, 11.7774410
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6487541, 10.6583672
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7358322, 12.7425804
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6441345, 12.6614017
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1499557, 14.1561508
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0339317, 11.0236664
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0503731, 13.0646629
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9317055, 13.9447708
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7564583, 9.7609863
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2269478, 18.2337189
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.3009224, 13.3045521
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0219650, 16.0268326
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8259811, 14.8288498
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2544785, 25.2873383
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0981407, 13.1190891
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7757263, 12.7706203
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.1195412, 22.1531639
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2237015, 19.2228775
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5786228, 10.5799847
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4489555, 12.4429779
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0852489, 14.0833607
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7495575, 12.7657166
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1548023, 11.1531219
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3346519, 9.3344593
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0062428, 12.0032768
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7678070, 18.7737732
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.9027367, 11.9054375
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4828205, 12.4808197
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5615120, 11.5834007
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5541687, 12.5425701
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2003651, 12.2043076
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9852295, 11.9802628
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7144852, 17.6964722
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5152054, 13.5017796
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6086884, 15.5951767
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5650902, 12.5526161
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6139126, 13.5923653
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1455154, 17.1417465
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9834518, 14.9689331
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6958008, 12.6778030
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.4099293, 11.4011593
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3846760, 11.3743649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2039
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1634

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 590

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4973160, upper bound: 12.4245303
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5176013, upper bound: 12.4074858
time: 15.73 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 23.51 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 23.51
Output dim: 33, lower bound: -12.4973160, upper bound: 12.4245303
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 23.51
Output dim: 33, lower bound: -12.5176013, upper bound: 12.4074858

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.7699432, 15.7988739
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7488098, 11.7675953
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6507721, 10.6615219
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7403946, 12.7473831
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6341476, 12.6524754
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1555824, 14.1623001
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0240364, 11.0104828
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0421333, 13.0581169
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.9220123, 13.9358063
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7551498, 9.7594700
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2217789, 18.2285538
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2900124, 13.2961769
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0292969, 16.0337067
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8267441, 14.8296051
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2131195, 25.2554932
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0854034, 13.1087875
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7734489, 12.7680225
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0855637, 22.1267815
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2168961, 19.2177658
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5766964, 10.5790367
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4475975, 12.4417343
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0773735, 14.0774479
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7396584, 12.7582798
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1509666, 11.1504021
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3271275, 9.3288059
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -12.0049305, 12.0025864
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7620239, 18.7694016
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.8974533, 11.9009819
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4812489, 12.4797783
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5476837, 11.5730171
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5457458, 12.5355453
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.2005024, 12.2050896
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9759674, 11.9674683
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7059097, 17.6859093
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5126534, 13.4968700
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6020088, 15.5865555
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5540886, 12.5379677
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6020927, 13.5782127
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1433449, 17.1363640
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9708843, 14.9522820
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6828537, 12.6610451
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3975449, 11.3847713
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3758316, 11.3629265

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2038
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 574

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 616

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5175151, upper bound: 12.4008247
time: 22.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5108797, upper bound: 12.4073892
time: 6.61 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 31.50 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 31.50
Output dim: 33, lower bound: -12.5175151, upper bound: 12.4008247
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 31.50
Output dim: 33, lower bound: -12.5108797, upper bound: 12.4073892

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.7590179, 15.7858658
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7437668, 11.7629528
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6466599, 10.6549187
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7342148, 12.7370148
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6119270, 12.6272469
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1459618, 14.1469498
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0249557, 11.0096550
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0390511, 13.0551147
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.8933601, 13.9007568
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7516060, 9.7545815
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2213669, 18.2263756
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2453079, 13.2657070
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0202789, 16.0278473
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8241882, 14.8249702
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.2078705, 25.2544785
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0697021, 13.0929089
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7703667, 12.7712154
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0434837, 22.0985184
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2128220, 19.2178726
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5414867, 10.5516586
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4460258, 12.4399872
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0454597, 14.0534019
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7192764, 12.7439976
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1179924, 11.1233654
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3137856, 9.3196793
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -11.9962444, 11.9958878
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7572021, 18.7671890
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.8878136, 11.8966370
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4505768, 12.4569626
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.5183754, 11.5557003
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.5175133, 12.5149059
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.1748371, 12.1850777
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9755058, 11.9663506
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7065086, 17.6834030
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5091419, 13.4918156
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.6020966, 15.5863266
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5497818, 12.5329208
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6010132, 13.5760994
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1380348, 17.1208839
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9576988, 14.9338188
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6756954, 12.6488686
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3979874, 11.3852768
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3744125, 11.3606815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2037
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1562

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 541

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4784966, upper bound: 12.4000904
time: 21.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5167823, upper bound: 12.3615535
time: 17.48 seconds

## Summary of splitting (split count: 11)
- Time for RS candidates: 41.71 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 12, time: 41.71
Output dim: 33, lower bound: -12.4784966, upper bound: 12.4000904
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 12, time: 41.71
Output dim: 33, lower bound: -12.5167823, upper bound: 12.3615535

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.7571564, 15.7840958
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7371864, 11.7596378
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6419907, 10.6499062
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7309952, 12.7327881
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.6103096, 12.6263351
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1411400, 14.1413460
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0246601, 11.0087929
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0313950, 13.0508575
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.8921928, 13.8996353
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7461967, 9.7488346
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.2080193, 18.2181129
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2139492, 13.2539902
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0131454, 16.0221291
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8209572, 14.8161507
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.1853561, 25.2425537
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0681419, 13.0922432
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7504044, 12.7610207
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -22.0167313, 22.0862007
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2081223, 19.2157516
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5298672, 10.5469856
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4423294, 12.4376678
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0238342, 14.0449543
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7115440, 12.7406197
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1075668, 11.1184998
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3059349, 9.3153019
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -11.9917107, 11.9928894
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7455215, 18.7616653
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.8733063, 11.8910980
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4448795, 12.4547768
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.4994774, 11.5494003
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.4948120, 12.5047169
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.1671886, 12.1819038
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9701080, 11.9593353
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.7030602, 17.6736641
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.5055485, 13.4825974
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5989151, 15.5773621
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5449791, 12.5228996
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.6009483, 13.5759850
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1311722, 17.1036758
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9537125, 14.9238319
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6711903, 12.6422329
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3978882, 11.3851585
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3733158, 11.3585243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2036
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1616

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5167040, upper bound: 12.3516733
time: 17.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5020042, upper bound: 12.3609105
time: 6.94 seconds

## Summary of splitting (split count: 12)
- Time for RS candidates: 26.52 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 13, time: 26.52
Output dim: 33, lower bound: -12.5167040, upper bound: 12.3516733
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 13, time: 26.52
Output dim: 33, lower bound: -12.5020042, upper bound: 12.3609105

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.7203674, 15.7564964
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7168198, 11.7440376
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6231995, 10.6362228
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7273216, 12.7327003
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.5882492, 12.6097889
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1222534, 14.1286049
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0108910, 10.9908218
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0021667, 13.0289459
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.8704681, 13.8833809
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7359009, 9.7407475
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.1903458, 18.2048340
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.2069817, 13.2472076
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0232468, 16.0267181
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8116837, 14.8099937
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.1406860, 25.2090225
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0506592, 13.0791359
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7366028, 12.7492523
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -21.9824219, 22.0599976
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2087402, 19.2162781
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5260391, 10.5453453
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4366875, 12.4335594
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0156403, 14.0382118
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7046547, 12.7352028
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1065750, 11.1180286
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3049107, 9.3146000
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -11.9896832, 11.9919281
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7481613, 18.7618332
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.8705559, 11.8869591
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4443893, 12.4542427
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.4967213, 11.5482559
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.4938698, 12.5038052
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.1622047, 12.1788177
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9574280, 11.9421196
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.6931343, 17.6605453
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.4886951, 13.4601250
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5870781, 15.5615768
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5278969, 12.5002670
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.5900574, 13.5614662
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1208916, 17.0895882
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9419708, 14.9087219
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6552296, 12.6209641
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3806992, 11.3624554
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3625488, 11.3441544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2035
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1438

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1697

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5164192, upper bound: 12.3511746
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5161981, upper bound: 12.3513790
time: 18.44 seconds

## Summary of splitting (split count: 13)
- Time for RS candidates: 26.99 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 14, time: 26.99
Output dim: 33, lower bound: -12.5164192, upper bound: 12.3511746
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 14, time: 26.99
Output dim: 33, lower bound: -12.5161981, upper bound: 12.3513790

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.7164841, 15.7535019
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7165985, 11.7438965
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6269035, 10.6395874
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7267799, 12.7322197
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.5893440, 12.6113243
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1222687, 14.1286240
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0135670, 10.9913292
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0010262, 13.0286636
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.8714714, 13.8845596
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7362823, 9.7412548
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.1665955, 18.1746330
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.1940746, 13.2336407
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0110550, 16.0112038
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8129883, 14.8106995
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.1357040, 25.2045898
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0455513, 13.0750217
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7221794, 12.7357712
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -21.9875412, 22.0651436
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2113495, 19.2186203
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5276794, 10.5473251
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4368744, 12.4337959
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0123978, 14.0356007
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.6978455, 12.7305870
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1068058, 11.1182995
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3052406, 9.3149586
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -11.9906349, 11.9931946
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7486801, 18.7622299
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.8638573, 11.8818207
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4335003, 12.4457283
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.4939060, 11.5457859
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.4935398, 12.5035095
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.1614113, 12.1783829
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9573822, 11.9420509
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.6931763, 17.6604729
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.4858513, 13.4570541
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5759201, 15.5513763
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5154171, 12.4892044
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.5902672, 13.5615768
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1257629, 17.0941200
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9356537, 14.9010048
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6544418, 12.6200638
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3835258, 11.3644695
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3592510, 11.3370132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2034
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1420

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 542

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4806649, upper bound: 12.3496414
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5158439, upper bound: 12.3234412
time: 12.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.7173691, 15.7526169
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7166824, 11.7438164
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6265526, 10.6399384
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7268372, 12.7321587
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.5897865, 12.6108818
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1222763, 14.1286201
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0114002, 10.9934998
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -13.0018921, 13.0277977
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.8716469, 13.8843842
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7364082, 9.7411270
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.1601410, 18.1810913
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.1934185, 13.2342949
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0077286, 16.0145302
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8123856, 14.8113022
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.1362534, 25.2040405
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0465393, 13.0740337
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7231216, 12.7348270
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -21.9875717, 22.0651131
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2110748, 19.2188950
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5280190, 10.5469856
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4369202, 12.4337502
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0130272, 14.0349693
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.7000427, 12.7283897
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1068439, 11.1182613
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3052711, 9.3149319
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -11.9909515, 11.9928780
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7485657, 18.7623444
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.8654213, 11.8802567
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4358768, 12.4433556
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.4942493, 11.5454426
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.4935780, 12.5034714
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.1617737, 12.1780205
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9573631, 11.9420738
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.6930771, 17.6605721
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.4856262, 13.4572792
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5768738, 15.5504227
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5168400, 12.4877815
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.5901680, 13.5616798
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1254196, 17.0944633
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9342499, 14.9024086
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6543274, 12.6201763
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3827171, 11.3652744
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3554173, 11.3408470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2034
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1405

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 894

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 33, lower bound: -12.5161397, upper bound: 12.3486027
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5134827, upper bound: 12.3513232
time: 15.51 seconds

## Summary of splitting (split count: 14)
- Time for RS candidates: 23.79 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 15, time: 23.79
Output dim: 33, lower bound: -12.4806649, upper bound: 12.3496414
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 15, time: 23.79
Output dim: 33, lower bound: -12.5158439, upper bound: 12.3234412
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 15, time: 23.79
Output dim: 33, lower bound: -12.5161397, upper bound: 12.3486027
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 15, time: 23.79
Output dim: 33, lower bound: -12.5134827, upper bound: 12.3513232

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.7002258, 15.7404289
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.6953506, 11.7277985
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6252632, 10.6386261
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7291946, 12.7351913
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.5875626, 12.6099167
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1173820, 14.1249008
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0148087, 10.9923553
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -12.9714432, 13.0059929
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.8649025, 13.8804245
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7230606, 9.7306232
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.1293526, 18.1463203
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.1373291, 13.1892376
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0066605, 16.0066376
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8121872, 14.8105965
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.0842819, 25.1634598
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0464668, 13.0762424
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.6794243, 12.7020683
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -21.9309464, 22.0216599
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2058563, 19.2138519
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5186672, 10.5423794
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4352150, 12.4337997
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -13.9732399, 14.0050163
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.6900330, 12.7246189
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1072273, 11.1207485
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3043423, 9.3153324
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -11.9899483, 11.9938812
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7414932, 18.7567444
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.8483810, 11.8685570
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4338150, 12.4459686
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.4661655, 11.5247478
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.4571571, 12.4731350
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.1511230, 12.1711922
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9411697, 11.9232864
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.6795044, 17.6423378
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.4668350, 13.4318199
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5577927, 15.5273247
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.4941502, 12.4610825
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.5858574, 13.5562935
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1019058, 17.0625267
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9234772, 14.8849907
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6430168, 12.6054535
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3764153, 11.3560390
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3616028, 11.3378658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2033
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1786

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1746

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5063127, upper bound: 12.2936991
time: 18.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4861416, upper bound: 12.3142773
time: 7.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -42.6874275, -18.9399910, -42.6874275, -18.9399910, -15.7104492, 15.7469482
1: -30.4739475, -12.9624367, -30.4739475, -12.9624367, -11.7092361, 11.7376976
2: -21.5952587, -5.9921961, -21.5952587, -5.9921961, -10.6243896, 10.6381531
3: -30.5879440, -12.4453697, -30.5879440, -12.4453697, -12.7210426, 12.7273941
4: -18.8445339, 0.6088285, -18.8445339, 0.6088285, -12.5837555, 12.6059322
5: -30.7314377, -9.8963623, -30.7314377, -9.8963623, -14.1179352, 14.1250572
6: -14.2883472, 1.3316171, -14.2883472, 1.3316171, -11.0118389, 10.9943314
7: -47.4976425, -25.9926605, -47.4976425, -25.9926605, -12.9969902, 13.0239220
8: -33.6759033, -12.1367245, -33.6759033, -12.1367245, -13.8632050, 13.8775902
9: -18.5999546, -6.9883556, -18.5999546, -6.9883556, -9.7355652, 9.7408237
10: -38.3487167, -15.4204731, -38.3487167, -15.4204731, -18.1614494, 18.1824684
11: -57.3802986, -34.7022781, -57.3802986, -34.7022781, -13.1917229, 13.2323227
12: -1.9409168, 17.9383068, -1.9409168, 17.9383068, -16.0091858, 16.0162048
13: -8.2643576, 9.4567671, -8.2643576, 9.4567671, -14.8111000, 14.8101463
14: -79.0532837, -46.7443428, -79.0532837, -46.7443428, -25.1323547, 25.1999283
15: -11.0553875, 7.7995930, -11.0553875, 7.7995930, -13.0417747, 13.0701180
16: -49.1398125, -29.5181141, -49.1398125, -29.5181141, -12.7222595, 12.7337265
17: -79.6467896, -45.0305099, -79.6467896, -45.0305099, -21.9874763, 22.0649872
18: -11.5796604, 9.5358448, -11.5796604, 9.5358448, -19.2092056, 19.2166290
19: -27.5895557, -12.5076895, -27.5895557, -12.5076895, -10.5226612, 10.5406685
20: -17.4594021, -4.3159504, -17.4594021, -4.3159504, -12.4352493, 12.4318275
21: -39.7976608, -20.1243858, -39.7976608, -20.1243858, -14.0101833, 14.0317917
22: -17.6748886, 0.3123665, -17.6748886, 0.3123665, -12.6982193, 12.7262344
23: -22.2937069, -7.3993239, -22.2937069, -7.3993239, -11.1021404, 11.1125393
24: -8.7166920, 4.7535696, -8.7166920, 4.7535696, -9.3013897, 9.3102169
25: -5.7729368, 8.8897533, -5.7729368, 8.8897533, -11.9877396, 11.9890823
26: -19.1811810, 2.4279919, -19.1811810, 2.4279919, -18.7489014, 18.7626953
27: -23.7054405, -7.1731005, -23.7054405, -7.1731005, -11.8626900, 11.8769989
28: -18.3387642, -0.1329041, -18.3387642, -0.1329041, -12.4298439, 12.4360123
29: -36.4321899, -18.7416058, -36.4321899, -18.7416058, -11.4934177, 11.5441284
30: -27.8891983, -7.8031111, -27.8891983, -7.8031111, -12.4891510, 12.4982452
31: -23.1913071, -7.2152233, -23.1913071, -7.2152233, -12.1550198, 12.1702461
32: -4.6359253, 10.3701916, -4.6359253, 10.3701916, -11.9580574, 11.9428520
33: 17.4503784, 40.2536736, 17.4503784, 40.2536736, -17.6935921, 17.6607895
34: 6.0333824, 25.5071201, 6.0333824, 25.5071201, -13.4857807, 13.4574261
35: 10.2628212, 29.9235191, 10.2628212, 29.9235191, -15.5780449, 15.5508461
36: 5.8066578, 23.6290283, 5.8066578, 23.6290283, -12.5164795, 12.4874001
37: 13.6705208, 32.5389824, 13.6705208, 32.5389824, -13.5903015, 13.5617905
38: 4.0681987, 25.2226028, 4.0681987, 25.2226028, -17.1295433, 17.0994530
39: 8.9867115, 30.9783421, 8.9867115, 30.9783421, -14.9330101, 14.9010735
40: 6.9200339, 25.1681404, 6.9200339, 25.1681404, -12.6546898, 12.6205387
41: -5.1083040, 10.3620338, -5.1083040, 10.3620338, -11.3837662, 11.3662758
42: -16.9170742, 0.0343406, -16.9170742, 0.0343406, -11.3550167, 11.3416424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=103, inp2_unstable=103, delta_unstable=2033
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=140, inp2_unstable=140, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=11, inp2_unstable=11, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=13, inp2_unstable=13, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1475
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1493
type: RSZ, layer: 1, pos: 620
type: RSZ, layer: 1, pos: 1405
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1531
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 1491
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1021
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1487
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1005
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 667
type: RSZ, layer: 1, pos: 1422
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1459
type: RSZ, layer: 1, pos: 1438
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1471
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1584

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1729

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.5150636, upper bound: 12.3247435
time: 17.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 33, lower bound: -12.4924132, upper bound: 12.3475449
time: 6.86 seconds

## Summary of splitting (split count: 15)
- Time for RS candidates: 26.52 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 16, time: 26.52
Output dim: 33, lower bound: -12.5063127, upper bound: 12.2936991
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 16, time: 26.52
Output dim: 33, lower bound: -12.4861416, upper bound: 12.3142773
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 16, time: 26.52
Output dim: 33, lower bound: -12.5150636, upper bound: 12.3247435
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 16, time: 26.52
Output dim: 33, lower bound: -12.4924132, upper bound: 12.3475449

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 20.34 + 1208.89 = 1229.23 seconds
