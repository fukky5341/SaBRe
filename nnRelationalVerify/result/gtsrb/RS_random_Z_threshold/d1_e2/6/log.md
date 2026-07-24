## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 6)
Time budget: 1800 seconds
Split limit: 100
Threshold: 3.6317502651


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3963165, 10.3963165)
1: (-21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2714996, 5.2714996)
2: (-12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2595863, 4.2595863)
3: (-12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3653259, 5.3653259)
4: (-10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0541573, 6.0541553)
5: (-13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1363792, 6.1363792)
6: (-8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4558334, 6.4558334)
7: (-32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8235893, 5.8235912)
8: (-18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2388401, 5.2388382)
9: (-5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0334988, 4.0334969)
10: (-36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2510471, 5.2510452)
11: (-55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9573784, 4.9573784)
12: (-11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2245445, 6.2245445)
13: (0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2925873, 5.2925873)
14: (-71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2461662, 8.2461662)
15: (-8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8938484, 4.8938484)
16: (-33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4447403, 6.4447403)
17: (-88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1758957, 8.1758957)
18: (-4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3929405, 3.3929405)
19: (-30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6509247, 4.6509247)
20: (-11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9269485, 4.9269485)
21: (-43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2718716, 4.2718697)
22: (-27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3330097, 4.3330116)
23: (-20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7792225, 4.7792225)
24: (-16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1681061, 7.1681061)
25: (-14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1913109, 4.1913090)
26: (-14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5346375, 6.5346375)
27: (-14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0586052, 4.0586052)
28: (-10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1485443, 6.1485443)
29: (-45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0057487, 5.0057487)
30: (-32.1891670, -23.0364208, -32.1891670, -23.0364208, -5.0089245, 5.0089226)
31: (-32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.3044777, 6.3044739)
32: (7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1480904, 4.1480904)
33: (4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6606770, 6.6606789)
34: (20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7274628, 5.7274628)
35: (16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4273701, 5.4273720)
36: (28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4247122, 3.4247131)
37: (11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9340363, 5.9340363)
38: (34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0302887, 6.0302887)
39: (9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.5053062, 6.5053062)
40: (15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7831497, 5.7831516)
41: (6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0081444, 5.0081444)
42: (-12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0350800, 7.0350761)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.28 + 16.63 = 18.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 38, lower bound: -3.6426783, upper bound: 3.6426783

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 753

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 717

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6426293, upper bound: 3.6420220
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6420220, upper bound: 3.6426293
time: 4.42 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.71 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.71
Output dim: 38, lower bound: -3.6426293, upper bound: 3.6420220
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.71
Output dim: 38, lower bound: -3.6420220, upper bound: 3.6426293

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3889542, 10.3858871
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2587872, 5.2548180
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2527580, 4.2500076
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3561249, 5.3529549
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0395432, 6.0337849
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1260109, 6.1221581
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4561195, 6.4563217
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8138428, 5.8102398
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2290173, 5.2254162
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0297070, 4.0289345
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2436104, 5.2420788
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9476948, 4.9501915
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2236061, 6.2242432
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2849579, 5.2823715
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2430687, 8.2427444
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8807583, 4.8752766
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4448814, 6.4448586
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1682549, 8.1673851
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3909950, 3.3912258
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6394882, 4.6422863
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9151764, 4.9183846
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2601185, 4.2630501
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3342533, 4.3348427
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7605820, 4.7652702
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1545334, 7.1578026
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1773663, 4.1808147
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5251465, 6.5273209
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0542603, 4.0552616
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1278114, 6.1329994
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0064087, 5.0069180
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9884834, 4.9934235
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2950401, 6.2973480
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1453781, 4.1450901
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6468163, 6.6497307
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7189941, 5.7204933
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4139481, 5.4161816
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4216862, 3.4218941
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9222984, 5.9255066
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0302696, 6.0300560
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4991798, 6.4996910
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7832527, 5.7840919
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0080185, 5.0080528
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0339699, 7.0350952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1755

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1416

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6391138, upper bound: 3.6417810
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6423883, upper bound: 3.6385074
time: 6.21 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3858871, 10.3889542
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2548161, 5.2587872
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2500076, 4.2527599
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3529549, 5.3561249
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0337830, 6.0395432
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1221581, 6.1260109
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4563217, 6.4561195
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8102417, 5.8138447
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2254162, 5.2290134
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0289326, 4.0297050
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2420769, 5.2436085
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9501915, 4.9476948
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2242432, 6.2236061
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2823715, 5.2849579
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2427444, 8.2430687
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8752766, 4.8807583
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4448586, 6.4448814
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1673851, 8.1682549
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3912258, 3.3909950
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6422882, 4.6394901
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9183846, 4.9151764
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2630501, 4.2601185
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3348427, 4.3342533
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7652702, 4.7605820
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1577988, 7.1545334
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1808147, 4.1773663
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5273209, 6.5251465
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0552616, 4.0542603
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1329994, 6.1278114
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0069199, 5.0064106
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9934235, 4.9884834
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2973480, 6.2950401
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1450920, 4.1453762
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6497307, 6.6468163
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7204933, 5.7189941
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4161835, 5.4139481
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4218941, 3.4216862
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9255066, 5.9222984
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0300560, 6.0302696
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4996910, 6.4991798
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7840919, 5.7832546
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0080528, 5.0080185
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0350990, 7.0339661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 759

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1458

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6406076, upper bound: 3.6414353
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6408208, upper bound: 3.6412211
time: 5.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.27
Output dim: 38, lower bound: -3.6391138, upper bound: 3.6417810
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.27
Output dim: 38, lower bound: -3.6423883, upper bound: 3.6385074
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.27
Output dim: 38, lower bound: -3.6406076, upper bound: 3.6414353
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.27
Output dim: 38, lower bound: -3.6408208, upper bound: 3.6412211

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3889465, 10.3859634
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2583351, 5.2543182
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2519035, 4.2493248
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3545723, 5.3517761
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0380249, 6.0326958
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1251183, 6.1214256
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4547539, 6.4547043
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8156242, 5.8115635
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2289906, 5.2254353
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0302010, 4.0293121
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2418556, 5.2400017
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9341030, 4.9345913
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2242432, 6.2249374
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2821999, 5.2800789
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2435913, 8.2431564
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8744278, 4.8699074
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4420319, 6.4412842
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1673317, 8.1660767
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3907852, 3.3909950
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6413307, 4.6434669
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9177208, 4.9201050
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2615261, 4.2633038
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3338032, 4.3348598
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7617016, 4.7657852
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1533737, 7.1562920
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1776409, 4.1809082
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5238342, 6.5256920
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0507717, 4.0514278
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1291962, 6.1339340
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0063972, 5.0068989
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9838219, 4.9874649
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2956314, 6.2974586
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1454544, 4.1452370
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6483078, 6.6517639
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7206745, 5.7227783
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4132233, 5.4159718
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4215012, 3.4219980
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9215431, 5.9246407
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0302887, 6.0304718
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4997330, 6.5005035
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7825317, 5.7835922
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0071640, 5.0072212
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0327644, 7.0334358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 535

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1614

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6387045, upper bound: 3.6363272
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6336591, upper bound: 3.6413732
time: 9.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3890228, 10.3858719
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2582893, 5.2543659
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2520790, 4.2491512
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3549423, 5.3514061
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0384521, 6.0322666
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1252785, 6.1212692
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4544983, 6.4549599
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8151665, 5.8120213
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2290325, 5.2253933
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0300827, 4.0294285
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2415314, 5.2403259
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9320946, 4.9365997
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2242966, 6.2248802
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2826614, 5.2796135
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2434807, 8.2432671
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8753891, 4.8689461
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4413109, 6.4420090
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1669426, 8.1664658
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3907642, 3.3910179
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6406708, 4.6441288
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9168968, 4.9209290
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2603741, 4.2644577
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3342686, 4.3343945
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7610989, 4.7663898
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1530228, 7.1566467
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1774578, 4.1810894
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5235176, 6.5260086
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0504265, 4.0517731
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1287460, 6.1343842
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0063896, 5.0069027
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9825249, 4.9887619
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2951508, 6.2979393
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1455231, 4.1451683
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6488495, 6.6512222
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7212811, 5.7221737
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4137383, 5.4154568
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4217911, 3.4217081
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9214325, 5.9247513
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0306854, 6.0300751
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4999924, 6.5002441
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7827530, 5.7833691
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0071907, 5.0071983
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0323067, 7.0338936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 692

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1586

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6423622, upper bound: 3.6320695
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6359496, upper bound: 3.6384814
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3854523, 10.3894234
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2543163, 5.2591057
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2495995, 4.2527161
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3526039, 5.3561172
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0331116, 6.0395889
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1216278, 6.1260757
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4563179, 6.4561081
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8097839, 5.8141537
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2248974, 5.2290401
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0286865, 4.0298233
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2418442, 5.2436790
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9502277, 4.9472675
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2241898, 6.2235069
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2818871, 5.2849159
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2425194, 8.2429771
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8746529, 4.8808231
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4447937, 6.4446793
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1673050, 8.1681061
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3914948, 3.3908520
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6423187, 4.6390858
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9185829, 4.9149590
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2631168, 4.2595501
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3349609, 4.3341560
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7652969, 4.7598953
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1578140, 7.1538734
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1808720, 4.1767807
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5278511, 6.5247650
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0552120, 4.0539932
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1334572, 6.1273842
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0069046, 5.0062885
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9934978, 4.9876709
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2973709, 6.2946434
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1449814, 4.1455650
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6498184, 6.6461563
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7204151, 5.7187214
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4162827, 5.4134579
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4218273, 3.4215956
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9256248, 5.9218597
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0300064, 6.0303192
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4996529, 6.4990959
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7840366, 5.7831879
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0079956, 5.0078278
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0350838, 7.0338593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1572

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1613

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6402754, upper bound: 3.6377579
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6369294, upper bound: 3.6411026
time: 5.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3858871, 10.3885155
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2548161, 5.2582855
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2500076, 4.2523499
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3529549, 5.3557739
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0337830, 6.0388699
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1221581, 6.1254768
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4563217, 6.4561157
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8102417, 5.8133869
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2254162, 5.2284946
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0289326, 4.0294590
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2420769, 5.2433758
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9497643, 4.9476948
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2242432, 6.2235489
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2823715, 5.2844696
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2427444, 8.2428436
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8752766, 4.8801346
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4448586, 6.4448204
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1673851, 8.1681747
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3910828, 3.3909950
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6418839, 4.6394901
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9181633, 4.9151764
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2624817, 4.2601185
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3347473, 4.3342533
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7645836, 4.7605820
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1571426, 7.1545334
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1802311, 4.1773663
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5269394, 6.5251465
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0549965, 4.0542603
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1325722, 6.1278114
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0067940, 5.0064106
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9926128, 4.9884834
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2969513, 6.2950401
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1450920, 4.1452713
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6490707, 6.6468163
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7202206, 5.7189941
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4156914, 5.4139481
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4218044, 3.4216862
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9250679, 5.9222984
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0300560, 6.0302200
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4996071, 6.4991798
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7840252, 5.7832546
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0080528, 5.0079613
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0349922, 7.0339661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6400968, upper bound: 3.6297629
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6293632, upper bound: 3.6404969
time: 5.24 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 12.10 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.10
Output dim: 38, lower bound: -3.6387045, upper bound: 3.6363272
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.10
Output dim: 38, lower bound: -3.6336591, upper bound: 3.6413732
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.10
Output dim: 38, lower bound: -3.6423622, upper bound: 3.6320695
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.10
Output dim: 38, lower bound: -3.6359496, upper bound: 3.6384814
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.10
Output dim: 38, lower bound: -3.6402754, upper bound: 3.6377579
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.10
Output dim: 38, lower bound: -3.6369294, upper bound: 3.6411026
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.10
Output dim: 38, lower bound: -3.6400968, upper bound: 3.6297629
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.10
Output dim: 38, lower bound: -3.6293632, upper bound: 3.6404969

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3876190, 10.3855667
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2564659, 5.2534313
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2512627, 4.2489471
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3542366, 5.3516312
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0369453, 6.0319271
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1249809, 6.1213913
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4540215, 6.4527435
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8147278, 5.8111458
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2281818, 5.2249794
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0295906, 4.0292110
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2412109, 5.2396164
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9337025, 4.9344082
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2240639, 6.2248001
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2821121, 5.2800140
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2406998, 8.2428856
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8724155, 4.8688641
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4419785, 6.4413681
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1654587, 8.1653976
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3898277, 3.3908386
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6413269, 4.6431942
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9172554, 4.9193840
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2615128, 4.2632999
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3332767, 4.3347702
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7616386, 4.7657185
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1532669, 7.1562157
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1774044, 4.1805153
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5231895, 6.5251961
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0502815, 4.0513992
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1290817, 6.1337395
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0054779, 5.0067616
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9837723, 4.9874420
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2953644, 6.2964249
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1451321, 4.1435986
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6475391, 6.6500015
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7201729, 5.7221222
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4129295, 5.4153404
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4212914, 3.4211502
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9211769, 5.9237709
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0302505, 6.0301781
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4996071, 6.4985504
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7820072, 5.7817192
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0069656, 5.0060539
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0324593, 7.0322456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 680

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 590

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6337671, upper bound: 3.6353382
time: 6.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6350390, upper bound: 3.6343280
time: 4.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3885498, 10.3846436
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2574463, 5.2524490
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2515221, 4.2486858
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3544312, 5.3514366
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0372581, 6.0316143
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1250877, 6.1212883
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4527931, 6.4539719
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8152046, 5.8106689
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2285366, 5.2246246
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0301018, 4.0286999
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2414703, 5.2393589
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9339218, 4.9341888
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2241020, 6.2247581
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2821388, 5.2799911
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2433205, 8.2402687
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8733845, 4.8678951
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4421158, 6.4412270
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1666527, 8.1641998
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3906307, 3.3900356
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6410599, 4.6434631
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9169998, 4.9196358
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2615223, 4.2632904
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3337154, 4.3343334
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7616348, 4.7657223
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1533051, 7.1561813
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1772480, 4.1806717
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5233383, 6.5250473
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0507412, 4.0509396
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1290016, 6.1338196
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0062599, 5.0059834
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9837990, 4.9874134
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2945938, 6.2971954
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1438160, 4.1449146
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6465473, 6.6509933
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7200165, 5.7222767
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4125900, 5.4156799
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4206524, 3.4217892
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9206696, 5.9242744
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0299950, 6.0304337
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4977798, 6.5003815
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7806568, 5.7830715
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0059967, 5.0070229
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0315742, 7.0331306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 751

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1391

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6336025, upper bound: 3.6404119
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6326973, upper bound: 3.6413159
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3922043, 10.3909760
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2550526, 5.2540092
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2502556, 4.2476845
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3532944, 5.3498688
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0315819, 6.0263081
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1214142, 6.1178589
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4487839, 6.4473534
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8086090, 5.8066483
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2273979, 5.2238789
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0248604, 4.0262966
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2499352, 5.2514019
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9345360, 4.9397888
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2278519, 6.2293396
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2791710, 5.2765999
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2227440, 8.2272453
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8615017, 4.8580608
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4477615, 6.4510612
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1530037, 8.1542015
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3859577, 3.3871994
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6421928, 4.6450424
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9266357, 4.9285374
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2604084, 4.2645226
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3314514, 4.3319378
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7600441, 4.7648659
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1520615, 7.1553116
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1790657, 4.1815681
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5256119, 6.5278358
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0485382, 4.0505047
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1367188, 6.1409378
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9998970, 5.0019283
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9835129, 4.9896698
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2918320, 6.2938118
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1411819, 4.1393318
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6387043, 6.6376038
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7160378, 5.7153111
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4079628, 5.4078350
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4175673, 3.4161386
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9167900, 5.9191513
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0250893, 6.0225525
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4911957, 6.4888535
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7731133, 5.7716446
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0044289, 5.0035629
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0263138, 7.0259972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 731

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1613

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6420243, upper bound: 3.6283929
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6386856, upper bound: 3.6317315
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3941269, 10.3890533
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2579327, 5.2511292
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2506104, 4.2473297
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3534088, 5.3497543
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0324936, 6.0253963
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1218681, 6.1174088
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4468956, 6.4492455
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8097954, 5.8054619
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2275200, 5.2237568
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0269527, 4.0242062
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2526093, 5.2487297
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9352818, 4.9390430
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2287560, 6.2284355
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2796478, 5.2761230
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2274551, 8.2225304
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8645039, 4.8550587
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4503632, 6.4484596
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1546822, 8.1525230
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3869457, 3.3862114
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6415825, 4.6456490
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9244995, 4.9306698
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2604389, 4.2644920
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3318138, 4.3315754
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7595749, 4.7653351
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1516876, 7.1556854
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1779366, 4.1826992
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5253448, 6.5281029
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0491581, 4.0498848
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1352997, 6.1423569
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0014153, 5.0004082
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9834328, 4.9897480
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2910233, 6.2946205
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1396866, 4.1408253
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6352329, 6.6410751
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7144165, 5.7169285
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4061165, 5.4096832
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4162207, 3.4174833
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9158363, 5.9201050
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0231667, 6.0244789
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4886017, 6.4914474
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7710304, 5.7737312
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0035515, 5.0044403
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0244141, 7.0279007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 685

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 726

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6357422, upper bound: 3.6379081
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6353777, upper bound: 3.6382753
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3851242, 10.3890305
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2538052, 5.2585888
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2496109, 4.2526608
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3525696, 5.3560028
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0334244, 6.0395546
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1213837, 6.1252365
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4552383, 6.4547005
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8097458, 5.8140965
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2246304, 5.2286816
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0286522, 4.0297546
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2416496, 5.2433605
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9494038, 4.9471912
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2240334, 6.2234039
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2819939, 5.2842216
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2401428, 8.2432365
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8741817, 4.8803864
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4440918, 6.4437370
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1663780, 8.1677017
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3905220, 3.3908482
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6421394, 4.6389866
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9184074, 4.9150505
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2627354, 4.2596874
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3346195, 4.3340836
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7649117, 4.7600956
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1568985, 7.1538429
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1805954, 4.1767197
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5261230, 6.5244293
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0544605, 4.0540009
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1325111, 6.1271019
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0062866, 5.0063457
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9921703, 4.9876537
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2966423, 6.2940636
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1441650, 4.1441269
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6493759, 6.6458549
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7201710, 5.7187290
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4160538, 5.4133835
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4216824, 3.4209900
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9254379, 5.9217262
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0299950, 6.0302429
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4995384, 6.4976959
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7832127, 5.7822647
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0075073, 5.0066948
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0347214, 7.0332947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1442

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6393779, upper bound: 3.6300714
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6325873, upper bound: 3.6368603
time: 6.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3854523, 10.3890915
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2543163, 5.2585945
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2495422, 4.2527161
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3524895, 5.3561172
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0330734, 6.0395889
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1207886, 6.1260757
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4549103, 6.4561081
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8097267, 5.8141537
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2248974, 5.2287769
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0286179, 4.0298233
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2415276, 5.2436790
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9502277, 4.9464436
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2240829, 6.2235069
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2811890, 5.2849159
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2425194, 8.2405968
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8746529, 4.8803520
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4438515, 6.4446793
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1673050, 8.1671791
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3914948, 3.3898773
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6422195, 4.6390858
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9185829, 4.9147835
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2631168, 4.2591667
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3349609, 4.3338146
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7652969, 4.7595119
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1578140, 7.1529655
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1808720, 4.1765022
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5278511, 6.5230370
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0552120, 4.0532398
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1334572, 6.1264420
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0069046, 5.0056705
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9934978, 4.9863415
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2967911, 6.2946434
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1435471, 4.1455650
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6495132, 6.6461563
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7204151, 5.7184753
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4162827, 5.4132290
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4212208, 3.4215956
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9254913, 5.9218597
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0300064, 6.0303078
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4982491, 6.4990959
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7831173, 5.7831879
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0068588, 5.0078278
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0345230, 7.0338593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 784

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6340786, upper bound: 3.6410388
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6368660, upper bound: 3.6382502
time: 4.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3853149, 10.3889885
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2502346, 5.2559986
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2495689, 4.2518921
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3522377, 5.3553162
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0316353, 6.0368843
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1221008, 6.1254349
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4539452, 6.4535370
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8064041, 5.8114319
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2237835, 5.2272205
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0280666, 4.0294132
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2366486, 5.2417774
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9445801, 4.9449234
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2233276, 6.2240753
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2823067, 5.2843933
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2318840, 8.2397881
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8702469, 4.8775272
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4444923, 6.4469948
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1564980, 8.1640701
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3895264, 3.3904152
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6416779, 4.6392632
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9177513, 4.9140930
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2603130, 4.2588615
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3312435, 4.3328362
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7644787, 4.7604637
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1567612, 7.1541977
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1798878, 4.1769581
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5261230, 6.5244370
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0532990, 4.0537109
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1324005, 6.1275330
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0012589, 5.0047073
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9892826, 4.9867535
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2962837, 6.2943611
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1424770, 4.1420002
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6466293, 6.6431007
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7187958, 5.7175293
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4139595, 5.4112778
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4199047, 3.4167404
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9239998, 5.9204521
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0279388, 6.0240936
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4973679, 6.4937019
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7819614, 5.7795277
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0066566, 5.0052071
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0337296, 7.0317764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1613

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6399198, upper bound: 3.6282654
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6363164, upper bound: 3.6294407
time: 6.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3863449, 10.3879585
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2525272, 5.2537060
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2495461, 4.2519093
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3524971, 5.3550529
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0317993, 6.0367203
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1221237, 6.1254120
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4537430, 6.4537392
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8082848, 5.8095512
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2241421, 5.2268620
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0288887, 4.0285892
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2404785, 5.2379417
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9469910, 4.9425106
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2247696, 6.2226295
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2822952, 5.2844048
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2396889, 8.2319832
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8726730, 4.8751030
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4470406, 6.4444466
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1632767, 8.1572914
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3905029, 3.3894386
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6416550, 4.6392822
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9170837, 4.9147568
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2612228, 4.2579498
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3333302, 4.3307495
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7644672, 4.7604771
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1568069, 7.1541519
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1798210, 4.1770248
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5262299, 6.5243301
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0544472, 4.0525627
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1322823, 6.1276474
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0050926, 5.0008774
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9908810, 4.9851551
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2962685, 6.2943764
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1418209, 4.1426544
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6453552, 6.6443748
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7187500, 5.7175751
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4130173, 5.4122200
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4168568, 3.4197884
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9232216, 5.9212303
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0239296, 6.0280991
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4941292, 6.4969444
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7802982, 5.7811871
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0052986, 5.0065651
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0328064, 7.0327034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 643

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6284182, upper bound: 3.6389699
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6278359, upper bound: 3.6395531
time: 5.12 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 12.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6337671, upper bound: 3.6353382
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6350390, upper bound: 3.6343280
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6336025, upper bound: 3.6404119
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6326973, upper bound: 3.6413159
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6420243, upper bound: 3.6283929
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6386856, upper bound: 3.6317315
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6357422, upper bound: 3.6379081
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6353777, upper bound: 3.6382753
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6393779, upper bound: 3.6300714
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6325873, upper bound: 3.6368603
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6340786, upper bound: 3.6410388
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6368660, upper bound: 3.6382502
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6399198, upper bound: 3.6282654
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6363164, upper bound: 3.6294407
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6284182, upper bound: 3.6389699
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.79
Output dim: 38, lower bound: -3.6278359, upper bound: 3.6395531

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3867798, 10.3839874
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2548523, 5.2508335
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2503948, 4.2476578
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3540459, 5.3508263
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0345840, 6.0292149
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1244011, 6.1207199
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4514275, 6.4512596
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8134651, 5.8091774
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2267570, 5.2229004
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0290508, 4.0283337
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2396946, 5.2377758
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9325600, 4.9329643
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2235718, 6.2242470
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2818794, 5.2798080
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2382050, 8.2374115
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8704414, 4.8657093
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4415817, 6.4406891
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1626854, 8.1608467
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3891563, 3.3893890
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6410294, 4.6431236
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9164085, 4.9187393
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2613754, 4.2631264
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3329353, 4.3341446
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7612724, 4.7653770
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1531982, 7.1560783
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1764336, 4.1798630
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5224953, 6.5241394
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0502548, 4.0506287
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1288300, 6.1335068
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0048542, 5.0055161
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9835587, 4.9870110
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2945290, 6.2961578
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1426792, 4.1424961
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6442051, 6.6476898
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7191963, 5.7210693
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4119530, 5.4146347
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4195576, 3.4201488
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9185410, 5.9220085
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0289612, 6.0295258
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4969711, 6.4973831
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7794895, 5.7803707
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0046730, 5.0050545
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0301628, 7.0311432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1711

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1766

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6314697, upper bound: 3.6326079
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6310474, upper bound: 3.6330264
time: 5.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3860321, 10.3849564
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2538681, 5.2520313
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2499752, 4.2481880
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3534317, 5.3514557
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0342331, 6.0297928
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1243134, 6.1208267
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4528694, 6.4501495
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8127556, 5.8100281
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2261009, 5.2237682
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0287132, 4.0287170
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2393703, 5.2382240
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9322586, 4.9333134
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2234879, 6.2243080
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2819061, 5.2797928
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2352257, 8.2407150
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8692608, 4.8670864
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4412994, 6.4410019
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1609077, 8.1627922
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3883762, 3.3901997
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6413612, 4.6428986
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9166832, 4.9185333
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2613411, 4.2631607
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3326492, 4.3344555
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7612991, 4.7653522
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1531296, 7.1561584
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1768494, 4.1795425
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5221329, 6.5245323
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0495110, 4.0513763
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1288834, 6.1334839
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0042324, 5.0061970
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9833374, 4.9872360
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2951355, 6.2955856
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1440754, 4.1411419
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6454639, 6.6466713
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7192726, 5.7211475
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4122734, 5.4143639
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4203739, 3.4194164
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9194412, 5.9211349
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0296097, 6.0288887
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4984398, 6.4959145
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7807865, 5.7791996
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0059814, 5.0037651
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0314293, 7.0299530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1594

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 685

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6344437, upper bound: 3.6337412
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6344432, upper bound: 3.6337417
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3869400, 10.3828201
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2568359, 5.2516994
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2516212, 4.2487411
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3539581, 5.3507957
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0376320, 6.0318203
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1247253, 6.1208000
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4519081, 6.4531708
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8150177, 5.8103886
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2280941, 5.2241039
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0305939, 4.0290051
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2402935, 5.2380123
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9306412, 4.9316406
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2239990, 6.2246704
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2821770, 5.2800140
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2425919, 8.2394218
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8733292, 4.8676319
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4406242, 6.4398155
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1662598, 8.1637573
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3906021, 3.3900089
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6394100, 4.6421928
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9171371, 4.9198666
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2609406, 4.2632809
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3337631, 4.3343849
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7597275, 4.7640991
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1527939, 7.1557961
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1766109, 4.1801567
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5233650, 6.5251198
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0507717, 4.0509739
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1288033, 6.1338234
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0054150, 5.0053177
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9827328, 4.9867020
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2939262, 6.2967033
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1427174, 4.1438847
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6452961, 6.6499176
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7188568, 5.7212715
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4126759, 5.4158382
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4199333, 3.4211216
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9204292, 5.9241562
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0293732, 6.0297165
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4977798, 6.5004501
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7794228, 5.7819672
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0060043, 5.0070267
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0315895, 7.0331573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1586

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6335766, upper bound: 3.6339719
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6271629, upper bound: 3.6403859
time: 6.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3867188, 10.3830490
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2566986, 5.2518368
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2515793, 4.2487850
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3537903, 5.3509636
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0374641, 6.0319901
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1245995, 6.1209259
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4519958, 6.4530830
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8149300, 5.8104763
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2280140, 5.2241840
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0304050, 4.0291939
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2401180, 5.2381802
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9313736, 4.9309082
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2240105, 6.2246552
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2821579, 5.2800331
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2424774, 8.2395401
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8731194, 4.8678417
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4407043, 6.4397354
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1662140, 8.1638069
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3906040, 3.3900089
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6397839, 4.6418133
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9172287, 4.9197712
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2615128, 4.2627087
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3337669, 4.3343792
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7600098, 4.7638149
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1529160, 7.1556664
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1767330, 4.1800346
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5234070, 6.5250740
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0507755, 4.0509701
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1290054, 6.1336212
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0055943, 5.0051384
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9830914, 4.9863453
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2941055, 6.2965240
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1427860, 4.1438179
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6454716, 6.6497383
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7190132, 5.7211170
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4127483, 5.4157639
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4199848, 3.4210691
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9205475, 5.9240379
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0292702, 6.0298157
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4978485, 6.5003815
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7795563, 5.7818336
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0060043, 5.0070267
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0315971, 7.0331497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1379

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6326474, upper bound: 3.6399940
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6313749, upper bound: 3.6412661
time: 4.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3918686, 10.3905792
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2545414, 5.2534924
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2502747, 4.2476311
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3532639, 5.3497581
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0318947, 6.0262680
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1211777, 6.1170235
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4477043, 6.4459457
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8085747, 5.8065948
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2271309, 5.2235203
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0248280, 4.0262299
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2497425, 5.2510815
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9337101, 4.9397125
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2276917, 6.2292328
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2792816, 5.2759056
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2203636, 8.2275009
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8610344, 4.8576298
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4470596, 6.4501190
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1520844, 8.1538048
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3849869, 3.3871994
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6420155, 4.6449471
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9264565, 4.9286270
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2600327, 4.2646675
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3311062, 4.3318634
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7596550, 4.7650623
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1511536, 7.1552811
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1787872, 4.1815052
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5238762, 6.5274925
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0477848, 4.0505123
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1357727, 6.1406479
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9992790, 5.0019855
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9821815, 4.9896545
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2910995, 6.2932281
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1403542, 4.1378880
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6382561, 6.6372948
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7157974, 5.7153263
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4077320, 5.4077587
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4174194, 3.4155302
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9165993, 5.9190140
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0250778, 6.0224762
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4910774, 6.4874496
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7722893, 5.7707214
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0039368, 5.0024223
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0259399, 7.0254250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 535

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1399

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6372204, upper bound: 3.6263769
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6400079, upper bound: 3.6235868
time: 4.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3922043, 10.3906403
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2550526, 5.2535000
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2502060, 4.2476845
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3531799, 5.3498688
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0315437, 6.0263081
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1205826, 6.1178589
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4473763, 6.4473534
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8085556, 5.8066483
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2273979, 5.2236156
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0247955, 4.0262966
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2496166, 5.2514019
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9345360, 4.9389648
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2277451, 6.2293396
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2784767, 5.2765999
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2227440, 8.2248611
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8615017, 4.8575954
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4468231, 6.4510612
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1530037, 8.1532822
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3859577, 3.3862286
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6420956, 4.6450424
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9266357, 4.9283562
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2604084, 4.2641468
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3314514, 4.3315926
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7600441, 4.7644787
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1520615, 7.1544037
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1790657, 4.1812878
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5256119, 6.5261002
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0485382, 4.0497513
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1367188, 6.1399879
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9998970, 5.0013103
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9835129, 4.9883423
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2912483, 6.2938118
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1397362, 4.1393318
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6383972, 6.6376038
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7160378, 5.7150726
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4079628, 5.4076042
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4169579, 3.4161386
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9166527, 5.9191513
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0250893, 6.0225410
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4897881, 6.4888535
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7721939, 5.7716446
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0032921, 5.0035629
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0257416, 7.0259972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6377906, upper bound: 3.6240299
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6309868, upper bound: 3.6308371
time: 5.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3913040, 10.3894730
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2569256, 5.2520790
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2474976, 4.2468243
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3498421, 5.3479919
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0270958, 6.0232010
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1185493, 6.1164513
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4455566, 6.4461212
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8061676, 5.8039665
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2203522, 5.2191868
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0264816, 4.0239124
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2511520, 5.2491627
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9351406, 4.9390049
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2280159, 6.2273750
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2772636, 5.2748032
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2200356, 8.2206154
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8596764, 4.8528862
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4499359, 6.4478989
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1441231, 8.1489906
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3861485, 3.3852940
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6381721, 4.6443748
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9245148, 4.9306011
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2585468, 4.2641487
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3289490, 4.3313408
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7590904, 4.7649136
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1516266, 7.1556473
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1771908, 4.1821690
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5244560, 6.5275574
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0482674, 4.0486145
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1351242, 6.1418343
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0001183, 5.0017071
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9817390, 4.9867382
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2886696, 6.2947350
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1390572, 4.1387234
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6336670, 6.6384583
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7110825, 5.7095146
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4041901, 5.4054890
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4139662, 3.4124689
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9133606, 5.9148483
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0230560, 6.0231590
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4885483, 6.4913597
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7716064, 5.7705269
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0015945, 5.0004425
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0232201, 7.0265388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1740

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6312873, upper bound: 3.6366428
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6345639, upper bound: 3.6334595
time: 6.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3945541, 10.3862228
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2588825, 5.2501221
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2501068, 4.2442169
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3516502, 5.3461876
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0303001, 6.0199966
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1209106, 6.1140900
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4437714, 6.4479065
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8082962, 5.8018379
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2229462, 5.2165909
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0266590, 4.0237350
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2530441, 5.2472725
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9352455, 4.9388981
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2276917, 6.2276993
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2783241, 5.2737389
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2255402, 8.2151108
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8623333, 4.8502293
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4497986, 6.4480362
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1511459, 8.1419678
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3860283, 3.3854141
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6403084, 4.6422367
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9244308, 4.9306850
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2600956, 4.2625999
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3315773, 4.3287125
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7591515, 4.7648506
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1516571, 7.1556282
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1774063, 4.1819534
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5247955, 6.5272141
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0478878, 4.0489941
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1347809, 6.1421776
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0027161, 4.9991074
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9804230, 4.9880505
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2911415, 6.2922630
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1375847, 4.1401958
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6326141, 6.6395111
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7070007, 5.7135906
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4019241, 5.4077549
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4112082, 3.4152288
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9105797, 5.9176292
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0218430, 6.0243721
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4885101, 6.4913979
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7678223, 5.7743073
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9995537, 5.0024834
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0230522, 7.0267105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 643

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6342830, upper bound: 3.6382636
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6353660, upper bound: 3.6371814
time: 5.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4198074, 10.4142418
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2718067, 5.2709522
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2593651, 4.2580528
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3443832, 5.3451691
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0089569, 6.0071850
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1168213, 6.1146011
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4469147, 6.4483299
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8064041, 5.8064041
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2014008, 5.1979465
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0159988, 4.0130348
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2373524, 5.2344093
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9262962, 4.9292641
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2177315, 6.2189255
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2693367, 5.2674751
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2458382, 8.2455711
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8525677, 4.8518124
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4453201, 6.4450111
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1608696, 8.1595268
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3773422, 3.3807182
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6394005, 4.6363716
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9222221, 4.9218807
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2500877, 4.2498035
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3373432, 4.3358612
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7476597, 4.7470436
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1420059, 7.1425629
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1737423, 4.1711559
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5375099, 6.5404053
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0456753, 4.0472164
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1329384, 6.1334229
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0087318, 5.0084381
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9548626, 4.9591770
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2959747, 6.2933769
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1417866, 4.1422539
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6314392, 6.6328354
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6896820, 5.6956081
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3880806, 5.3921871
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4163828, 3.4181957
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9119377, 5.9115181
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0267143, 6.0261612
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4994392, 6.4972458
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7883987, 5.7907143
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0030365, 5.0032578
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0337448, 7.0324707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 590

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 756

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6388947, upper bound: 3.6300657
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6393721, upper bound: 3.6295865
time: 4.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4103317, 10.4237099
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2661686, 5.2765923
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2550049, 4.2624130
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3417358, 5.3478127
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0010567, 6.0150852
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1107483, 6.1206741
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4488678, 6.4463730
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8020554, 5.8107529
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.1939011, 5.2054482
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0119324, 4.0171013
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2327023, 5.2390594
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9314766, 4.9240837
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2195549, 6.2171021
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2652512, 5.2715607
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2424774, 8.2489357
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8456078, 4.8587723
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4453659, 6.4449692
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1582031, 8.1621895
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3803902, 3.3776684
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6395264, 4.6362476
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9252357, 4.9188709
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2528515, 4.2470398
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3363953, 4.3368073
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7518635, 4.7428417
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1456223, 7.1389542
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1750317, 4.1698666
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5420952, 6.5358200
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0476742, 4.0452175
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1388321, 6.1275253
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0083809, 5.0087910
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9636898, 4.9503479
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2959557, 6.2933960
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1422939, 4.1417465
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6363564, 6.6279182
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6970520, 5.6882439
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3948555, 5.3854103
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4188871, 3.4156914
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9152298, 5.9082222
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0259094, 6.0269623
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4990883, 6.4975967
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7916603, 5.7874508
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0040703, 5.0022202
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0338974, 7.0323181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1570

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 685

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6320037, upper bound: 3.6362676
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6320031, upper bound: 3.6362683
time: 5.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3818054, 10.3861618
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2526093, 5.2569485
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2476883, 4.2512302
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3527222, 5.3563995
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0331116, 6.0396290
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1207542, 6.1260414
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4566956, 6.4573822
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8095055, 5.8138638
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2249451, 5.2288322
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0287018, 4.0298805
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2416115, 5.2437267
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9502449, 4.9464207
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2242889, 6.2236633
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2816277, 5.2854805
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2394485, 8.2379456
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8747654, 4.8807716
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4445724, 6.4451790
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1693115, 8.1700096
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3903599, 3.3888111
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6422806, 4.6391411
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9179268, 4.9139442
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2629204, 4.2587833
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3336468, 4.3327255
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7635841, 4.7581882
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1568146, 7.1521950
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1807919, 4.1764221
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5287323, 6.5241585
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0552654, 4.0532799
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1342583, 6.1274223
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0048141, 5.0035133
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9925537, 4.9850655
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2974472, 6.2951965
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1444988, 4.1462612
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6503410, 6.6470604
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7212524, 5.7192211
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4168282, 5.4138355
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4214191, 3.4218960
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9270973, 5.9231377
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0292778, 6.0297546
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.5000076, 6.5006790
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7802143, 5.7795868
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0074196, 5.0082970
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0345879, 7.0339088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1569

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6229513, upper bound: 3.6291735
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6229171, upper bound: 3.6292077
time: 7.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3825226, 10.3854370
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2526703, 5.2568874
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2480583, 4.2508602
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3527718, 5.3563499
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0331192, 6.0396214
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1207619, 6.1260376
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4561882, 6.4578896
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8094406, 5.8139286
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2249527, 5.2288227
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0286808, 4.0299015
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2415771, 5.2437592
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9502048, 4.9464607
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2242432, 6.2237091
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2817574, 5.2853508
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2398720, 8.2375259
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8750725, 4.8804626
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4443550, 6.4453964
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1701355, 8.1691856
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3904324, 3.3887386
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6422729, 4.6391506
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9177399, 4.9141312
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2627316, 4.2589703
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3338699, 4.3325005
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7639732, 4.7577991
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1570282, 7.1519775
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1807938, 4.1764202
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5289764, 6.5239143
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0552521, 4.0532932
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1344337, 6.1272430
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0047417, 5.0035839
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9922218, 4.9853973
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2973442, 6.2952995
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1442432, 4.1465168
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6504135, 6.6469879
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7211533, 5.7193203
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4168892, 5.4137764
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4215221, 3.4217930
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9267731, 5.9234657
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0294533, 6.0295792
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4998283, 6.5008545
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7795200, 5.7802811
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0073204, 5.0083961
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0345726, 7.0339317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1570

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6367387, upper bound: 3.6330223
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6316415, upper bound: 3.6381227
time: 6.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3849869, 10.3886604
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2497616, 5.2555428
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2495804, 4.2518349
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3521996, 5.3551979
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0319519, 6.0368500
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1218567, 6.1245995
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4529533, 6.4521332
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8063698, 5.8113747
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2235546, 5.2269306
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0280323, 4.0293446
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2364540, 5.2414589
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9437904, 4.9449711
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2231712, 6.2239723
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2824173, 5.2836990
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2294998, 8.2400475
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8697758, 4.8771343
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4437866, 6.4460526
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1556091, 8.1636925
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3885784, 3.3904381
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6414986, 4.6391621
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9175720, 4.9141884
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2599831, 4.2591343
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3308983, 4.3327599
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7640934, 4.7606640
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1558533, 7.1541710
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1796093, 4.1768990
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5243835, 6.5241508
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0525665, 4.0537605
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1314545, 6.1272507
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0006409, 5.0047626
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9880123, 4.9868698
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2955742, 6.2937775
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1418209, 4.1405602
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6462383, 6.6427956
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7185478, 5.7175369
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4137306, 5.4112015
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4197769, 3.4161549
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9238358, 5.9203072
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0279198, 6.0240173
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4974747, 6.4923592
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7811489, 5.7786083
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0063362, 5.0040588
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0334320, 7.0312157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 571

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1442

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6395254, upper bound: 3.6280305
time: 5.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6397013, upper bound: 3.6279323
time: 4.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3853149, 10.3886528
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2502346, 5.2555237
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2495117, 4.2518921
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3521194, 5.3553162
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0316010, 6.0368843
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1212616, 6.1254349
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4525414, 6.4535370
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8063469, 5.8114319
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2237835, 5.2269878
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0279980, 4.0294132
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2363243, 5.2417774
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9445801, 4.9441338
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2232246, 6.2240753
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2816124, 5.2843933
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2318840, 8.2374077
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8702469, 4.8770599
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4435501, 6.4469948
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1564980, 8.1631699
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3895264, 3.3894653
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6415749, 4.6392632
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9177513, 4.9139175
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2603130, 4.2585316
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3312435, 4.3324909
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7644787, 4.7600803
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1567612, 7.1532898
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1798878, 4.1766815
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5261230, 6.5226936
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0532990, 4.0529785
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1324005, 6.1265907
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0012589, 5.0040894
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9892826, 4.9854832
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2957039, 6.2943611
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1410351, 4.1420002
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6463299, 6.6431007
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7187958, 5.7172832
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4139595, 5.4110489
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4193192, 3.4167404
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9238625, 5.9204521
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0279388, 6.0240784
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4960289, 6.4937019
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7810383, 5.7795277
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0055084, 5.0052071
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0331573, 7.0317764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 535

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6352222, upper bound: 3.6294286
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6363041, upper bound: 3.6283465
time: 5.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3830338, 10.3890991
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2487545, 5.2545757
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2486267, 4.2533817
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3507004, 5.3552780
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0305481, 6.0400772
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1197281, 6.1264992
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4526100, 6.4525909
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8064423, 5.8121910
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2202816, 5.2275753
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0263672, 4.0288124
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2347717, 5.2384415
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9468651, 4.9425526
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2234383, 6.2216301
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2807426, 5.2849617
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2375221, 8.2345428
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8692455, 4.8755741
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4445953, 6.4451294
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1604118, 8.1582909
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3906975, 3.3870296
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6415577, 4.6377506
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9173126, 4.9122696
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2611618, 4.2559338
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3333569, 4.3303242
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7647953, 4.7567120
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1575623, 7.1517982
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1801071, 4.1737785
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5281258, 6.5222168
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0541553, 4.0520134
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1326675, 6.1232719
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0046692, 5.0013504
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9906502, 4.9825764
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2960472, 6.2928581
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1406307, 4.1417332
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6483955, 6.6429787
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7201519, 5.7173767
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4139767, 5.4085026
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4185953, 3.4192266
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9254684, 5.9206390
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0252037, 6.0280380
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4951324, 6.4967880
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7803574, 5.7811852
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0046349, 5.0059090
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0322723, 7.0318527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1619

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6281295, upper bound: 3.6361064
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6255581, upper bound: 3.6386778
time: 4.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3863449, 10.3846436
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2525272, 5.2499313
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2495461, 4.2509880
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3524971, 5.3532562
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0317993, 6.0354748
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1221237, 6.1230240
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4537430, 6.4526100
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8082848, 5.8077087
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2241421, 5.2230091
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0288887, 4.0260696
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2404785, 5.2322369
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9469910, 4.9423847
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2247696, 6.2212982
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2822952, 5.2828560
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2396889, 8.2298279
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8726730, 4.8716774
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4470406, 6.4420013
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1632767, 8.1544228
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3880920, 3.3894386
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6401234, 4.6392822
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9145927, 4.9147568
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2592087, 4.2579498
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3329048, 4.3307495
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7606983, 4.7604771
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1544571, 7.1541519
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1765766, 4.1770248
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5241165, 6.5243301
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0538960, 4.0525627
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1279144, 6.1276474
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0050926, 5.0004539
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9883041, 4.9851551
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2947502, 6.2943764
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1418209, 4.1414700
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6439667, 6.6443748
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7185535, 5.7175751
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4093037, 5.4122200
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4162970, 3.4197884
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9226227, 5.9212303
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0238647, 6.0280991
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4939728, 6.4969444
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7802963, 5.7811871
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0052986, 5.0059052
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0328064, 7.0321693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 756

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1441

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6271886, upper bound: 3.6387245
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6270054, upper bound: 3.6389274
time: 4.68 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 11.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6314697, upper bound: 3.6326079
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6310474, upper bound: 3.6330264
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6344437, upper bound: 3.6337412
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6344432, upper bound: 3.6337417
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6335766, upper bound: 3.6339719
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6271629, upper bound: 3.6403859
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6326474, upper bound: 3.6399940
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6313749, upper bound: 3.6412661
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6372204, upper bound: 3.6263769
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6400079, upper bound: 3.6235868
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6377906, upper bound: 3.6240299
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6309868, upper bound: 3.6308371
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6312873, upper bound: 3.6366428
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6345639, upper bound: 3.6334595
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6342830, upper bound: 3.6382636
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6353660, upper bound: 3.6371814
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6388947, upper bound: 3.6300657
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6393721, upper bound: 3.6295865
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6320037, upper bound: 3.6362676
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6320031, upper bound: 3.6362683
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6229513, upper bound: 3.6291735
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6229171, upper bound: 3.6292077
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6367387, upper bound: 3.6330223
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6316415, upper bound: 3.6381227
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6395254, upper bound: 3.6280305
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6397013, upper bound: 3.6279323
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6352222, upper bound: 3.6294286
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6363041, upper bound: 3.6283465
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6281295, upper bound: 3.6361064
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6255581, upper bound: 3.6386778
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6271886, upper bound: 3.6387245
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 11.94
Output dim: 38, lower bound: -3.6270054, upper bound: 3.6389274

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3892670, 10.3835526
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2559776, 5.2505436
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2511425, 4.2474937
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3532944, 5.3496246
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0326462, 6.0258484
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1251945, 6.1196289
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4514313, 6.4511719
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8120499, 5.8054123
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2245827, 5.2192554
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0281982, 4.0270805
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2403297, 5.2374096
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9322834, 4.9324760
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2225914, 6.2238922
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2812157, 5.2789536
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2385674, 8.2373199
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8693027, 4.8640652
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4418030, 6.4404755
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1615219, 8.1553307
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3883629, 3.3889637
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6422653, 4.6423798
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9151115, 4.9171715
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2617168, 4.2625809
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3333168, 4.3325424
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7613029, 4.7653198
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1531372, 7.1560097
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1771641, 4.1796131
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5224571, 6.5241737
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0502243, 4.0505981
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1287575, 6.1334877
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0069580, 5.0051517
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9827118, 4.9861755
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2963104, 6.2958488
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1415157, 4.1419697
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6429539, 6.6469688
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7115250, 5.7174149
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4056282, 5.4115543
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4179764, 3.4205055
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9174156, 5.9213181
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0287743, 6.0300446
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4968300, 6.4971771
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7792606, 5.7816181
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0042038, 5.0047455
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0297890, 7.0294266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 759

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6237618, upper bound: 3.6324880
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6313499, upper bound: 3.6249021
time: 5.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3863525, 10.3864746
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2545662, 5.2519588
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2502270, 4.2484074
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3528404, 5.3500748
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0312195, 6.0272770
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1233063, 6.1215134
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4513397, 6.4512634
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8097000, 5.8077583
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2231140, 5.2207222
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0277977, 4.0274830
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2393303, 5.2384109
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9320736, 4.9326878
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2232170, 6.2232666
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2810249, 5.2791443
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2381134, 8.2377701
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8687973, 4.8645706
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4413681, 6.4409103
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1571693, 8.1596794
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3887310, 3.3885937
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6402893, 4.6443577
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9148407, 4.9174461
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2608280, 4.2634697
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3313351, 4.3345242
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7612152, 4.7654095
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1531296, 7.1560173
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1761837, 4.1805935
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5225296, 6.5241051
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0502262, 4.0505981
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1288071, 6.1334381
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0044899, 5.0076180
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9827271, 4.9861603
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2942162, 6.2979431
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1421528, 4.1413345
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6434803, 6.6464386
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7155457, 5.7133980
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4088745, 5.4083080
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4199123, 3.4185677
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9178505, 5.9208832
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0294800, 6.0293388
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4967690, 6.4972382
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7807331, 5.7801437
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0043678, 5.0045815
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0284462, 7.0307655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1563

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6305444, upper bound: 3.6177592
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6157744, upper bound: 3.6325235
time: 5.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3853149, 10.3866043
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2525940, 5.2546711
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2495937, 4.2490559
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3533783, 5.3513260
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0341644, 6.0298004
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1239738, 6.1207809
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4514771, 6.4500618
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8119164, 5.8135452
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2259712, 5.2237339
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0286541, 4.0287056
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2393322, 5.2382183
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9308567, 4.9374771
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2232666, 6.2248306
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2819405, 5.2794991
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2343597, 8.2405701
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8690033, 4.8667221
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4403000, 6.4438057
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1607513, 8.1649094
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3880081, 3.3912563
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6411762, 4.6427040
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9165649, 4.9185715
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2611923, 4.2631416
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3327980, 4.3339920
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7609558, 4.7653198
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1528778, 7.1560440
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1775875, 4.1794071
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5220604, 6.5245476
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0483341, 4.0534515
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1286507, 6.1331673
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0039921, 5.0072327
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9833298, 4.9872475
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2955818, 6.2954292
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1436329, 4.1399708
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6513386, 6.6462212
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7247219, 5.7209435
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4183407, 5.4137859
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4210644, 3.4193611
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9219666, 5.9209976
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0296898, 6.0288811
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.5026360, 6.4951172
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7808609, 5.7791862
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0051117, 5.0034332
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0303612, 7.0313416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1442

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6339698, upper bound: 3.6335433
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6342458, upper bound: 3.6332674
time: 5.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3860321, 10.3842316
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2538681, 5.2507572
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2499752, 4.2478065
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3534317, 5.3514023
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0342331, 6.0297279
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1243134, 6.1204872
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4528694, 6.4487572
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8127556, 5.8091850
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2261009, 5.2236347
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0287018, 4.0287170
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2393665, 5.2382240
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9322586, 4.9319115
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2234879, 6.2240868
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2816162, 5.2797928
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2350769, 8.2407150
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8688984, 4.8670864
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4412994, 6.4399986
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1609077, 8.1626358
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3883762, 3.3898315
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6411648, 4.6428986
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9166832, 4.9184151
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2613220, 4.2631607
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3321877, 4.3344555
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7612648, 4.7653522
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1530075, 7.1561584
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1767139, 4.1795425
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5221329, 6.5244598
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0495110, 4.0501976
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1285667, 6.1334839
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0042324, 5.0059566
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9833374, 4.9872246
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2949791, 6.2955856
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1440754, 4.1406994
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6450100, 6.6466713
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7190685, 5.7211475
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4116955, 5.4143639
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4203186, 3.4194164
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9193077, 5.9211349
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0296059, 6.0288887
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4976425, 6.4959145
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7807770, 5.7791996
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0059814, 5.0028954
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0314293, 7.0288925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1763

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 756

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6339630, upper bound: 3.6337359
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6344375, upper bound: 3.6332636
time: 5.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3901215, 10.3879242
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2535954, 5.2513428
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2498055, 4.2472801
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3523064, 5.3492622
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0307541, 6.0258560
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1208725, 6.1174011
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4461937, 6.4455681
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8084641, 5.8050213
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2264557, 5.2225876
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0253696, 4.0258713
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2486954, 5.2490883
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9330807, 4.9348259
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2275543, 6.2291298
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2786903, 5.2770004
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2218590, 8.2234001
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8594456, 4.8567505
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4470634, 6.4488564
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1523285, 8.1515045
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3858032, 3.3861980
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6409340, 4.6431103
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9268837, 4.9274788
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2609844, 4.2633533
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3309441, 4.3319283
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7586708, 4.7625751
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1518173, 7.1544495
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1782169, 4.1806316
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5254555, 6.5269394
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0488873, 4.0497055
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1367760, 6.1403770
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9989243, 5.0003490
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9837189, 4.9876156
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2906113, 6.2925835
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1383762, 4.1380463
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6351376, 6.6362839
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7136211, 5.7144165
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4069004, 5.4082184
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4157124, 3.4155560
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9157867, 5.9185562
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0237694, 6.0221863
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4889832, 6.4890518
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7697849, 5.7702446
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0032387, 5.0033875
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0255852, 7.0252533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 835

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1442

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6331019, upper bound: 3.6337741
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6333788, upper bound: 3.6334980
time: 5.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3920441, 10.3860016
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2564793, 5.2484608
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2501564, 4.2469254
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3524246, 5.3491478
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0316658, 6.0249443
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1213264, 6.1169510
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4443016, 6.4474564
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8096466, 5.8038387
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2265816, 5.2224655
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0274601, 4.0237808
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2513695, 5.2464161
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9338264, 4.9340782
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2284584, 6.2282257
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2791634, 5.2765236
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2265701, 8.2186890
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8624477, 4.8537483
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4496613, 6.4462585
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1540070, 8.1498299
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3867912, 3.3852100
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6403236, 4.6437168
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9247475, 4.9296150
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2610149, 4.2633228
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3313065, 4.3315659
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7582054, 4.7630424
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1514435, 7.1548195
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1770878, 4.1817627
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5251846, 6.5272064
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0495052, 4.0490875
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1353569, 6.1417961
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0004425, 4.9988308
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9836426, 4.9876919
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2898064, 6.2933884
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1368809, 4.1395416
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6316624, 6.6397591
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7120037, 5.7160339
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4050541, 5.4100647
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4143677, 3.4169006
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9148331, 5.9195099
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0218430, 6.0241089
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4863853, 6.4916496
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7676983, 5.7723312
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0023651, 5.0042648
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0236855, 7.0271530

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1758

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 629

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6198452, upper bound: 3.6330760
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6198452, upper bound: 3.6330760
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3866501, 10.3829079
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2563820, 5.2513828
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2515297, 4.2486267
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3544312, 5.3514328
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0373993, 6.0317955
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1248436, 6.1210518
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4520721, 6.4532280
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8151283, 5.8104649
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2284966, 5.2244968
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0304661, 4.0292358
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2403107, 5.2382526
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9301300, 4.9298763
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2239342, 6.2246208
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2821426, 5.2800102
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2425041, 8.2393723
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8731499, 4.8678379
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4406891, 6.4398537
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1663284, 8.1636391
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3906326, 3.3900452
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6397972, 4.6419182
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9172325, 4.9197769
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2615662, 4.2628155
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3338776, 4.3345699
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7601013, 4.7640018
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1529694, 7.1557426
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1768532, 4.1803036
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5233727, 6.5250473
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0504322, 4.0506973
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1290512, 6.1337395
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0055275, 5.0050831
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9824409, 4.9858208
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2942390, 6.2967758
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1428223, 4.1438465
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6457081, 6.6501083
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7188110, 5.7208328
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4129562, 5.4160175
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4200478, 3.4210815
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9208107, 5.9244652
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0292015, 6.0296936
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4978867, 6.5004349
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7796516, 5.7819366
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0059776, 5.0070076
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0316315, 7.0332069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1571

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6323754, upper bound: 3.6307088
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6233651, upper bound: 3.6397219
time: 6.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3865814, 10.3829765
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2562447, 5.2515221
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2514229, 4.2487373
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3542595, 5.3516006
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0372658, 6.0319271
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1247292, 6.1211700
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4521408, 6.4531631
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8149147, 5.8106785
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2283287, 5.2246666
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0304489, 4.0292530
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2401962, 5.2383709
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9303417, 4.9296665
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2239761, 6.2245750
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2821388, 5.2800140
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2423096, 8.2395630
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8731155, 4.8678722
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4408226, 6.4397240
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1660461, 8.1639214
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3906403, 3.3900375
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6398888, 4.6418228
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9172363, 4.9197731
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2616196, 4.2627621
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3339577, 4.3344898
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7601967, 4.7639046
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1529999, 7.1557198
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1770020, 4.1801548
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5233841, 6.5250359
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0505028, 4.0506268
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1291275, 6.1336670
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0055389, 5.0050716
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9825668, 4.9856949
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2943573, 6.2966576
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1428146, 4.1438541
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6458378, 6.6499786
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7187309, 5.7209148
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4130020, 5.4159698
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4199982, 3.4211321
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9209709, 5.9243050
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0291481, 6.0297470
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4979019, 6.5004196
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7796555, 5.7819309
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0059814, 5.0070038
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0316544, 7.0331841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1575

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6313674, upper bound: 3.6412653
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6313742, upper bound: 3.6412585
time: 5.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3912659, 10.3893242
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2551613, 5.2541142
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2499123, 4.2464886
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3554230, 5.3513184
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0244598, 6.0176506
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1255379, 6.1200905
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4439316, 6.4425125
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8079910, 5.8059692
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2303600, 5.2250557
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0255871, 4.0271168
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2504482, 5.2514286
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9241905, 4.9301262
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2297249, 6.2329674
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2810555, 5.2775383
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2167664, 8.2229729
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8556290, 4.8514919
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4423256, 6.4458275
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1463470, 8.1463394
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3854198, 3.3876076
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6397514, 4.6419659
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9243393, 4.9265099
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2541351, 4.2575245
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3315411, 4.3321705
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7541199, 4.7584629
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1524506, 7.1559792
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1786747, 4.1813316
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5245934, 6.5276909
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0485783, 4.0514050
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1310654, 6.1351776
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0025635, 5.0039978
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9773827, 4.9851589
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2910957, 6.2930946
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1390362, 4.1367455
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6376038, 6.6365204
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7160797, 5.7157726
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4069176, 5.4070415
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4132481, 3.4120836
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9169083, 5.9195595
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0213966, 6.0199432
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4856529, 6.4808426
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7723579, 5.7708511
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0042953, 5.0031548
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0200157, 7.0203629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1423

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1765

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6318205, upper bound: 3.6209668
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6318205, upper bound: 3.6209668
time: 4.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3906097, 10.3899727
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2551651, 5.2541103
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2491302, 4.2472706
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3548241, 5.3519173
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0232773, 6.0188351
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1242447, 6.1213837
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4442444, 6.4421730
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8079491, 5.8060074
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2286663, 5.2267647
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0257149, 4.0269890
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2500896, 5.2517891
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9241238, 4.9301929
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2314262, 6.2312660
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2809181, 5.2776794
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2158356, 8.2239037
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8548965, 4.8522549
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4427681, 6.4453812
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1446190, 8.1480675
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3853951, 3.3876324
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6390343, 4.6426849
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9243431, 4.9265060
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2528896, 4.2587700
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3314152, 4.3322983
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7530594, 4.7595272
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1518478, 7.1565781
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1786137, 4.1813927
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5240746, 6.5282097
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0486774, 4.0513058
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1303024, 6.1359406
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0012932, 5.0052662
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9776878, 4.9848537
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2909660, 6.2932243
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1392117, 4.1365623
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6374817, 6.6366386
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7162437, 5.7156105
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4070168, 5.4069443
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4139709, 3.4113598
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9171638, 5.9193230
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0225449, 6.0187988
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4844666, 6.4820251
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7724152, 5.7707920
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0046692, 5.0027809
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0208778, 7.0195007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1391

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6399509, upper bound: 3.6226248
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6390465, upper bound: 3.6235304
time: 5.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4268875, 10.4158478
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2730637, 5.2658710
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2599525, 4.2530785
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3449821, 5.3390274
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0070648, 5.9939232
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1160316, 6.1072350
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4390488, 6.4409752
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8052139, 5.7989616
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2041664, 5.1928825
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0121384, 4.0095787
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2453117, 5.2424488
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9114246, 4.9210339
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2214394, 6.2248573
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2658234, 5.2598648
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2284546, 8.2272072
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8398895, 4.8290176
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4480629, 6.4523430
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1474915, 8.1450996
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3727818, 3.3760986
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6393585, 4.6424332
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9304543, 4.9351883
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2477608, 4.2542610
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3341770, 4.3333683
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7427864, 4.7514324
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1371765, 7.1431274
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1722126, 4.1757259
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5369987, 6.5420685
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0397549, 4.0429688
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1371422, 6.1463051
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0023422, 5.0034027
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9462051, 4.9598675
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2905922, 6.2931328
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1373539, 4.1374569
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6204567, 6.6245766
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6855564, 5.6919460
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3800011, 5.3864155
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4116611, 3.4133453
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9031487, 5.9089432
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0218239, 6.0184593
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4897003, 6.4884148
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7773800, 5.7800922
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9988251, 5.0001335
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0247726, 7.0251770

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 685

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6374903, upper bound: 3.6211525
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6349207, upper bound: 3.6237214
time: 5.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3909302, 10.3888779
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2568226, 5.2519932
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2468948, 4.2469711
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3481522, 5.3482246
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0259781, 6.0238132
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1174927, 6.1167641
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4455833, 6.4461136
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8054428, 5.8036575
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2190704, 5.2189484
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0266571, 4.0238380
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2518044, 5.2490368
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9351978, 4.9381294
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2281570, 6.2269936
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2770576, 5.2747574
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2198143, 8.2193298
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8595276, 4.8527946
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4499741, 6.4477196
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1437531, 8.1478577
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3861256, 3.3853817
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6383457, 4.6429119
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9245148, 4.9305992
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2587986, 4.2629032
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3281059, 4.3298874
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7592583, 4.7634716
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1515198, 7.1551170
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1772270, 4.1806908
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5243721, 6.5274162
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0482140, 4.0485134
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1351585, 6.1407738
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0004025, 4.9992905
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9814396, 4.9861202
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2884407, 6.2935295
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1387634, 4.1384544
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6334934, 6.6381836
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7107315, 5.7098885
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4040966, 5.4059677
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4134817, 3.4131365
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9132843, 5.9140320
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0219231, 6.0234756
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4880104, 6.4909515
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7708511, 5.7700310
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0016365, 5.0004272
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0244255, 7.0264549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1563

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6307844, upper bound: 3.6213887
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6160220, upper bound: 3.6361402
time: 5.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3907013, 10.3894730
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2568378, 5.2520790
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2474976, 4.2462177
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3498421, 5.3463020
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0270958, 6.0220833
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1185493, 6.1153946
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4455528, 6.4461212
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8061676, 5.8032379
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2203522, 5.2179031
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0264091, 4.0239124
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2510262, 5.2491627
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9342632, 4.9390049
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2276382, 6.2273750
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2772636, 5.2745972
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2187500, 8.2206154
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8595829, 4.8528862
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4497604, 6.4478989
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1429939, 8.1489906
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3861485, 3.3852711
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6367092, 4.6443748
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9245148, 4.9305992
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2573013, 4.2641487
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3274956, 4.3313408
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7576485, 4.7649136
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1511002, 7.1556473
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1757126, 4.1821690
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5244560, 6.5274734
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0481644, 4.0486145
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1340599, 6.1418343
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9976978, 5.0017071
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9811192, 4.9867382
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2874680, 6.2947350
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1390572, 4.1384335
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6333942, 6.6384583
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7110825, 5.7091656
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4041901, 5.4053993
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4139662, 3.4119825
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9125404, 5.9148483
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0230560, 6.0220222
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4885483, 6.4908218
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7716064, 5.7697735
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0015831, 5.0004425
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0231438, 7.0265388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1399

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 753

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6289634, upper bound: 3.6332662
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6343707, upper bound: 3.6278574
time: 6.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3914185, 10.3819542
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2573338, 5.2479858
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2485371, 4.2423801
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3506622, 5.3451080
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0290680, 6.0181847
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1189117, 6.1116753
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4428329, 6.4472809
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8061523, 5.7992477
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2223301, 5.2157078
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0242958, 4.0206261
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2539406, 5.2469616
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9352436, 4.9388885
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2274094, 6.2274170
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2764702, 5.2712975
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2234039, 8.2127609
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8617153, 4.8492050
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4473343, 6.4447327
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1498680, 8.1402130
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3851547, 3.3847141
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6400928, 4.6419487
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9256744, 4.9322147
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2602711, 4.2627316
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3316555, 4.3287296
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7596092, 4.7653923
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1503830, 7.1546822
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1772385, 4.1819172
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5223579, 6.5253830
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0465393, 4.0479794
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1361656, 6.1440163
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0026131, 4.9987202
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9793968, 4.9874630
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2912369, 6.2923393
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1372108, 4.1399994
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6295433, 6.6371040
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7042542, 5.7115269
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4002686, 5.4067135
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4097176, 3.4141188
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9094543, 5.9165878
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0208893, 6.0236359
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4883003, 6.4912071
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7659950, 5.7730179
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9995232, 5.0024529
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0224686, 7.0262375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 750

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1391

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6342263, upper bound: 3.6373021
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6333218, upper bound: 3.6382080
time: 5.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3902817, 10.3830910
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2567463, 5.2485733
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2482700, 4.2426510
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3505707, 5.3451996
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0284882, 6.0187645
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1184998, 6.1120911
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4431419, 6.4469719
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8057060, 5.7996941
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2220631, 5.2159729
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0235481, 4.0213737
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2527313, 5.2481728
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9352341, 4.9388962
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2274094, 6.2274170
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2758827, 5.2718849
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2231903, 8.2129745
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8613091, 4.8496113
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4464989, 6.4455681
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1493912, 8.1406860
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3853283, 3.3845425
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6400204, 4.6420193
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9259644, 4.9319248
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2602253, 4.2627773
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3315964, 4.3287907
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7596970, 4.7653065
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1507034, 7.1543655
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1773720, 4.1817856
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5229645, 6.5247765
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0468712, 4.0476456
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1366158, 6.1435623
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0023270, 4.9990044
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9798355, 4.9870205
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2912140, 6.2923622
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1373863, 4.1398258
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6302032, 6.6364441
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7049408, 5.7108402
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4008827, 5.4060993
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4100952, 3.4137402
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9095383, 5.9165001
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0211105, 6.0234146
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4883156, 6.4911919
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7665367, 5.7724800
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9995232, 5.0024490
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0225754, 7.0261345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1619

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1399

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6305622, upper bound: 3.6351652
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6333496, upper bound: 3.6323771
time: 5.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4105606, 10.4014168
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2608128, 5.2561321
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2518387, 4.2479229
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3383179, 5.3364944
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0131416, 6.0070667
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1097794, 6.1052551
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4510460, 6.4551735
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7992630, 5.7969837
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2024193, 5.1960964
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0100365, 4.0050907
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2275200, 5.2212391
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9244728, 4.9289074
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2166595, 6.2175255
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2647400, 5.2604637
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2309113, 8.2257233
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8540745, 4.8485775
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4427910, 6.4417725
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1561966, 8.1540413
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3775215, 3.3809719
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6294785, 4.6294289
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9093018, 4.9127235
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2430363, 4.2457218
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3344231, 4.3336735
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7409973, 4.7437077
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1359634, 7.1388092
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1588898, 4.1604748
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5320396, 6.5363770
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0428562, 4.0453186
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1175995, 6.1221199
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0084095, 5.0083179
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9436951, 4.9517708
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2882843, 6.2890892
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1432419, 4.1440506
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6145821, 6.6204948
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6889763, 5.6950016
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3846741, 5.3900642
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4147816, 3.4168625
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9023094, 5.9051399
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0265999, 6.0260086
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4922600, 6.4921265
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7804852, 5.7852859
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0029640, 5.0039825
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0347290, 7.0365868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1423

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6388090, upper bound: 3.6292412
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6380702, upper bound: 3.6299799
time: 9.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4069824, 10.4049950
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2569866, 5.2599564
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2492371, 4.2505245
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3357048, 5.3391037
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0088387, 6.0113678
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1074715, 6.1075630
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4537582, 6.4524612
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7969856, 5.7992611
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.1995468, 5.1989670
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0080547, 4.0070724
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2241821, 5.2245770
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9259396, 4.9274406
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2163315, 6.2178535
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2623215, 5.2628822
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2259941, 8.2306404
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8493328, 4.8533192
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4420815, 6.4424782
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1553841, 8.1548538
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3775959, 3.3808975
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6324577, 4.6264496
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9130669, 4.9089584
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2460060, 4.2427502
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3351555, 4.3329411
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7443199, 4.7403831
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1382599, 7.1365166
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1630592, 4.1563053
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5334854, 6.5349312
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0437775, 4.0443954
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1216354, 6.1180840
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0086117, 5.0081139
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9474564, 4.9480076
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2916870, 6.2856865
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1435852, 4.1437092
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6190987, 6.6159744
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6890793, 5.6948986
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3859558, 5.3887806
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4150505, 3.4165936
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9055595, 5.9018898
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0265656, 6.0260468
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4943199, 6.4900665
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7829685, 5.7828026
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0037575, 5.0031853
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0378571, 7.0334549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 928

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6382591, upper bound: 3.6294914
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6392771, upper bound: 3.6284725
time: 5.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4096222, 10.4253654
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2648945, 5.2792301
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2546272, 4.2632828
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3416786, 5.3476791
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0009918, 6.0150928
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1104126, 6.1206322
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4474754, 6.4462852
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8012085, 5.8142719
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.1937637, 5.2054138
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0118713, 4.0170879
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2326584, 5.2390499
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9300747, 4.9282475
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2193336, 6.2176247
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2652855, 5.2712669
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2416153, 8.2487907
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8453465, 4.8584061
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4443588, 6.4477692
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1580467, 8.1643105
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3800259, 3.3787289
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6393433, 4.6360550
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9251213, 4.9189091
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2527008, 4.2470207
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3365421, 4.3363419
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7515221, 4.7428112
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1453705, 7.1388359
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1757698, 4.1697311
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5420265, 6.5358391
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0464954, 4.0472927
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1385956, 6.1272049
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0081367, 5.0098248
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9636822, 4.9503613
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2963982, 6.2932396
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1418552, 4.1405792
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6422348, 6.6274681
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7025032, 5.6880417
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4009209, 5.3848305
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4195786, 3.4156370
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9177628, 5.9081001
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0259857, 6.0269547
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.5032921, 6.4968033
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7917328, 5.7874374
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0032005, 5.0018921
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0328445, 7.0337181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 534

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6251864, upper bound: 3.6294541
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6251864, upper bound: 3.6294541
time: 5.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4103317, 10.4230003
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2661686, 5.2753143
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2550049, 4.2620335
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3417358, 5.3477554
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0010567, 6.0150204
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1107483, 6.1203384
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4488678, 6.4449806
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8020554, 5.8099117
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.1939011, 5.2053127
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0119190, 4.0171013
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2326965, 5.2390594
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9314766, 4.9226818
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2195549, 6.2168808
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2649612, 5.2715607
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2423325, 8.2489357
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8452435, 4.8587723
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4453659, 6.4439621
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1582031, 8.1620331
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3803902, 3.3773041
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6393318, 4.6362476
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9252357, 4.9187565
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2528305, 4.2470398
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3359299, 4.3368073
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7518311, 4.7428417
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1455078, 7.1389542
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1748962, 4.1698666
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5420952, 6.5357552
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0476742, 4.0440388
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1385117, 6.1275253
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0083809, 5.0085506
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9636898, 4.9503384
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2957954, 6.2933960
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1422939, 4.1413078
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6359100, 6.6279182
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6968460, 5.6882439
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3942757, 5.3854103
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4188309, 3.4156914
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9151039, 5.9082222
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0259018, 6.0269623
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4982948, 6.4975967
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7916489, 5.7874508
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0040703, 5.0013542
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0338974, 7.0312691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 650

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 784

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6291487, upper bound: 3.6362048
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6319392, upper bound: 3.6334156
time: 5.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3850250, 10.3884354
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2540550, 5.2594719
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2480412, 4.2508278
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3550262, 5.3581848
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0332756, 6.0397778
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1207809, 6.1260567
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4567642, 6.4585381
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8061638, 5.8113403
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2272434, 5.2305889
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0241699, 4.0265236
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2429390, 5.2472191
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9518719, 4.9495564
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2281647, 6.2288399
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2805443, 5.2842255
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2329559, 8.2325859
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8721161, 4.8780594
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4491043, 6.4524841
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1653595, 8.1647377
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3901978, 3.3887005
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6426067, 4.6393890
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9217300, 4.9171448
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2629528, 4.2593594
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3341618, 4.3327751
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7627487, 4.7563610
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1554947, 7.1500168
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1784744, 4.1732979
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5313072, 6.5254440
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0564060, 4.0550995
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1378479, 6.1298485
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0050392, 5.0048256
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9925079, 4.9858360
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2963753, 6.2942200
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1440315, 4.1463070
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6436729, 6.6382370
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7168083, 5.7143326
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4121914, 5.4075909
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4190855, 3.4186516
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9252625, 5.9218597
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0246315, 6.0232468
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4954681, 6.4951973
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7783203, 5.7789536
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0082626, 5.0094185
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0337143, 7.0329552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1580

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1634

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6366584, upper bound: 3.6313538
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6350698, upper bound: 3.6329410
time: 5.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3855209, 10.3879318
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2552528, 5.2582741
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2480259, 4.2508430
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3546028, 5.3586044
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0332756, 6.0397778
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1207809, 6.1260567
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4568329, 6.4584656
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8068504, 5.8106537
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2267208, 5.2311115
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0252991, 4.0253925
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2450371, 5.2451248
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9533024, 4.9481239
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2293777, 6.2276306
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2806282, 5.2841415
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2349281, 8.2306137
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8726654, 4.8775082
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4514427, 6.4501457
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1656876, 8.1644096
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3903885, 3.3885078
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6425152, 4.6394787
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9207611, 4.9181175
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2631187, 4.2591934
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3341465, 4.3327904
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7625313, 4.7565746
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1550827, 7.1504288
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1776714, 4.1741009
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5305061, 6.5262451
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0570583, 4.0544491
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1370430, 6.1306534
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0059853, 5.0038795
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9926605, 4.9856815
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2962608, 6.2943344
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1440239, 4.1463146
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6416626, 6.6402435
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7161674, 5.7149734
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4106998, 5.4090786
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4183779, 3.4193592
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9251633, 5.9219589
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0231247, 6.0247536
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4941711, 6.4964981
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7781906, 5.7790890
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0083427, 5.0093384
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0335922, 7.0330734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 643

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6314566, upper bound: 3.6379080
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6314271, upper bound: 3.6379375
time: 4.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3846359, 10.3883171
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2497196, 5.2555008
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2490540, 4.2513695
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3522148, 5.3552437
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0309563, 6.0357914
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1210175, 6.1238251
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4531670, 6.4523544
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8059921, 5.8109379
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2237759, 5.2272282
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0280571, 4.0293655
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2360764, 5.2409801
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9440289, 4.9449863
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2227440, 6.2233734
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2822609, 5.2835083
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2302475, 8.2405777
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8697777, 4.8771381
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4439545, 6.4460907
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1552773, 8.1630783
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3888187, 3.3904762
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6420135, 4.6395397
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9176674, 4.9143124
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2591991, 4.2581539
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3312283, 4.3327885
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7642899, 4.7607555
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1558685, 7.1541634
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1795979, 4.1768742
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5237579, 6.5234756
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0525341, 4.0537090
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1320801, 6.1277771
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0008106, 5.0046749
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9884224, 4.9872761
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2958984, 6.2940826
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1420231, 4.1408653
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6457520, 6.6423492
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7186642, 5.7176647
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4135017, 5.4110107
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4200268, 3.4165087
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9232674, 5.9197540
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0285873, 6.0248795
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4977341, 6.4926796
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7815361, 5.7791958
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0065460, 5.0042839
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0335007, 7.0312843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1765

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6339963, upper bound: 3.6225391
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6339963, upper bound: 3.6225391
time: 5.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3846359, 10.3883171
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2496891, 5.2555275
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2491188, 4.2513046
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3522377, 5.3552208
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0309219, 6.0358276
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1210785, 6.1237602
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4531670, 6.4523544
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8059311, 5.8109989
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2238522, 5.2271519
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0280495, 4.0293732
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2359734, 5.2410812
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9438095, 4.9452076
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2225838, 6.2235336
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2822227, 5.2835464
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2300339, 8.2407951
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8697815, 4.8771343
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4438286, 6.4462166
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1550026, 8.1633530
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3886318, 3.3906631
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6419487, 4.6396065
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9177055, 4.9142742
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2590027, 4.2583504
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3309288, 4.3330898
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7641945, 4.7608490
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1558609, 7.1541748
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1795902, 4.1768818
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5237083, 6.5235291
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0525303, 4.0537148
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1320343, 6.1278229
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0005550, 5.0049286
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9884109, 4.9872856
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2958794, 6.2941017
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1421223, 4.1407700
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6457901, 6.6423111
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7186871, 5.7176418
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4135437, 5.4109650
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4201336, 3.4164038
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9232750, 5.9197464
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0287819, 6.0246811
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4977951, 6.4926186
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7817383, 5.7789917
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0065536, 5.0042763
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0335007, 7.0312843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1763

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1634

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6396205, upper bound: 3.6262641
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6380318, upper bound: 3.6278517
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3821640, 10.3843651
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2486839, 5.2533817
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2479477, 4.2500534
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3511314, 5.3542328
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0303688, 6.0350723
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1192665, 6.1230278
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4516029, 6.4529076
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8042030, 5.8088417
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2231712, 5.2261047
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0256310, 4.0263023
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2372303, 5.2414703
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9445744, 4.9441166
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2229385, 6.2237892
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2797470, 5.2819443
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2297363, 8.2350540
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8696308, 4.8760357
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4410896, 6.4436989
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1552238, 8.1614151
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3886528, 3.3887653
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6413555, 4.6389713
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9190025, 4.9154587
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2604847, 4.2586613
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3313217, 4.3325043
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7649460, 4.7606239
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1554871, 7.1523399
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1797218, 4.1766453
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5236778, 6.5208626
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0519524, 4.0519676
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1337814, 6.1284256
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0011597, 5.0037003
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9882507, 4.9848900
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2957954, 6.2944336
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1406689, 4.1418018
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6432533, 6.6406898
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7160416, 5.7152252
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4123116, 5.4100075
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4178295, 3.4156284
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9227371, 5.9194107
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0269775, 6.0233383
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4958229, 6.4935188
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7792130, 5.7782459
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0054741, 5.0051689
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0325775, 7.0312996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 571

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1572

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6349937, upper bound: 3.6244217
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6302154, upper bound: 3.6292002
time: 5.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3810349, 10.3855095
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2480965, 5.2539711
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2476768, 4.2503262
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3510437, 5.3543243
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0297890, 6.0356522
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1188545, 6.1234398
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4519119, 6.4525986
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8037605, 5.8092842
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2229042, 5.2263718
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0248833, 4.0270500
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2360172, 5.2426796
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9445667, 4.9441242
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2229385, 6.2237892
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2791634, 5.2825317
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2295227, 8.2352676
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8692265, 4.8764400
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4402542, 6.4445343
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1547470, 8.1618919
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3888264, 3.3885918
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6412868, 4.6390419
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9192886, 4.9151688
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2604389, 4.2587070
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3312607, 4.3325634
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7650299, 4.7605381
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1558075, 7.1520195
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1798553, 4.1765137
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5242844, 6.5202560
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0522842, 4.0516357
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1342354, 6.1279716
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0008736, 5.0039864
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9886932, 4.9844494
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2957726, 6.2944565
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1408367, 4.1416283
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6439171, 6.6400299
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7167320, 5.7145367
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4129257, 5.4093933
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4182072, 3.4152508
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9228249, 5.9193230
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0271988, 6.0231171
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4958382, 6.4934998
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7797508, 5.7777061
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0054741, 5.0051651
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0326843, 7.0311928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1765

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1648

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6325694, upper bound: 3.6249498
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6325410, upper bound: 3.6249784
time: 8.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3761978, 10.3828621
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2442131, 5.2505283
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2511082, 4.2553616
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3465424, 5.3498154
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0283546, 6.0363388
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1165428, 6.1215744
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4490242, 6.4477844
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8081856, 5.8133087
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2124577, 5.2175007
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0268955, 4.0294456
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2351646, 5.2392216
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9406509, 4.9374313
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2255135, 6.2248688
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2807236, 5.2849464
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2274437, 8.2265053
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8686047, 4.8752518
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4408646, 6.4414330
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1551590, 8.1544762
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3851376, 3.3822021
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6390457, 4.6360817
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9156723, 4.9105339
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2544289, 4.2502766
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3272305, 4.3259869
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7556953, 4.7498398
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1529274, 7.1483498
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1754570, 4.1702919
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5324249, 6.5278397
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0539188, 4.0518322
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1275215, 6.1197624
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9930248, 4.9924030
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9862061, 4.9795322
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2965279, 6.2933121
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1375351, 4.1373730
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6472187, 6.6418228
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7192612, 5.7162971
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4122753, 5.4071198
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4172001, 3.4171696
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9236259, 5.9192924
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0229874, 6.0249748
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4877739, 6.4871445
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7688389, 5.7681408
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0063477, 5.0071869
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0307465, 7.0299721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1442

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6277352, upper bound: 3.6358713
time: 8.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6279112, upper bound: 3.6357733
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3768082, 10.3822517
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2447052, 5.2500343
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2506084, 4.2558594
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3452339, 5.3511238
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0268135, 6.0378799
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1147995, 6.1233139
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4478111, 6.4489975
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8075562, 5.8139381
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2102070, 5.2197475
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0270004, 4.0293427
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2355537, 5.2388344
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9417439, 4.9363403
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2266769, 6.2237091
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2807236, 5.2849464
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2294884, 8.2244606
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8689213, 4.8749352
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4408951, 6.4414024
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1565971, 8.1530380
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3858700, 3.3814697
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6398849, 4.6352406
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9155769, 4.9106293
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2555046, 4.2492008
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3290195, 4.3241978
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7579231, 4.7476101
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1541100, 7.1471710
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1766186, 4.1691303
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5337486, 6.5265160
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0539722, 4.0517788
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1291580, 6.1181259
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9957256, 4.9896984
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9876022, 4.9781361
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2965012, 6.2933388
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1362762, 4.1386375
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6472378, 6.6418037
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7190781, 5.7164803
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4125881, 5.4068050
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4165363, 3.4178333
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9241257, 5.9187927
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0221405, 6.0258217
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4854813, 6.4894409
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7673130, 5.7696667
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0059128, 5.0076218
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0303802, 7.0303307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1634

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6164620, upper bound: 3.6287502
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6156396, upper bound: 3.6296714
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3860321, 10.3846893
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2523975, 5.2500172
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2493477, 4.2509136
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3523521, 5.3531799
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0316849, 6.0354042
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1218987, 6.1230125
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4537201, 6.4525757
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8078270, 5.8078365
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2240200, 5.2231617
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0286026, 4.0260429
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2402687, 5.2318306
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9465294, 4.9421520
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2248001, 6.2211914
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2822418, 5.2827682
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2390480, 8.2295761
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8726292, 4.8716335
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4465027, 6.4417076
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1628304, 8.1543045
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3880920, 3.3894386
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6401157, 4.6392784
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9145317, 4.9147453
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2591839, 4.2579575
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3328342, 4.3306980
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7606945, 4.7604771
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1544495, 7.1541519
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1763439, 4.1769505
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5239677, 6.5242462
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0538235, 4.0527115
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1278801, 6.1276283
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0049820, 5.0004616
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9880753, 4.9851151
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2946320, 6.2943306
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1418839, 4.1413002
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6436348, 6.6439476
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7185707, 5.7174397
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4093819, 5.4118652
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4163046, 3.4196062
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9223709, 5.9210281
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0237312, 6.0278397
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4936905, 6.4966431
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7801819, 5.7810612
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0052872, 5.0058746
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0327835, 7.0321541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1423

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6271030, upper bound: 3.6379001
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6263641, upper bound: 3.6386390
time: 6.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3863449, 10.3843307
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2525272, 5.2498150
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2495461, 4.2507858
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3524971, 5.3531151
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0317993, 6.0353546
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1221237, 6.1228027
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4537430, 6.4525909
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8082848, 5.8072491
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2241421, 5.2228947
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0288887, 4.0257835
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2404785, 5.2320251
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9469910, 4.9419231
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2246666, 6.2212982
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2822952, 5.2828026
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2396889, 8.2291908
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8726730, 4.8716373
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4470406, 6.4414635
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1632767, 8.1539726
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3880920, 3.3894367
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6401234, 4.6392727
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9145775, 4.9147568
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2592087, 4.2579269
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3328533, 4.3307495
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7606983, 4.7604733
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1544571, 7.1541443
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1765003, 4.1770248
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5240326, 6.5243301
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0538960, 4.0524902
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1278992, 6.1276474
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0050926, 5.0003414
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9883041, 4.9849281
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2947044, 6.2943764
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1416512, 4.1414700
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6435432, 6.6443748
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7184219, 5.7175751
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4089546, 5.4122200
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4161158, 3.4197884
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9224281, 5.9212303
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0236015, 6.0280991
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4936752, 6.4969444
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7801704, 5.7811871
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0052986, 5.0058975
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0328064, 7.0321503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1571

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1594

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6217412, upper bound: 3.6385853
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6266630, upper bound: 3.6336643
time: 5.33 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 12.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6237618, upper bound: 3.6324880
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6313499, upper bound: 3.6249021
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6305444, upper bound: 3.6177592
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6157744, upper bound: 3.6325235
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6339698, upper bound: 3.6335433
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6342458, upper bound: 3.6332674
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6339630, upper bound: 3.6337359
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6344375, upper bound: 3.6332636
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6331019, upper bound: 3.6337741
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6333788, upper bound: 3.6334980
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6198452, upper bound: 3.6330760
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6198452, upper bound: 3.6330760
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6323754, upper bound: 3.6307088
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6233651, upper bound: 3.6397219
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6313674, upper bound: 3.6412653
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6313742, upper bound: 3.6412585
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6318205, upper bound: 3.6209668
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6318205, upper bound: 3.6209668
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6399509, upper bound: 3.6226248
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6390465, upper bound: 3.6235304
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6374903, upper bound: 3.6211525
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6349207, upper bound: 3.6237214
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6307844, upper bound: 3.6213887
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6160220, upper bound: 3.6361402
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6289634, upper bound: 3.6332662
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6343707, upper bound: 3.6278574
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6342263, upper bound: 3.6373021
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6333218, upper bound: 3.6382080
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6305622, upper bound: 3.6351652
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6333496, upper bound: 3.6323771
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6388090, upper bound: 3.6292412
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6380702, upper bound: 3.6299799
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6382591, upper bound: 3.6294914
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6392771, upper bound: 3.6284725
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6251864, upper bound: 3.6294541
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6251864, upper bound: 3.6294541
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6291487, upper bound: 3.6362048
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6319392, upper bound: 3.6334156
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6366584, upper bound: 3.6313538
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6350698, upper bound: 3.6329410
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6314566, upper bound: 3.6379080
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6314271, upper bound: 3.6379375
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6339963, upper bound: 3.6225391
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6339963, upper bound: 3.6225391
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6396205, upper bound: 3.6262641
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6380318, upper bound: 3.6278517
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6349937, upper bound: 3.6244217
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6302154, upper bound: 3.6292002
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6325694, upper bound: 3.6249498
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6325410, upper bound: 3.6249784
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6277352, upper bound: 3.6358713
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6279112, upper bound: 3.6357733
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6164620, upper bound: 3.6287502
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6156396, upper bound: 3.6296714
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6271030, upper bound: 3.6379001
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6263641, upper bound: 3.6386390
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6217412, upper bound: 3.6385853
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 12.42
Output dim: 38, lower bound: -3.6266630, upper bound: 3.6336643

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3793106, 10.3703194
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2500286, 5.2426357
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2476044, 4.2427921
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3513527, 5.3467598
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0255470, 6.0164146
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1197853, 6.1124077
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4432297, 6.4449997
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8088570, 5.8011703
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2219677, 5.2156754
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0232258, 4.0204754
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2350159, 5.2303467
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9365559, 4.9378548
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2198753, 6.2218475
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2784309, 5.2745590
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2337914, 8.2309761
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8603916, 4.8522606
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4382515, 6.4357605
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1526070, 8.1434898
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3875446, 3.3877811
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6420479, 4.6421070
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9091530, 4.9126854
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2619610, 4.2627563
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3286610, 4.3261185
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7610512, 4.7655048
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1547089, 7.1581421
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1772461, 4.1796818
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5194969, 6.5218658
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0512600, 4.0512486
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1231499, 6.1292725
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0028458, 4.9996948
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9843521, 4.9897652
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2963371, 6.2958755
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1355152, 4.1374531
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6380844, 6.6433029
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7042542, 5.7119389
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4011879, 5.4082451
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4145660, 3.4179373
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9145889, 5.9191971
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0251617, 6.0273285
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4950752, 6.4958649
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7686501, 5.7736282
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0000191, 5.0015984
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0239182, 7.0250092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1594

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1648

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6200818, upper bound: 3.6287711
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6200532, upper bound: 3.6287997
time: 5.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3856659, 10.3864746
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2545662, 5.2516994
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2502270, 4.2481117
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3528404, 5.3493729
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0312195, 6.0252399
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1233063, 6.1199150
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4513397, 6.4503212
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8097000, 5.8031464
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2229214, 5.2207222
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0271568, 4.0274830
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2393303, 5.2381458
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9320736, 4.9293346
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2232170, 6.2229042
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2810135, 5.2791443
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2381134, 8.2364807
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8687973, 4.8627090
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4413681, 6.4403114
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1571693, 8.1510086
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3887310, 3.3874378
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6402473, 4.6443577
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9147148, 4.9174461
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2608280, 4.2631836
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3313351, 4.3321819
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7589722, 4.7654095
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1516571, 7.1560173
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1749458, 4.1805935
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5221214, 6.5241051
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0502262, 4.0489731
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1282959, 6.1334381
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0044899, 5.0059586
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9827271, 4.9853878
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2942162, 6.2969398
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1421528, 4.1406403
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6399765, 6.6464386
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7111168, 5.7133980
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4033184, 5.4083080
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4173174, 3.4185677
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9152641, 5.9208832
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0237160, 6.0293388
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4929657, 6.4972382
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7804756, 5.7801437
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0043678, 5.0044975
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0284462, 7.0306702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1612

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1391

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6157182, upper bound: 3.6315631
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6148129, upper bound: 3.6324712
time: 5.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3849945, 10.3862724
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2525501, 5.2546043
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2490559, 4.2485867
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3533897, 5.3513565
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0331688, 6.0287647
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1231346, 6.1200104
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4516869, 6.4502716
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8115311, 5.8131008
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2261944, 5.2240353
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0286789, 4.0287247
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2389526, 5.2377377
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9310932, 4.9374943
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2228279, 6.2242317
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2817917, 5.2793121
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2351074, 8.2411003
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8690033, 4.8667259
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4404602, 6.4438438
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1604118, 8.1642914
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3882408, 3.3913021
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6416283, 4.6430874
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9166489, 4.9186954
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2604027, 4.2621574
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3331337, 4.3340263
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7611408, 4.7654114
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1528816, 7.1560402
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1775742, 4.1793861
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5214310, 6.5238686
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0482864, 4.0533981
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1292267, 6.1336975
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0041618, 5.0071487
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9837475, 4.9876537
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2959099, 6.2957382
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1438370, 4.1402702
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6508484, 6.6457748
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7248230, 5.7210693
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4181080, 5.4135990
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4213114, 3.4197149
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9214058, 5.9204483
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0303612, 6.0297470
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.5028915, 6.4954338
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7812424, 5.7797680
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0053177, 5.0036469
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0304337, 7.0314140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 680

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1761

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6336534, upper bound: 3.6275380
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6279616, upper bound: 3.6332271
time: 5.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3849945, 10.3862801
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2525234, 5.2546310
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2491245, 4.2485218
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3534088, 5.3513374
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0331306, 6.0288010
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1231995, 6.1199455
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4516869, 6.4502716
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8114700, 5.8131618
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2262707, 5.2239590
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0286713, 4.0287323
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2388496, 5.2378387
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9308720, 4.9377155
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2226677, 6.2243919
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2817535, 5.2793503
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2348938, 8.2413177
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8690052, 4.8667240
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4403343, 6.4439697
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1601334, 8.1645699
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3880539, 3.3914890
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6415596, 4.6431522
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9166870, 4.9186573
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2602062, 4.2623539
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3328342, 4.3343258
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7610493, 4.7655067
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1528740, 7.1560516
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1775646, 4.1793938
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5213814, 6.5239182
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0482807, 4.0534039
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1291771, 6.1337433
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0039101, 5.0074024
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9837360, 4.9876652
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2958908, 6.2957573
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1439323, 4.1401749
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6508904, 6.6457367
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7248459, 5.7210464
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4181538, 5.4135532
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4214182, 3.4196091
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9214134, 5.9204369
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0305557, 6.0295563
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.5029526, 6.4953728
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7814445, 5.7795639
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0053253, 5.0036392
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0304413, 7.0314102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 726

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 928

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6331331, upper bound: 3.6331721
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6341506, upper bound: 3.6321546
time: 5.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3767853, 10.3714142
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2428741, 5.2359333
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2424412, 4.2376690
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3473625, 5.3427200
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0384254, 6.0296211
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1172676, 6.1111374
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4569969, 6.4556007
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8056068, 5.7997570
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2271233, 5.2217865
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0227432, 4.0207806
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2295380, 5.2250595
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9304352, 4.9315510
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2224121, 6.2226830
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2770195, 5.2727814
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2201462, 8.2208633
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8704071, 4.8638554
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4387627, 6.4367523
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1562538, 8.1571617
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3885517, 3.3900852
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6312523, 4.6359596
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9037590, 4.9092541
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2542667, 4.2590752
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3292732, 4.3322735
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7546005, 4.7620163
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1469650, 7.1524010
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1618652, 4.1688614
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5166550, 6.5204353
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0466881, 4.0482960
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1132355, 6.1221848
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0039158, 5.0058479
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9721718, 4.9798203
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2872810, 6.2912903
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1455231, 4.1424961
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6281548, 6.6343307
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7183552, 5.7205353
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4082813, 5.4122353
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4187126, 3.4180794
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9096756, 5.9147491
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0294914, 6.0287437
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4904633, 6.4907951
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7728577, 5.7737675
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0059128, 5.0036201
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0324173, 7.0330124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 759

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 731

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6334098, upper bound: 3.6335041
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6337296, upper bound: 3.6331922
time: 5.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3732071, 10.3749847
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2390442, 5.2397594
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2398357, 4.2402706
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3447533, 5.3453331
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0341225, 6.0339203
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1149597, 6.1134453
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4597130, 6.4528847
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8033257, 5.8020382
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2242546, 5.2246571
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0207615, 4.0227623
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2262001, 5.2283974
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9319019, 4.9300842
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2220879, 6.2230110
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2746010, 5.2752037
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2152328, 8.2257805
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8656654, 4.8685970
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4380569, 6.4374619
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1554375, 8.1579742
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3886280, 3.3900108
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6342278, 4.6329803
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9075203, 4.9054852
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2572384, 4.2561035
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3300056, 4.3315411
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7579269, 4.7586899
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1492615, 7.1501083
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1660347, 4.1646938
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5181046, 6.5189857
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0476093, 4.0473728
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1172714, 6.1181526
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0041218, 5.0056419
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9759369, 4.9760551
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2906837, 6.2878914
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1458664, 4.1421547
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6326714, 6.6298141
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7184582, 5.7204323
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4095669, 5.4109516
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4189816, 3.4178104
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9129295, 5.9114990
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0294571, 6.0287781
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4925232, 6.4887390
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7753410, 5.7712841
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0067101, 5.0028267
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0355453, 7.0298805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 753

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1569

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6226309, upper bound: 3.6214182
time: 10.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6225964, upper bound: 3.6214528
time: 5.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3897858, 10.3875923
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2535515, 5.2512703
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2492676, 4.2468109
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3523216, 5.3492966
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0297546, 6.0248203
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1200294, 6.1166191
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4464149, 6.4457893
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8080864, 5.8045845
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2266769, 5.2228870
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0253963, 4.0258904
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2483139, 5.2486076
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9333134, 4.9348373
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2271194, 6.2285309
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2785378, 5.2768097
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2226067, 8.2239342
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8594437, 4.8567505
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4472237, 6.4488907
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1519852, 8.1508827
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3860283, 3.3862343
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6413727, 4.6434860
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9269638, 4.9275970
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2601986, 4.2623730
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3312740, 4.3319569
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7588539, 4.7626629
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1518250, 7.1544418
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1782036, 4.1806107
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5248260, 6.5262604
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0488415, 4.0496559
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1373482, 6.1409073
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9990959, 5.0002651
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9841404, 4.9880219
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2909355, 6.2928848
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1385860, 4.1383495
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6346493, 6.6358414
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7137222, 5.7145405
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4066696, 5.4080315
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4159613, 3.4159107
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9152222, 5.9180031
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0244408, 6.0230522
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4892502, 6.4893799
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7701645, 5.7708302
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0034485, 5.0036049
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0256424, 7.0253105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 742

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1695

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6329694, upper bound: 3.6306756
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6300058, upper bound: 3.6336417
time: 9.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3897858, 10.3875923
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2535248, 5.2512989
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2493362, 4.2467461
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3523445, 5.3492775
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0297203, 6.0248566
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1200943, 6.1165543
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4464149, 6.4457893
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8080254, 5.8046455
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2267570, 5.2228107
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0253887, 4.0258980
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2482147, 5.2487087
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9330921, 4.9350586
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2269592, 6.2286911
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2784996, 5.2768478
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2223892, 8.2241478
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8594456, 4.8567486
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4470978, 6.4490166
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1517105, 8.1511612
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3858395, 3.3864212
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6413078, 4.6435528
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9270020, 4.9275589
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2600040, 4.2625675
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3309727, 4.3322563
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7587585, 4.7627583
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1518097, 7.1544533
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1781960, 4.1806183
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5247765, 6.5263138
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0488358, 4.0496616
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1373024, 6.1409531
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9988403, 5.0005188
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9841290, 4.9880314
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2909164, 6.2929039
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1386776, 4.1382542
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6346912, 6.6357994
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7137451, 5.7145176
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4067116, 5.4079857
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4160681, 3.4158049
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9152298, 5.9179955
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0246353, 6.0228577
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4893074, 6.4893188
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7703705, 5.7706261
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0034599, 5.0035973
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0256500, 7.0253105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1572

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1613

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6330608, upper bound: 3.6298674
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6301292, upper bound: 3.6332143
time: 6.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3919144, 10.3869591
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2561588, 5.2514000
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2498894, 4.2510815
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3523865, 5.3540268
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0315323, 6.0271893
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1212921, 6.1181450
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4439240, 6.4505196
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8096008, 5.8056049
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2265167, 5.2269535
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0272884, 4.0237198
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2506828, 5.2546864
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9337425, 4.9340706
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2279549, 6.2321701
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2791519, 5.2789230
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2259789, 8.2298508
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8624153, 4.8580379
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4495430, 6.4462471
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1537094, 8.1581001
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3928452, 3.3851604
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6440430, 4.6433315
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9304657, 4.9290314
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2694035, 4.2624302
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3342934, 4.3314838
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7631779, 4.7626934
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1552887, 7.1547890
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1860180, 4.1813049
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5313644, 6.5268059
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0517693, 4.0490723
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1399651, 6.1414032
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0004005, 4.9989452
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9871063, 4.9875813
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2929955, 6.2930527
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1364307, 4.1440964
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6348057, 6.6396828
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7111950, 5.7170963
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4058266, 5.4100609
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4147682, 3.4168673
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9230080, 5.9194756
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0218391, 6.0240211
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4874191, 6.4916153
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7675533, 5.7741146
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0019493, 5.0060959
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0235405, 7.0281296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 742

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 535

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6126616, upper bound: 3.6323410
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6191103, upper bound: 3.6258921
time: 5.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3920441, 10.3858681
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2564793, 5.2481422
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2501564, 4.2466545
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3524246, 5.3491096
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0316658, 6.0248089
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1213264, 6.1169205
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4443016, 6.4470787
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8096466, 5.8037891
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2265816, 5.2224007
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0273991, 4.0237808
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2513695, 5.2457275
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9338207, 4.9340782
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2284584, 6.2277222
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2791634, 5.2765121
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2265701, 8.2180977
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8624477, 4.8537140
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4496536, 6.4462585
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1540070, 8.1495285
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3867416, 3.3852100
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6399384, 4.6437168
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9241638, 4.9296150
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2601223, 4.2633228
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3312244, 4.3315659
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7578526, 4.7630424
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1514130, 7.1548195
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1766300, 4.1817627
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5247841, 6.5272064
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0494919, 4.0490875
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1349640, 6.1417961
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0004425, 4.9987869
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9835358, 4.9876919
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2894669, 6.2933884
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1368809, 4.1390896
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6315861, 6.6397591
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7120037, 5.7152271
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4050484, 5.4100647
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4143333, 3.4169006
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9147949, 5.9195099
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0218430, 6.0241051
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4863510, 6.4916496
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7676983, 5.7721882
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0023651, 5.0038490
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0236855, 7.0270081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1597

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1761

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6195294, upper bound: 3.6270882
time: 5.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6138531, upper bound: 3.6327605
time: 5.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3841400, 10.3801842
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2572784, 5.2523975
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2501278, 4.2469826
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3559303, 5.3526268
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0398598, 6.0337524
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1253166, 6.1214485
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4549370, 6.4569931
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8142853, 5.8096962
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2293072, 5.2239857
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0304661, 4.0292912
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2434216, 5.2424240
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9251595, 4.9262619
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2283897, 6.2305527
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2828331, 5.2806702
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2343063, 8.2334518
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8709526, 4.8659706
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4438705, 6.4446754
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1557083, 8.1537247
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3869553, 3.3872299
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6401711, 4.6422672
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9200897, 4.9219704
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2600822, 4.2616730
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3303299, 4.3312893
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7597027, 4.7635536
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1531754, 7.1559410
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1765499, 4.1798592
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5243378, 6.5259209
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0458794, 4.0474968
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1299667, 6.1344948
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0003986, 5.0014820
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9763069, 4.9811230
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2946014, 6.2971687
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1434155, 4.1444454
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6436672, 6.6475792
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7188225, 5.7208443
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4105587, 5.4129353
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4178352, 3.4183502
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9189453, 5.9225655
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0218964, 6.0201340
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4942780, 6.4956665
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7796288, 5.7819118
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0062523, 5.0072708
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0312195, 7.0327034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6317618, upper bound: 3.6247037
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6266050, upper bound: 3.6302507
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3839264, 10.3803902
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2573967, 5.2522793
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2498875, 4.2472248
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3556252, 5.3529320
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0393562, 6.0342541
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1252365, 6.1215248
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4558411, 6.4560928
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8143578, 5.8096199
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2279873, 5.2253056
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0305195, 4.0292358
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2444820, 5.2413635
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9265175, 4.9249058
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2298660, 6.2290764
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2827988, 5.2807007
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2365837, 8.2311745
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8712845, 4.8656387
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4455070, 6.4430389
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1564178, 8.1530151
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3878155, 3.3863697
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6401482, 4.6422920
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9194260, 4.9226341
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2604218, 4.2613335
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3305988, 4.3310223
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7596531, 4.7636051
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1531754, 7.1559486
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1764088, 4.1800003
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5242462, 6.5260124
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0472317, 4.0461445
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1298065, 6.1346550
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0019245, 4.9999523
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9777412, 4.9796886
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2946320, 6.2971382
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1434231, 4.1444435
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6431828, 6.6480675
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7188225, 5.7208443
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4098759, 5.4136200
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4173164, 3.4188681
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9189148, 5.9225998
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0196419, 6.0223923
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4931221, 6.4968224
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7796288, 5.7819118
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0062447, 5.0072784
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0311279, 7.0327988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1769

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6231814, upper bound: 3.6395080
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6231515, upper bound: 3.6395378
time: 5.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3815460, 10.3775291
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2549629, 5.2501583
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2514267, 4.2487450
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3532791, 5.3510780
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0355606, 6.0308475
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1238441, 6.1209641
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4509315, 6.4521332
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8154411, 5.8117199
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2300339, 5.2265549
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0306168, 4.0294724
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2403793, 5.2391624
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9324150, 4.9318333
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2237358, 6.2241783
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2805901, 5.2791557
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2334442, 8.2282906
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8713169, 4.8658562
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4465218, 6.4465752
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1644173, 8.1620636
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3892574, 3.3885498
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6406002, 4.6425648
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9175720, 4.9199524
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2632198, 4.2644005
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3326035, 4.3327675
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7574902, 4.7603321
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1510506, 7.1526489
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1744366, 4.1767368
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5253601, 6.5262985
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0492363, 4.0484695
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1268845, 6.1302643
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0026855, 5.0015011
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9819679, 4.9844551
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2952347, 6.2980614
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1408405, 4.1421051
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6461582, 6.6502228
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7192841, 5.7213745
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4132767, 5.4162617
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4191313, 3.4206066
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9211884, 5.9244957
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0298080, 6.0302963
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4920998, 6.4956436
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7775879, 5.7799187
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0063744, 5.0077972
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0317535, 7.0332642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1741

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1441

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6307473, upper bound: 3.6404541
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6305608, upper bound: 3.6406389
time: 5.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3811340, 10.3779488
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2548828, 5.2502403
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2514267, 4.2487431
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3537369, 5.3506203
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0361862, 6.0302200
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1245270, 6.1202812
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4511108, 6.4519501
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8159523, 5.8112011
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2302170, 5.2263699
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0306683, 4.0294209
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2409897, 5.2385559
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9325085, 4.9317398
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2235794, 6.2243347
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2812805, 5.2784653
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2310371, 8.2307014
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8711014, 4.8660717
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4476738, 6.4454193
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1641884, 8.1622925
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3891506, 3.3886566
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6406307, 4.6425323
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9174156, 4.9201050
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2632561, 4.2643623
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3322334, 4.3331356
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7566280, 4.7611961
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1499214, 7.1537819
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1735821, 4.1775913
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5246429, 6.5270157
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0483437, 4.0493603
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1257286, 6.1314201
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0019722, 5.0022182
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9813271, 4.9850960
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2957611, 6.2975349
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1410618, 4.1418781
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6460857, 6.6502953
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7191925, 5.7214699
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4132957, 5.4162426
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4194708, 3.4202652
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9211655, 5.9245186
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0297012, 6.0304031
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4931259, 6.4946213
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7776413, 5.7798615
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0067749, 5.0073967
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0317383, 7.0332870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 534

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1741

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6299178, upper bound: 3.6408153
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6310568, upper bound: 3.6402939
time: 5.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3893967, 10.3887024
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2531090, 5.2536278
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2482185, 4.2461033
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3549728, 5.3525772
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0241661, 6.0204201
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1246376, 6.1199188
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4476662, 6.4421997
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8077011, 5.8055725
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2302380, 5.2262154
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0241241, 4.0279903
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2488537, 5.2510052
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9278698, 4.9283905
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2293015, 6.2327156
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2808266, 5.2799187
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2126884, 8.2222137
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8551807, 4.8557186
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4423141, 6.4457512
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1483650, 8.1450348
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3853168, 3.3875446
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6416702, 4.6379719
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9243088, 4.9210014
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2589455, 4.2559319
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3321686, 4.3297329
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7585201, 4.7581501
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1549187, 7.1556015
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1826992, 4.1796474
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5244064, 6.5253220
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0495510, 4.0510693
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1315613, 6.1310081
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0057774, 5.0034847
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9805870, 4.9837093
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2969551, 6.2926445
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1389713, 4.1367264
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6407757, 6.6358833
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7129250, 5.7165985
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4054165, 5.4065800
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4130669, 3.4135809
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9188728, 5.9193840
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0212402, 6.0217743
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4860344, 6.4806519
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7716618, 5.7699280
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0047951, 5.0030861
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0245056, 7.0190887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1597

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6312484, upper bound: 3.6116999
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6203739, upper bound: 3.6201352
time: 4.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3912659, 10.3874588
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2551613, 5.2520638
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2499123, 4.2447948
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3554230, 5.3508682
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0244598, 6.0173550
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1255379, 6.1191940
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4436188, 6.4425125
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8075943, 5.8059692
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2303600, 5.2249317
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0255871, 4.0256538
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2504482, 5.2498302
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9224548, 4.9301262
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2297249, 6.2325401
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2810555, 5.2773094
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2167664, 8.2188950
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8556290, 4.8510418
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4422493, 6.4458275
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1450386, 8.1463394
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3854198, 3.3875046
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6357574, 4.6419659
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9188271, 4.9265099
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2525425, 4.2575245
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3291035, 4.3321705
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7538052, 4.7584629
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1520653, 7.1559792
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1769886, 4.1813316
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5222244, 6.5276909
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0482445, 4.0514050
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1268959, 6.1351776
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0020504, 5.0039978
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9759331, 4.9851589
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2906456, 6.2930946
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1390171, 4.1367455
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6369648, 6.6365204
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7160797, 5.7126179
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4069176, 5.4055386
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4132481, 3.4119034
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9167290, 5.9195595
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0213966, 6.0197906
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4854622, 6.4808426
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7714367, 5.7708511
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0042229, 5.0031548
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0187378, 7.0203629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 759

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1563

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6313179, upper bound: 3.6057164
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6165699, upper bound: 3.6204643
time: 6.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3890076, 10.3881454
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2545528, 5.2533646
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2492332, 4.2473316
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3543510, 5.3512802
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0236511, 6.0190392
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1238861, 6.1208992
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4433746, 6.4413910
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8077660, 5.8057365
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2282257, 5.2262440
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0261993, 4.0272846
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2489090, 5.2504406
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9208355, 4.9276371
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2313309, 6.2311821
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2809563, 5.2776947
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2150993, 8.2230492
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8548393, 4.8519859
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4412766, 6.4439697
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1442223, 8.1476212
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3853703, 3.3876076
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6373806, 4.6414108
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9244881, 4.9267445
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2523098, 4.2587624
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3314590, 4.3323479
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7511539, 4.7579079
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1513367, 7.1561890
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1779785, 4.1808758
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5241089, 6.5282860
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0487099, 4.0513420
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1301041, 6.1359482
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0004520, 5.0046043
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9766197, 4.9841461
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2902985, 6.2927322
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1381226, 4.1355362
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6362381, 6.6355743
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7150936, 5.7146149
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4070950, 5.4070988
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4132519, 3.4106913
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9169312, 5.9192085
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0219269, 6.0180740
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4844627, 6.4820862
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7711830, 5.7696915
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0046616, 5.0027733
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0208817, 7.0195160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1572

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 742

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6373921, upper bound: 3.6226191
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6399452, upper bound: 3.6200748
time: 4.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3887863, 10.3883667
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2544193, 5.2535000
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2491913, 4.2473755
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3541870, 5.3514481
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0234833, 6.0192089
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1237640, 6.1210251
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4434624, 6.4413033
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8076782, 5.8058243
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2281456, 5.2263241
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0260105, 4.0274734
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2487373, 5.2506084
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9215660, 4.9269047
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2313423, 6.2311668
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2809372, 5.2777176
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2149849, 8.2231674
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8546295, 4.8521957
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4413567, 6.4438896
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1441765, 8.1476707
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3853703, 3.3876076
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6377583, 4.6410313
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9245796, 4.9266491
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2528820, 4.2581921
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3314648, 4.3323421
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7514362, 4.7576237
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1514587, 7.1560631
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1780987, 4.1807556
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5241508, 6.5282440
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0487137, 4.0513382
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1303101, 6.1357460
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0006313, 5.0044250
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9769783, 4.9837875
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2904739, 6.2925529
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1381836, 4.1354694
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6364174, 6.6353951
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7152462, 5.7144585
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4071712, 5.4070244
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4133034, 3.4106388
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9170494, 5.9190941
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0218239, 6.0181770
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4845276, 6.4820213
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7713165, 5.7695580
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0046654, 5.0027733
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0208893, 7.0195084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 833

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1379

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6389966, upper bound: 3.6222101
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6377241, upper bound: 3.6234805
time: 5.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4200592, 10.4096336
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2685261, 5.2618256
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2624264, 4.2550526
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3408279, 5.3335648
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0048714, 5.9901886
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1128578, 6.1023140
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4354630, 6.4361725
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8069649, 5.8000793
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.1963310, 5.1828041
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0126724, 4.0102100
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2457047, 5.2432289
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9052086, 4.9159107
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2235260, 6.2280960
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2658157, 5.2598534
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2183647, 8.2191658
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8392448, 4.8286896
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4443321, 6.4486427
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1422310, 8.1412849
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3672218, 3.3712711
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6368465, 4.6407623
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9288139, 4.9334450
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2410316, 4.2486038
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3280487, 4.3290310
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7336807, 4.7445526
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1325417, 7.1396751
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1675568, 4.1722298
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5413094, 6.5477028
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0395241, 4.0427876
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1320000, 6.1427994
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9906960, 4.9944572
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9417706, 4.9568272
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2910728, 6.2935905
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1342659, 4.1331005
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6192703, 6.6234055
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6846657, 5.6908798
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3782978, 5.3850288
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4102669, 3.4112844
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9013138, 5.9076042
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0195999, 6.0153961
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4823532, 6.4787750
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7658615, 5.7670498
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0005302, 5.0014000
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0232468, 7.0232849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 590

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 850

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6349812, upper bound: 3.6210780
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6374192, upper bound: 3.6186646
time: 5.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4206696, 10.4090157
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2690182, 5.2613316
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2619305, 4.2555504
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3395195, 5.3348732
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0033302, 5.9917297
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1111145, 6.1040573
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4342461, 6.4373894
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8063354, 5.8007050
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.1940842, 5.1850510
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0127754, 4.0101051
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2460938, 5.2428417
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9063015, 4.9148197
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2246857, 6.2269363
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2658119, 5.2598534
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2204094, 8.2171211
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8395596, 4.8283730
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4443588, 6.4486122
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1436729, 8.1398430
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3679543, 3.3705387
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6376896, 4.6399212
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9287224, 4.9335403
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2421055, 4.2475300
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3298378, 4.3272419
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7359123, 4.7423229
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1337242, 7.1385002
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1687183, 4.1710701
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5426331, 6.5463791
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0395756, 4.0427341
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1336365, 6.1411629
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9933968, 4.9917526
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9431667, 4.9554310
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2910423, 6.2936172
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1329994, 4.1343651
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6192856, 6.6233902
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6844826, 5.6910629
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3786106, 5.3847141
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4096012, 3.4119492
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9018135, 5.9071083
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0187531, 6.0162430
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4800568, 6.4810715
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7643356, 5.7685757
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0000954, 5.0018349
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0228806, 7.0236473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 928

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1594

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6296583, upper bound: 3.6233798
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6345790, upper bound: 3.6184534
time: 5.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3902588, 10.3888779
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2568226, 5.2517357
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2468948, 4.2466755
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3481522, 5.3475227
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0259781, 6.0217686
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1174927, 6.1151695
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4455833, 6.4451714
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8054428, 5.7990475
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2188797, 5.2189484
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0260239, 4.0238380
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2518044, 5.2487659
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9351978, 4.9347782
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2281570, 6.2266426
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2770462, 5.2747574
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2198143, 8.2180367
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8595276, 4.8509312
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4499741, 6.4471207
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1437531, 8.1391869
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3861256, 3.3842201
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6383018, 4.6429119
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9243927, 4.9305992
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2587986, 4.2626171
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3281059, 4.3275433
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7570171, 4.7634716
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1500473, 7.1551170
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1759911, 4.1806908
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5239677, 6.5274162
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0482140, 4.0468884
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1346397, 6.1407738
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0004025, 4.9976330
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9814396, 4.9853458
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2884407, 6.2925262
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1387634, 4.1377563
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6299877, 6.6381836
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7063026, 5.7098885
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3985348, 5.4059677
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4108849, 3.4131365
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9106979, 5.9140320
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0161514, 6.0234756
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4842110, 6.4909515
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7705975, 5.7700310
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0016365, 5.0003471
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0244255, 7.0263596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1596

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1572

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6157935, upper bound: 3.6311376
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6110155, upper bound: 3.6359119
time: 4.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3913803, 10.3855362
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2541142, 5.2447777
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2465496, 4.2436409
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3488045, 5.3446579
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0236359, 6.0144062
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1160011, 6.1097298
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4442253, 6.4454041
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8023758, 5.7941933
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2164688, 5.2092075
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0246181, 4.0203552
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2483673, 5.2444401
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9322262, 4.9350471
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2270966, 6.2270050
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2765694, 5.2735710
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2152214, 8.2130280
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8576336, 4.8480778
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4477730, 6.4443130
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1399574, 8.1427002
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3854694, 3.3847370
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6346436, 4.6418190
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9240150, 4.9301224
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2549515, 4.2585640
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3271751, 4.3298168
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7552795, 4.7638111
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1489792, 7.1545982
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1754456, 4.1819744
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5237846, 6.5269928
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0491066, 4.0483532
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1288490, 6.1394691
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9972191, 4.9990540
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9777584, 4.9851303
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2866325, 6.2929497
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1368446, 4.1374531
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6244602, 6.6345901
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6963425, 5.7034054
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3928585, 5.4009895
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4076033, 3.4094124
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9088554, 5.9129448
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0195694, 6.0204239
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4855385, 6.4896507
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7669144, 5.7677746
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9987946, 4.9992790
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0231056, 7.0265160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1664

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 629

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6216027, upper bound: 3.6259013
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6216027, upper bound: 3.6259013
time: 5.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3867798, 10.3901443
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2495365, 5.2493553
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2449245, 4.2452641
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3481941, 5.3452644
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0194168, 6.0186253
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1128807, 6.1128502
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4448318, 6.4447937
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7971230, 5.7994461
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2116585, 5.2140198
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0228539, 4.0221195
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2463074, 5.2465000
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9303036, 4.9369698
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2272682, 6.2268333
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2762375, 5.2739067
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2111549, 8.2170944
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8547745, 4.8509369
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4461823, 6.4459076
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1366997, 8.1459541
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3856125, 3.3845959
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6341476, 4.6423111
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9240379, 4.9300995
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2517166, 4.2617989
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3259697, 4.3310204
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7565498, 4.7625427
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1500473, 7.1535301
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1755180, 4.1819019
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5239716, 6.5268059
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0479069, 4.0495548
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1316948, 6.1366234
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9950447, 5.0012264
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9795132, 4.9833775
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2856789, 6.2939034
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1380768, 4.1362209
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6295185, 6.6295242
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7053223, 5.6944218
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3997784, 5.3940697
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4113951, 3.4056187
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9106369, 5.9111633
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0214577, 6.0185394
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4873772, 6.4878120
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7696075, 5.7650757
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0004196, 4.9976540
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0231209, 7.0265045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1767

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1442

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6338945, upper bound: 3.6276597
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6341732, upper bound: 3.6273818
time: 5.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3898163, 10.3801231
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2567177, 5.2472343
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2486420, 4.2424393
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3501816, 5.3444633
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0294304, 6.0183792
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1185570, 6.1111984
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4419518, 6.4464836
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8059654, 5.7989731
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2218914, 5.2151871
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0247803, 4.0209217
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2527599, 5.2456131
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9319572, 4.9363327
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2273102, 6.2273293
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2765083, 5.2713165
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2226753, 8.2119179
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8616581, 4.8489399
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4458389, 6.4433174
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1494713, 8.1397667
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3851299, 3.3846893
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6384411, 4.6406784
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9258194, 4.9324532
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2596912, 4.2627220
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3317051, 4.3287830
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7576981, 4.7637672
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1498795, 7.1542969
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1766033, 4.1814041
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5223846, 6.5254517
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0465736, 4.0480156
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1359596, 6.1440163
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0017662, 4.9980545
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9783249, 4.9867516
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2905655, 6.2918472
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1361122, 4.1389675
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6282883, 6.6360245
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7030964, 5.7105236
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4003506, 5.4068699
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4090014, 3.4134541
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9092178, 5.9164696
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0202599, 6.0229111
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4883003, 6.4912720
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7647629, 5.7719212
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9995193, 5.0024490
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0224838, 7.0262566

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1763

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1648

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6308696, upper bound: 3.6335613
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6308407, upper bound: 3.6335900
time: 5.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3895874, 10.3803520
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2565804, 5.2473698
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2486000, 4.2424831
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3500137, 5.3446274
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0292625, 6.0185490
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1184311, 6.1113205
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4420395, 6.4463959
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8058777, 5.7990608
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2218113, 5.2152672
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0245914, 4.0211105
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2525921, 5.2457809
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9326878, 4.9356022
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2273216, 6.2273140
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2764893, 5.2713356
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2225609, 8.2120323
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8614483, 4.8491497
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4459190, 6.4432373
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1494217, 8.1398125
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3851299, 3.3846893
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6388226, 4.6402988
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9259148, 4.9323578
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2602615, 4.2621498
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3317089, 4.3287792
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7579842, 4.7634830
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1500015, 7.1541710
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1767254, 4.1812820
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5224266, 6.5254059
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0465755, 4.0480137
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1361656, 6.1438103
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0019455, 4.9978733
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9786835, 4.9863930
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2907448, 6.2916679
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1361771, 4.1389027
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6284637, 6.6358490
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7032490, 5.7103691
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4004269, 5.4067955
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4090528, 3.4134007
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9093361, 5.9163551
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0201607, 6.0230103
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4883652, 6.4912071
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7649002, 5.7717876
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9995232, 5.0024490
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0224915, 7.0262489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1596

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6325327, upper bound: 3.6322070
time: 6.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6273223, upper bound: 3.6374184
time: 5.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3896637, 10.3818130
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2573662, 5.2491989
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2479134, 4.2415142
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3527298, 5.3467598
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0210533, 6.0101452
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1228485, 6.1151505
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4393692, 6.4435387
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8051300, 5.7990742
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2252922, 5.2175102
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0243053, 4.0222569
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2534409, 5.2485199
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9257088, 4.9293022
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2294388, 6.2311440
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2776566, 5.2735176
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2195854, 8.2084389
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8558998, 4.8434715
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4417725, 6.4412842
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1436501, 8.1332207
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3857613, 3.3849506
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6377563, 4.6390362
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9238434, 4.9298096
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2543316, 4.2556400
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3320293, 4.3290958
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7541618, 4.7587109
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1520004, 7.1550598
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1772575, 4.1816120
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5236816, 6.5249710
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0476665, 4.0485401
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1319046, 6.1380882
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0056076, 5.0010128
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9750290, 4.9825191
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2911987, 6.2922173
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1360531, 4.1386795
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6295509, 6.6356697
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7052231, 5.7112865
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4000702, 5.4053822
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4059219, 3.4102898
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9098549, 5.9170570
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0174294, 6.0208893
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4828911, 6.4845772
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7666054, 5.7726078
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9998741, 5.0031776
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0166397, 7.0210609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1112

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 762

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6199538, upper bound: 3.6346855
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6300825, upper bound: 3.6245549
time: 5.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3890076, 10.3824692
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2573738, 5.2491951
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2471313, 4.2422962
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3521309, 5.3473587
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0198708, 6.0113297
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1215591, 6.1164398
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4396820, 6.4431953
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8050842, 5.7991161
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2235985, 5.2192192
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0244331, 4.0221291
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2530823, 5.2488804
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9256420, 4.9293709
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2311401, 6.2294426
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2775154, 5.2736549
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2186546, 8.2093697
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8551693, 4.8442345
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4422150, 6.4408417
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1419220, 8.1349487
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3857365, 3.3849754
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6370392, 4.6397552
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9238472, 4.9298058
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2530880, 4.2568855
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3319016, 4.3292236
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7531013, 4.7597752
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1514053, 7.1556587
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1771984, 4.1816730
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5231628, 6.5254936
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0477657, 4.0484409
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1311417, 6.1388474
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0043335, 5.0022831
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9753342, 4.9822140
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2910690, 6.2923470
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1362362, 4.1384983
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6294327, 6.6357918
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7053871, 5.7111244
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4001656, 5.4052849
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4066448, 3.4095659
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9101105, 5.9168167
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0185814, 6.0197372
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4817047, 6.4857635
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7666626, 5.7725487
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0002480, 5.0028000
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0175018, 7.0202026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 801

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6241280, upper bound: 3.6227589
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6243565, upper bound: 3.6225336
time: 5.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4106674, 10.4012909
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2605019, 5.2556782
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2521210, 4.2480755
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3379250, 5.3359718
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0137177, 6.0074043
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1092911, 6.1046219
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4512062, 6.4554520
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7994041, 5.7969971
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2012177, 5.1946106
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0096798, 4.0046501
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2273750, 5.2210579
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9238758, 4.9283962
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2167816, 6.2177505
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2647476, 5.2604599
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2303505, 8.2247581
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8531551, 4.8473701
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4428787, 6.4418068
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1563339, 8.1541252
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3775635, 3.3810043
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6287041, 4.6288509
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9090271, 4.9125309
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2430439, 4.2458706
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3342171, 4.3335190
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7402267, 4.7430973
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1358871, 7.1387444
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1583538, 4.1600342
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5319633, 6.5363159
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0430908, 4.0454559
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1169052, 6.1215439
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0082092, 5.0081158
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9434814, 4.9516144
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2875557, 6.2885475
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1434002, 4.1443729
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6148224, 6.6209106
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6890411, 5.6951027
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3839836, 5.3894939
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4149055, 3.4170570
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9025154, 5.9054947
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0264702, 6.0258904
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4919090, 6.4918709
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7805367, 5.7853947
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0031357, 5.0043640
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0349236, 7.0369911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1586

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 750

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6257251, upper bound: 3.6290694
time: 9.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6386374, upper bound: 3.6161522
time: 5.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4104309, 10.4015274
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2603607, 5.2558212
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2519913, 4.2482090
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3377953, 5.3361015
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0134773, 6.0076447
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1091423, 6.1047668
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4513245, 6.4553375
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7992744, 5.7971230
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2009354, 5.1948929
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0095959, 4.0047340
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2273369, 5.2210922
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9239597, 4.9283104
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2168846, 6.2176514
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2647400, 5.2604675
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2299461, 8.2251663
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8528671, 4.8476601
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4428253, 6.4418602
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1562805, 8.1541786
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3775539, 3.3810158
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6288986, 4.6286583
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9091034, 4.9124508
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2431850, 4.2457294
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3342686, 4.3334675
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7403831, 4.7429390
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1359024, 7.1387291
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1584511, 4.1599369
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5319748, 6.5363045
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0429935, 4.0455551
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1170235, 6.1214218
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0082054, 5.0081215
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9435349, 4.9515610
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2877464, 6.2883568
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1435680, 4.1442070
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6149979, 6.6207352
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6890755, 5.6950703
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3841019, 5.3893757
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4149761, 3.4169865
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9026642, 5.9053459
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0264816, 6.0258789
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4920044, 6.4917755
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7805939, 5.7853374
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0033455, 5.0041542
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0351372, 7.0367851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 864

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 784

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6352173, upper bound: 3.6299162
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6380065, upper bound: 3.6271271
time: 5.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4069061, 10.4041710
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2571430, 5.2597637
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2491264, 4.2503471
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3356133, 5.3389740
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0086899, 6.0109863
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1073151, 6.1071815
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4530907, 6.4520912
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7966957, 5.7986012
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.1995506, 5.1985149
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0078506, 4.0067062
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2240868, 5.2244835
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9256020, 4.9269791
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2156487, 6.2177277
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2622528, 5.2629776
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2253761, 8.2295113
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8490562, 4.8528957
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4419136, 6.4422722
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1547546, 8.1537361
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3775120, 3.3805923
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6322002, 4.6259518
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9129219, 4.9084854
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2454853, 4.2415428
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3349190, 4.3321972
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7439938, 4.7398243
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1384888, 7.1363525
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1627121, 4.1554127
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5337868, 6.5346794
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0439186, 4.0442543
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1214828, 6.1177826
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0085964, 5.0077953
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9470654, 4.9472809
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2914581, 6.2852669
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1424179, 4.1433849
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6188660, 6.6158028
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6880608, 5.6944599
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3853855, 5.3884487
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4144249, 3.4162693
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9055367, 5.9018631
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0261765, 6.0258293
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4938393, 6.4898262
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7820339, 5.7824879
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0029602, 5.0030212
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0376663, 7.0332794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 731

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1572

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6380306, upper bound: 3.6244828
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6332516, upper bound: 3.6292629
time: 5.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4061584, 10.4049110
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2567921, 5.2601109
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2490578, 4.2504158
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3355751, 5.3390121
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0084572, 6.0112190
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1070900, 6.1074028
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4533920, 6.4517937
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7963219, 5.7989712
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.1990967, 5.1989689
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0076904, 4.0068684
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2240868, 5.2244816
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9254780, 4.9271030
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2162056, 6.2171707
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2624168, 5.2628136
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2248650, 8.2300224
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8489094, 4.8530445
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4418716, 6.4423141
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1542664, 8.1542244
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3772926, 3.3808117
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6319599, 4.6261959
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9125977, 4.9088135
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2447987, 4.2422314
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3344116, 4.3327026
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7437649, 4.7400532
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1380920, 7.1367493
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1621666, 4.1559582
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5332375, 6.5352287
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0436363, 4.0445366
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1213341, 6.1179314
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0082951, 5.0080986
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9467297, 4.9476185
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2912674, 6.2854576
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1432610, 4.1425438
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6189232, 6.6157417
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6886368, 5.6938820
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3856258, 5.3882065
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4147263, 3.4159679
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9055328, 5.9018669
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0263481, 6.0256577
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4940796, 6.4895859
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7826557, 5.7818680
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0035934, 5.0023842
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0376816, 7.0332565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1619

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6324651, upper bound: 3.6216559
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6324651, upper bound: 3.6216559
time: 4.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4066925, 10.4200668
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2644653, 5.2736702
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2531452, 4.2605438
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3419685, 5.3480301
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0010910, 6.0150642
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1107178, 6.1203079
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4506493, 6.4462547
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8018265, 5.8096199
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.1939507, 5.2053719
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0120010, 4.0171604
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2327785, 5.2391090
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9314919, 4.9226570
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2197609, 6.2170410
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2653885, 5.2721252
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2392578, 8.2462883
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8453522, 4.8591881
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4460907, 6.4444695
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1602020, 8.1648598
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3792553, 3.3762417
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6393909, 4.6363010
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9245834, 4.9179192
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2526340, 4.2466583
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3346176, 4.3357143
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7501240, 4.7415218
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1445122, 7.1381721
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1748142, 4.1697884
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5429726, 6.5368767
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0477276, 4.0440807
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1393127, 6.1285019
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0062866, 5.0063934
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9627419, 4.9490604
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2964630, 6.2939568
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1432457, 4.1420021
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6367397, 6.6288223
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6976929, 5.6889915
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3948250, 5.3860207
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4190292, 3.4159889
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9167061, 5.9095001
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0251770, 6.0264168
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.5000572, 6.4991837
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7887421, 5.7838535
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0046349, 5.0018120
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0339546, 7.0313072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 680

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1634

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6290669, upper bound: 3.6345360
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6274806, upper bound: 3.6361238
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4074097, 10.4193420
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2645264, 5.2736092
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2535191, 4.2601738
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3420143, 5.3479805
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0010986, 6.0150566
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1107216, 6.1203003
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4501419, 6.4467621
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8017616, 5.8096848
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.1939583, 5.2053623
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0119781, 4.0171814
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2327480, 5.2391415
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9314499, 4.9226971
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2197151, 6.2170868
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2655182, 5.2719955
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2396812, 8.2458649
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8456593, 4.8588810
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4458733, 6.4446907
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1610298, 8.1640358
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3793278, 3.3761692
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6393833, 4.6363087
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9243965, 4.9181061
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2524471, 4.2468452
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3348408, 4.3354912
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7505131, 4.7411327
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1447334, 7.1379547
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1748161, 4.1697845
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5432167, 6.5366325
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0477142, 4.0440960
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1394920, 6.1283264
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0062180, 5.0064640
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9624100, 4.9493923
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2963562, 6.2940636
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1429863, 4.1422577
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6368122, 6.6287537
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6975937, 5.6890888
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3948860, 5.3859615
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4191303, 3.4158869
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9163780, 5.9098282
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0253525, 6.0262413
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4998779, 6.4993629
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7880478, 5.7845478
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0045319, 5.0019112
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0339394, 7.0313263

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1451

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1575

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6319314, upper bound: 3.6334149
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6319385, upper bound: 3.6334078
time: 5.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3882751, 10.3910446
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2557011, 5.2605515
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2498741, 4.2513103
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3510246, 5.3526154
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0313797, 6.0365772
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1200638, 6.1240883
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4567528, 6.4585228
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8072815, 5.8115730
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2224178, 5.2245045
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0227413, 4.0249443
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2386131, 5.2418537
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9506397, 4.9485340
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2288933, 6.2298279
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2774811, 5.2803459
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2338142, 8.2328606
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8699226, 4.8751030
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4490967, 6.4525108
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1626625, 8.1607513
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3875427, 3.3865356
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6384106, 4.6362343
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9186974, 4.9149437
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2556343, 4.2536945
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3323154, 4.3312836
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7576656, 4.7524300
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1525116, 7.1477928
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1717396, 4.1682205
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5316277, 6.5265312
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0533829, 4.0527649
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1336403, 6.1267090
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0041752, 5.0040665
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9889641, 4.9829826
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2909431, 6.2900810
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1430817, 4.1451950
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6420708, 6.6369438
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7159920, 5.7122078
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4123001, 5.4076977
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4190826, 3.4186440
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9209824, 5.9190521
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0239182, 6.0223389
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4954224, 6.4951706
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7784290, 5.7790527
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0081291, 5.0092545
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0343781, 7.0337219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1569

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6248259, upper bound: 3.6210228
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6247917, upper bound: 3.6210551
time: 6.35 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3876266, 10.3916931
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2551365, 5.2611122
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2485199, 4.2526627
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3494530, 5.3541870
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0300713, 6.0378876
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1188087, 6.1253433
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4567528, 6.4585228
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8064003, 5.8124580
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2211590, 5.2257633
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0225887, 4.0250969
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2375755, 5.2428932
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9508457, 4.9483280
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2291489, 6.2295723
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2766609, 5.2811623
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2332268, 8.2334480
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8691597, 4.8758659
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4491310, 6.4524803
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1613731, 8.1620445
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3880367, 3.3860416
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6394482, 4.6351967
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9195290, 4.9141159
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2572880, 4.2520390
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3326702, 4.3309269
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7588177, 4.7512760
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1532669, 7.1470413
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1733971, 4.1665630
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5323944, 6.5257683
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0540695, 4.0520782
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1347084, 6.1256409
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0042820, 5.0039577
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9896545, 4.9822922
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2922401, 6.2887840
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1429253, 4.1453495
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6423759, 6.6366348
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7146950, 5.7135067
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4123001, 5.4076977
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4190788, 3.4186478
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9224586, 5.9175758
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0237274, 6.0225296
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4954453, 6.4951477
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7784214, 5.7790546
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0080948, 5.0092888
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0344696, 7.0336189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 629

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 835

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6314606, upper bound: 3.6306527
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6327787, upper bound: 3.6293352
time: 5.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3781357, 10.3792343
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2522202, 5.2549801
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2479191, 4.2508659
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3540688, 5.3582993
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0319824, 6.0386391
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1188393, 6.1246414
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4526482, 6.4552765
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8058052, 5.8103828
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2264290, 5.2308540
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0251884, 4.0252934
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2484016, 5.2492943
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9549942, 4.9504528
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2288933, 6.2274208
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2800827, 5.2838554
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2309036, 8.2260246
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8698673, 4.8739643
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4555550, 6.4554443
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1633110, 8.1616249
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3876247, 3.3853436
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6428967, 4.6399136
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9209061, 4.9182396
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2644501, 4.2605782
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3303661, 4.3278732
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7587318, 4.7516594
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1544647, 7.1487389
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1744251, 4.1697998
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5299301, 6.5246124
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0567455, 4.0540199
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1336899, 6.1259613
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0021152, 4.9995117
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9931526, 4.9860725
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2962189, 6.2947845
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1377697, 4.1410847
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6412430, 6.6396484
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7161827, 5.7149696
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4104195, 5.4087372
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4177227, 3.4187546
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9250641, 5.9218483
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0243568, 6.0257187
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4865990, 6.4897957
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7712154, 5.7727184
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0079346, 5.0097237
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0334625, 7.0329399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1569

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 526

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6285542, upper bound: 3.6376594
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6312081, upper bound: 3.6350053
time: 5.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3768311, 10.3805389
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2519608, 5.2552414
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2480488, 4.2507381
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3542976, 5.3580704
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0321274, 6.0384941
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1193695, 6.1241112
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4536552, 6.4542694
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8065758, 5.8096123
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2264671, 5.2308159
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0251961, 4.0252857
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2492027, 5.2484913
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9556255, 4.9498215
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2291679, 6.2271461
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2803383, 5.2835999
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2303429, 8.2265854
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8691235, 4.8747082
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4567413, 6.4542542
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1628990, 8.1620369
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3872242, 3.3857422
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6429539, 4.6398582
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9208832, 4.9182663
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2645054, 4.2605228
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3292294, 4.3290100
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7576180, 4.7527714
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1533890, 7.1498108
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1733723, 4.1708527
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5288696, 6.5256729
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0566311, 4.0541344
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1323509, 6.1273003
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0016193, 5.0000038
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9930534, 4.9861736
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2967072, 6.2942963
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1388035, 4.1400509
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6410675, 6.6398239
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7161789, 5.7149734
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4103622, 5.4087963
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4177742, 3.4187040
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9250450, 5.9218674
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0240936, 6.0259857
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4874687, 6.4889221
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7718182, 5.7721138
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0087166, 5.0089378
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0334625, 7.0329361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 759

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6200739, upper bound: 3.6376516
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6311415, upper bound: 3.6265791
time: 6.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3827744, 10.3876915
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2476673, 5.2550106
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2473602, 4.2509861
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3517685, 5.3564949
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0306587, 6.0385628
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1201248, 6.1236610
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4569054, 6.4520416
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8057022, 5.8105373
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2236500, 5.2283897
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0265980, 4.0302391
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2344742, 5.2405529
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9477119, 4.9432545
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2223167, 6.2231255
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2820320, 5.2858963
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2261620, 8.2398109
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8693333, 4.8813686
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4439430, 6.4460144
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1572952, 8.1617699
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3887138, 3.3904095
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6439323, 4.6355495
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9176407, 4.9088078
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2640095, 4.2565594
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3318558, 4.3303528
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7686939, 4.7604408
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1583481, 7.1537781
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1836205, 4.1751881
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5235672, 6.5211029
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0535088, 4.0533772
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1325798, 6.1236115
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0040226, 5.0041637
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9916306, 4.9858303
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.3017578, 6.2936363
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1419640, 4.1408463
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6489220, 6.6417084
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7155094, 5.7184906
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4119968, 5.4105511
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4198475, 3.4180069
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9252319, 5.9195747
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0284233, 6.0266991
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4981155, 6.4924889
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7808380, 5.7782745
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0070381, 5.0042076
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0379906, 7.0300140

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1570

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6335312, upper bound: 3.6168012
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6280306, upper bound: 3.6219184
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3846359, 10.3864479
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2497196, 5.2534466
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2490540, 4.2496777
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3522148, 5.3547859
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0309563, 6.0354958
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1210175, 6.1229324
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4528580, 6.4523544
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8055954, 5.8109379
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2237759, 5.2271061
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0280571, 4.0279026
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2360764, 5.2393780
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9422970, 4.9449863
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2227440, 6.2229500
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2822609, 5.2832870
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2302475, 8.2364960
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8697777, 4.8766918
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4438782, 6.4460907
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1539688, 8.1630783
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3888187, 3.3903713
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6380196, 4.6395397
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9121552, 4.9143124
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2576065, 4.2581539
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3287907, 4.3327885
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7639790, 4.7607555
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1554871, 7.1541634
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1779099, 4.1768742
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5213852, 6.5234756
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0522022, 4.0537090
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1279144, 6.1277771
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0002956, 5.0046749
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9869766, 4.9872761
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2954483, 6.2940826
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1420097, 4.1408653
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6451111, 6.6423492
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7186642, 5.7145100
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4135017, 5.4095078
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4200268, 3.4163294
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9230843, 5.9197540
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0285873, 6.0247116
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4975433, 6.4926796
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7806129, 5.7791958
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0064697, 5.0042839
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0322304, 7.0312843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1766

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1711

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6338914, upper bound: 3.6193500
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6308084, upper bound: 3.6224350
time: 5.14 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3879089, 10.3909264
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2513332, 5.2566071
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2509499, 4.2517834
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3482437, 5.3496475
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0290222, 6.0326233
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1203575, 6.1217842
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4531670, 6.4523430
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8070450, 5.8112278
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2190285, 5.2210712
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0266171, 4.0277901
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2316437, 5.2357121
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9425755, 4.9441795
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2233124, 6.2245255
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2791595, 5.2796669
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2308884, 8.2410583
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8675823, 4.8741703
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4438210, 6.4462433
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1523018, 8.1593628
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3859749, 3.3885059
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6377487, 4.6364479
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9146767, 4.9120789
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2516880, 4.2526894
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3290787, 4.3315983
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7591152, 4.7569218
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1528778, 7.1519547
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1728535, 4.1718044
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5240250, 6.5246124
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0495052, 4.0513763
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1278267, 6.1246796
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9996891, 5.0041695
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9848728, 4.9844341
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2904434, 6.2899666
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1411610, 4.1396561
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6441917, 6.6410179
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7178497, 5.7155132
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4136562, 5.4110756
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4201288, 3.4163971
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9189873, 5.9169312
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0280724, 6.0237770
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4977417, 6.4925919
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7818432, 5.7790966
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0064240, 5.0041122
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0341644, 7.0320396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1391

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1612

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6395300, upper bound: 3.6260918
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6394602, upper bound: 3.6261729
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3872528, 10.3915825
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2507687, 5.2571678
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2495995, 4.2531357
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3466721, 5.3512192
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0277138, 6.0339336
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1191025, 6.1230431
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4531670, 6.4523430
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8061638, 5.8121128
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2177696, 5.2223301
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0264645, 4.0279427
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2306061, 5.2367516
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9427814, 4.9439735
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2235680, 6.2242699
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2783394, 5.2804832
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2303009, 8.2416458
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8668175, 4.8749352
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4438553, 6.4462128
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1510086, 8.1606522
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3864670, 3.3880119
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6387863, 4.6354103
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9155083, 4.9112473
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2533417, 4.2510338
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3294334, 4.3312435
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7602673, 4.7557678
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1536255, 7.1512032
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1745110, 4.1701469
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5247917, 6.5238457
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0501919, 4.0506916
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1288948, 6.1236153
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9997959, 5.0040607
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9855595, 4.9837456
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2917404, 6.2886696
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1410084, 4.1398106
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6445007, 6.6407089
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7165527, 5.7168121
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4136562, 5.4110756
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4201250, 3.4164009
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9204597, 5.9154549
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0278778, 6.0239716
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4977646, 6.4925690
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7818432, 5.7790985
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0063896, 5.0041466
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0342560, 7.0319405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1619

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1612

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6379416, upper bound: 3.6276795
time: 5.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6378715, upper bound: 3.6277606
time: 5.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3793030, 10.3806419
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2478065, 5.2523651
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2464752, 4.2481880
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3518143, 5.3546410
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0314522, 6.0357723
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1191216, 6.1227951
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4542236, 6.4563713
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8046074, 5.8092003
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2224770, 5.2244740
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0251560, 4.0257607
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2395229, 5.2438908
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9428825, 4.9427910
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2254944, 6.2272873
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2800713, 5.2822037
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2293663, 8.2352715
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8691311, 4.8755054
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4436073, 6.4470177
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1498489, 8.1564293
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3871765, 3.3875446
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6415730, 4.6391735
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9205933, 4.9169083
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2598629, 4.2581635
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3292389, 4.3307056
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7646408, 4.7602463
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1555367, 7.1523781
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1798325, 4.1767502
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5248871, 6.5219421
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0491352, 4.0498428
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1343384, 6.1289291
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9991150, 5.0022697
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9844093, 4.9818401
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2959747, 6.2948494
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1421432, 4.1436920
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6434402, 6.6409035
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7171822, 5.7164612
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4115486, 5.4091911
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4167814, 3.4145470
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9223633, 5.9190445
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0229416, 6.0179520
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4947968, 6.4923744
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7792110, 5.7784977
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0059662, 5.0057106
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0325775, 7.0312881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6348088, upper bound: 3.6242072
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6347793, upper bound: 3.6242367
time: 6.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3808136, 10.3873062
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2480583, 5.2545872
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2475777, 4.2507858
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3509827, 5.3546371
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0297775, 6.0366020
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1186981, 6.1244087
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4526443, 6.4525528
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8035774, 5.8102360
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2228699, 5.2272663
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0247269, 4.0289879
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2359791, 5.2461452
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9445171, 4.9449635
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2237549, 6.2237129
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2790451, 5.2836952
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2294006, 8.2357788
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8692284, 4.8767433
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4399872, 6.4473305
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1547356, 8.1626663
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3897171, 3.3885307
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6412106, 4.6400528
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9194412, 4.9151649
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2603683, 4.2593575
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3312626, 4.3325691
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7650261, 4.7605324
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1562004, 7.1519699
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1799831, 4.1765060
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5251923, 6.5201530
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0531845, 4.0515842
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1347923, 6.1279602
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0008545, 5.0043621
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9893417, 4.9844208
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2957191, 6.2950478
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1416225, 4.1415977
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6451492, 6.6398811
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7187290, 5.7143307
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4139881, 5.4093285
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4189987, 3.4151640
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9229698, 5.9192810
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0279350, 6.0230408
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4958458, 6.4939728
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7809525, 5.7776375
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0055084, 5.0051422
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0328674, 7.0311661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 590

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1572

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6323410, upper bound: 3.6199419
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6275625, upper bound: 3.6247213
time: 6.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3810349, 10.3852921
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2480965, 5.2539349
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2476768, 4.2502308
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3510437, 5.3542671
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0297890, 6.0356388
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1188545, 6.1232910
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4518661, 6.4525986
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8037605, 5.8091030
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2229042, 5.2263374
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0248833, 4.0268917
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2360172, 5.2426395
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9445667, 4.9440765
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2228699, 6.2237892
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2791634, 5.2824173
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2295227, 8.2351456
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8692265, 4.8764439
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4402542, 6.4442673
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1547470, 8.1618805
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3887653, 3.3885918
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6412868, 4.6389694
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9192848, 4.9151688
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2604389, 4.2586346
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3312607, 4.3325653
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7650261, 4.7605381
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1557579, 7.1520195
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1798515, 4.1765137
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5241852, 6.5202560
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0522327, 4.0516357
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1342278, 6.1279716
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0008736, 5.0039692
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9886589, 4.9844494
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2957726, 6.2944031
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1408062, 4.1416283
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6437645, 6.6400299
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7165241, 5.7145367
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4128551, 5.4093933
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4181194, 3.4152508
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9227829, 5.9193230
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0271187, 6.0231171
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4958382, 6.4935074
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7796822, 5.7777061
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0054550, 5.0051651
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0326538, 7.0311928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1695

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 643

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6316535, upper bound: 3.6234972
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6310604, upper bound: 3.6240929
time: 6.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3758698, 10.3825302
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2441788, 5.2504902
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2505798, 4.2548981
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3465614, 5.3498573
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0273514, 6.0352764
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1157036, 6.1207962
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4492416, 6.4480019
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8078117, 5.8128700
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2126808, 5.2178020
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0269165, 4.0294590
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2347851, 5.2387371
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9408913, 4.9374466
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2250862, 6.2242737
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2805824, 5.2847633
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2281799, 8.2270241
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8686047, 4.8752518
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4410248, 6.4414597
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1548195, 8.1538544
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3853760, 3.3822441
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6395702, 4.6364594
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9157677, 4.9106560
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2536411, 4.2492924
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3275585, 4.3260155
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7558918, 4.7499275
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1529541, 7.1483421
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1754417, 4.1702633
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5318069, 6.5271683
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0538883, 4.0517769
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1281395, 6.1202927
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9931850, 4.9923153
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9866314, 4.9799442
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2968559, 6.2936172
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1377487, 4.1376781
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6467285, 6.6413727
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7193699, 5.7164230
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4120483, 5.4069347
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4174519, 3.4175234
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9230690, 5.9187393
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0236473, 6.0258331
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4880295, 6.4874535
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7692184, 5.7687225
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0065536, 5.0074005
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0308189, 7.0300369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 850

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 629

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6204144, upper bound: 3.6285627
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6204144, upper bound: 3.6285627
time: 5.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3758621, 10.3825302
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2441483, 5.2505169
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2506447, 4.2548332
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3465805, 5.3498344
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0273170, 6.0353127
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1157684, 6.1207314
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4492416, 6.4480019
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8077507, 5.8129311
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2127571, 5.2177258
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0269089, 4.0294666
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2346821, 5.2388401
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9406700, 4.9376678
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2249260, 6.2244339
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2805443, 5.2848015
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2279625, 8.2272377
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8686066, 4.8752499
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4408989, 6.4415855
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1545448, 8.1541290
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3851891, 3.3824329
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6395054, 4.6365261
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9158058, 4.9106178
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2534447, 4.2494888
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3272591, 4.3263149
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7557964, 4.7500210
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1529465, 7.1483536
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1754322, 4.1702709
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5317574, 6.5272179
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0538826, 4.0517826
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1280937, 6.1203384
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9929333, 4.9925690
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9866199, 4.9799538
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2968330, 6.2936401
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1378441, 4.1375828
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6467705, 6.6413345
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7193928, 5.7164021
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4120941, 5.4068890
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4175587, 3.4174185
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9230804, 5.9187279
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0238419, 6.0256386
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4880905, 6.4873924
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7694244, 5.7685184
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0065613, 5.0073891
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0308189, 7.0300331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6272393, upper bound: 3.6332502
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6254047, upper bound: 3.6351043
time: 5.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3861389, 10.3845520
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2520828, 5.2495594
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2496452, 4.2510757
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3519630, 5.3526573
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0322533, 6.0357399
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1214142, 6.1223755
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4538879, 6.4528503
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8079681, 5.8078499
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2228260, 5.2216797
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0282440, 4.0256004
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2401180, 5.2316456
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9459343, 4.9416409
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2249184, 6.2214127
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2822495, 5.2827682
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2384872, 8.2286072
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8717155, 4.8704281
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4465866, 6.4417419
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1629562, 8.1543808
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3881321, 3.3894691
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6393433, 4.6386948
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9142685, 4.9145527
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2591934, 4.2581043
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3326263, 4.3305416
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7599277, 4.7598572
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1543655, 7.1540833
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1758118, 4.1765099
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5238876, 6.5241776
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0540638, 4.0528545
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1271744, 6.1270447
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0047855, 5.0002556
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9878693, 4.9849606
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2938957, 6.2937813
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1420403, 4.1416206
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6438732, 6.6443596
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7186394, 5.7175426
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4086933, 5.4112988
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4164314, 3.4198027
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9225769, 5.9213791
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0235977, 6.0277252
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4933395, 6.4963913
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7802429, 5.7811794
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0054588, 5.0062523
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0329819, 7.0325623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 551

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1586

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6270752, upper bound: 3.6314676
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6206713, upper bound: 3.6378726
time: 6.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3859024, 10.3847885
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2519455, 5.2497025
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2495155, 4.2512093
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3518295, 5.3527870
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0320129, 6.0359802
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1212654, 6.1225243
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4540024, 6.4527321
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8078384, 5.8079796
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2225437, 5.2219601
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0281601, 4.0256844
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2400875, 5.2316799
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9460182, 4.9415569
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2250175, 6.2213097
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2822418, 5.2827759
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2380791, 8.2290154
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8714256, 4.8707161
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4465370, 6.4417953
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1629028, 8.1544342
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3881207, 3.3894806
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6395340, 4.6385021
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9143448, 4.9144764
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2593346, 4.2579651
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3326759, 4.3304901
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7600880, 4.7596989
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1543808, 7.1540680
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1759071, 4.1764126
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5238991, 6.5241661
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0539665, 4.0529518
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1272964, 6.1269226
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0047779, 5.0002632
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9879227, 4.9849091
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2940865, 6.2935905
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1422081, 4.1414547
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6440411, 6.6441879
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7186699, 5.7175102
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4088116, 5.4111805
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4165001, 3.4197321
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9227295, 5.9212303
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0236092, 6.0277138
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4934349, 6.4962921
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7803001, 5.7811203
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0056686, 5.0060463
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0331955, 7.0323601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 756

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 864

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6252692, upper bound: 3.6386268
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6263518, upper bound: 3.6375446
time: 10.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3760071, 10.3705406
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2448654, 5.2393379
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2481995, 4.2494240
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3533287, 5.3536606
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0325241, 6.0344906
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1218491, 6.1220245
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4445381, 6.4463005
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8050804, 5.8036613
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2258244, 5.2242050
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0256405, 4.0213852
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2402573, 5.2317772
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9472370, 4.9424324
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2245903, 6.2212524
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2825089, 5.2829857
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2279434, 8.2135506
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8607159, 4.8558235
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4476929, 6.4422188
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1550140, 8.1416702
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3837280, 3.3829460
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6407433, 4.6401711
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9112625, 4.9123764
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2576122, 4.2559414
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3266830, 4.3217735
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7608299, 4.7605820
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1532021, 7.1524582
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1750164, 4.1750298
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5297089, 6.5286980
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0518227, 4.0500851
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1280785, 6.1278114
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9992332, 4.9918766
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9869041, 4.9832573
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2967186, 6.2973824
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1327991, 4.1354141
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6443901, 6.6450424
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7196388, 5.7189770
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4111176, 5.4140167
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4110613, 3.4161901
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9236755, 5.9221497
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0191345, 6.0245361
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4830246, 6.4895287
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7695179, 5.7738800
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9984932, 5.0013084
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0266266, 7.0278740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6210688, upper bound: 3.6360642
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6192358, upper bound: 3.6379158
time: 5.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3725510, 10.3739967
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2420502, 5.2421513
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2481880, 4.2494335
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3530388, 5.3539505
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0309486, 6.0360641
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1213379, 6.1225319
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4474640, 6.4433784
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8046951, 5.8040466
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2254505, 5.2245770
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0244884, 4.0225372
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2402344, 5.2318001
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9475021, 4.9421673
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2246132, 6.2212296
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2824821, 5.2830124
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2240410, 8.2174492
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8568592, 4.8596783
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4477959, 6.4421196
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1509705, 8.1457138
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3815994, 3.3850746
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6410217, 4.6398926
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9121895, 4.9114494
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2572250, 4.2563305
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3238792, 4.3245773
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7608185, 4.7605915
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1527672, 7.1528931
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1745090, 4.1755390
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5284004, 6.5300102
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0514889, 4.0504189
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1280632, 6.1278305
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9966316, 4.9944839
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9866371, 4.9835262
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2977142, 6.2963829
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1356030, 4.1326103
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6442146, 6.6452179
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7198257, 5.7187901
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4107513, 5.4143810
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4125185, 3.4147329
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9233513, 5.9224777
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0200310, 6.0236397
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4862595, 6.4862900
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7728596, 5.7705364
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0007057, 4.9990997
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0285416, 7.0259628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 680

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 757

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6255287, upper bound: 3.6333238
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6263233, upper bound: 3.6325258
time: 4.68 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 12.58 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6200818, upper bound: 3.6287711
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6200532, upper bound: 3.6287997
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6157182, upper bound: 3.6315631
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6148129, upper bound: 3.6324712
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6336534, upper bound: 3.6275380
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6279616, upper bound: 3.6332271
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6331331, upper bound: 3.6331721
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6341506, upper bound: 3.6321546
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6334098, upper bound: 3.6335041
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6337296, upper bound: 3.6331922
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6226309, upper bound: 3.6214182
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6225964, upper bound: 3.6214528
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6329694, upper bound: 3.6306756
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6300058, upper bound: 3.6336417
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6330608, upper bound: 3.6298674
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6301292, upper bound: 3.6332143
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6126616, upper bound: 3.6323410
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6191103, upper bound: 3.6258921
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6195294, upper bound: 3.6270882
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6138531, upper bound: 3.6327605
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6317618, upper bound: 3.6247037
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6266050, upper bound: 3.6302507
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6231814, upper bound: 3.6395080
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6231515, upper bound: 3.6395378
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6307473, upper bound: 3.6404541
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6305608, upper bound: 3.6406389
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6299178, upper bound: 3.6408153
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6310568, upper bound: 3.6402939
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6312484, upper bound: 3.6116999
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6203739, upper bound: 3.6201352
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6313179, upper bound: 3.6057164
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6165699, upper bound: 3.6204643
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6373921, upper bound: 3.6226191
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6399452, upper bound: 3.6200748
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6389966, upper bound: 3.6222101
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6377241, upper bound: 3.6234805
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6349812, upper bound: 3.6210780
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6374192, upper bound: 3.6186646
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6296583, upper bound: 3.6233798
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6345790, upper bound: 3.6184534
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6157935, upper bound: 3.6311376
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6110155, upper bound: 3.6359119
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6216027, upper bound: 3.6259013
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6216027, upper bound: 3.6259013
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6338945, upper bound: 3.6276597
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6341732, upper bound: 3.6273818
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6308696, upper bound: 3.6335613
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6308407, upper bound: 3.6335900
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6325327, upper bound: 3.6322070
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6273223, upper bound: 3.6374184
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6199538, upper bound: 3.6346855
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6300825, upper bound: 3.6245549
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6241280, upper bound: 3.6227589
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6243565, upper bound: 3.6225336
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6257251, upper bound: 3.6290694
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6386374, upper bound: 3.6161522
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6352173, upper bound: 3.6299162
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6380065, upper bound: 3.6271271
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6380306, upper bound: 3.6244828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6332516, upper bound: 3.6292629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6324651, upper bound: 3.6216559
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6324651, upper bound: 3.6216559
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6290669, upper bound: 3.6345360
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6274806, upper bound: 3.6361238
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6319314, upper bound: 3.6334149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6319385, upper bound: 3.6334078
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6248259, upper bound: 3.6210228
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6247917, upper bound: 3.6210551
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6314606, upper bound: 3.6306527
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6327787, upper bound: 3.6293352
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6285542, upper bound: 3.6376594
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6312081, upper bound: 3.6350053
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6200739, upper bound: 3.6376516
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6311415, upper bound: 3.6265791
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6335312, upper bound: 3.6168012
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6280306, upper bound: 3.6219184
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6338914, upper bound: 3.6193500
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6308084, upper bound: 3.6224350
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6395300, upper bound: 3.6260918
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6394602, upper bound: 3.6261729
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6379416, upper bound: 3.6276795
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6378715, upper bound: 3.6277606
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6348088, upper bound: 3.6242072
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6347793, upper bound: 3.6242367
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6323410, upper bound: 3.6199419
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6275625, upper bound: 3.6247213
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6316535, upper bound: 3.6234972
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6310604, upper bound: 3.6240929
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6204144, upper bound: 3.6285627
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6204144, upper bound: 3.6285627
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6272393, upper bound: 3.6332502
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6254047, upper bound: 3.6351043
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6270752, upper bound: 3.6314676
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6206713, upper bound: 3.6378726
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6252692, upper bound: 3.6386268
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6263518, upper bound: 3.6375446
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6210688, upper bound: 3.6360642
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6192358, upper bound: 3.6379158
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6255287, upper bound: 3.6333238
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.58
Output dim: 38, lower bound: -3.6263233, upper bound: 3.6325258

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3838425, 10.3848724
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2538166, 5.2510891
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2502861, 4.2482128
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3521957, 5.3488884
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0314255, 6.0256119
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1228142, 6.1195488
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4505463, 6.4494476
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8094254, 5.8029594
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2224026, 5.2202778
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0274639, 4.0279713
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2379875, 5.2369652
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9295216, 4.9260464
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2231293, 6.2228088
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2810326, 5.2791901
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2372665, 8.2357521
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8685341, 4.8626556
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4399529, 6.4388084
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1567268, 8.1506157
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3887062, 3.3874111
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6389771, 4.6427059
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9149590, 4.9175911
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2608185, 4.2626038
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3313904, 4.3322353
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7573528, 4.7635021
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1512680, 7.1554985
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1744308, 4.1799564
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5221977, 6.5241356
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0502644, 4.0490074
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1282845, 6.1332321
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0038280, 5.0051155
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9820175, 4.9843197
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2937279, 6.2962761
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1411152, 4.1395397
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6389008, 6.6451874
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7101078, 5.7122345
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4034748, 5.4083939
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4166546, 3.4178495
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9151459, 5.9206429
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0229874, 6.0287094
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4930344, 6.4972382
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7793713, 5.7789040
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0043640, 5.0045052
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0284767, 7.0306778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1619

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1725

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6118103, upper bound: 3.6313903
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6138003, upper bound: 3.6297038
time: 5.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3809204, 10.3917313
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2310123, 5.2411194
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2391090, 4.2418804
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3462753, 5.3455315
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0073547, 6.0097961
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1031380, 6.1048737
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4428444, 6.4388161
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7822800, 5.7910042
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.1979351, 5.2026043
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0090809, 4.0136986
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2245979, 5.2269268
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9284649, 4.9354401
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2220650, 6.2222710
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2750854, 5.2737236
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2064247, 8.2193031
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8444901, 4.8483181
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4316368, 6.4381447
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1441727, 8.1512184
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3896961, 3.3927784
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6416092, 4.6429157
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9204445, 4.9218960
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2571278, 4.2595768
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3331566, 4.3346081
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7520962, 4.7534447
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1461945, 7.1472740
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1712284, 4.1721897
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5200310, 6.5215950
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0526409, 4.0592098
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1272964, 6.1268120
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0022335, 5.0071869
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9721870, 4.9724483
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2937737, 6.2933578
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1353168, 4.1287231
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6234398, 6.6092720
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6931572, 5.6794167
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3906994, 5.3774719
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4089165, 3.4033470
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9114380, 5.9072571
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0254402, 6.0238113
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4890556, 6.4773216
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7698975, 5.7638683
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9981079, 4.9940338
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0280876, 7.0280609

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1763

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 756

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6331733, upper bound: 3.6275323
time: 5.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6336477, upper bound: 3.6270527
time: 4.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3904495, 10.3822021
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2390690, 5.2330627
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2423515, 4.2386379
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3475647, 5.3442459
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0141983, 6.0029545
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1079979, 6.1000137
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4402313, 6.4414291
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7894363, 5.7838478
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2047634, 5.1957779
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0136528, 4.0091267
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2281418, 5.2233829
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9290390, 4.9348660
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2208672, 6.2234688
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2761993, 5.2726097
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2133141, 8.2124138
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8505955, 4.8422127
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4347610, 6.4350166
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1473351, 8.1480560
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3897171, 3.3927574
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6414566, 4.6430721
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9198494, 4.9224834
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2578201, 4.2588825
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3337173, 4.3340473
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7491741, 4.7563667
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1441193, 7.1493530
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1703777, 4.1730404
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5191574, 6.5224648
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0540962, 4.0577526
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1223412, 6.1317673
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0042019, 5.0052185
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9685402, 4.9760971
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2935295, 6.2936020
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1322880, 4.1317539
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6143494, 6.6183624
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6831741, 5.6894016
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3819828, 5.3861885
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4049454, 3.4073200
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9082146, 5.9104767
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0244255, 6.0248260
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4847794, 6.4815979
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7653427, 5.7684231
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9957047, 4.9964371
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0270805, 7.0290642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 643

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1569

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6161506, upper bound: 3.6213848
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6161160, upper bound: 3.6214192
time: 4.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3848877, 10.3854332
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2526722, 5.2544327
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2490120, 4.2483425
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3533173, 5.3512039
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0329819, 6.0284195
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1230354, 6.1195602
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4510155, 6.4499016
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8111839, 5.8125057
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2262707, 5.2235050
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0284672, 4.0283642
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2387543, 5.2377453
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9305382, 4.9372559
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2219887, 6.2242699
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2816849, 5.2794456
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2342796, 8.2401924
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8687286, 4.8662987
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4401665, 6.4437561
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1595001, 8.1634483
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3879681, 3.3911839
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6413002, 4.6426525
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9165421, 4.9181843
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2596874, 4.2611465
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3325958, 4.3335800
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7607155, 4.7649460
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1531105, 7.1558876
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1772232, 4.1785049
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5216827, 6.5236740
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0484238, 4.0532646
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1290245, 6.1334419
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0038910, 5.0070839
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9833469, 4.9869385
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2956581, 6.2953339
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1427650, 4.1398506
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6506615, 6.6455650
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7238312, 5.7206097
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4175797, 5.4132214
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4207926, 3.4192848
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9213905, 5.9204178
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0301666, 6.0293388
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.5024719, 6.4951324
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7805138, 5.7792530
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0045242, 5.0034714
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0302277, 7.0312233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1648

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 849

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6263227, upper bound: 3.6263629
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6263227, upper bound: 3.6263629
time: 5.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3841400, 10.3861809
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2523251, 5.2547817
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2489433, 4.2484112
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3532791, 5.3512421
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0327492, 6.0286522
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1228104, 6.1197815
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4513168, 6.4496040
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8108139, 5.8128757
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2258167, 5.2239590
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0283051, 4.0285263
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2387581, 5.2377434
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9304142, 4.9373817
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2225456, 6.2237129
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2818489, 5.2792816
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2337685, 8.2406998
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8685799, 4.8664474
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4401245, 6.4437981
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1590118, 8.1639366
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3877487, 3.3914051
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6410599, 4.6428947
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9162216, 4.9185123
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2589989, 4.2618351
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3320885, 4.3340874
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7604866, 4.7651749
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1527138, 7.1562881
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1766777, 4.1790504
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5211334, 6.5242233
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0481415, 4.0535469
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1288757, 6.1335907
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0035934, 5.0073872
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9830112, 4.9872761
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2954636, 6.2955246
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1436081, 4.1390095
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6507187, 6.6455078
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7244110, 5.7200317
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4178238, 5.4129810
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4210939, 3.4189835
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9213905, 5.9204178
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0303383, 6.0291672
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.5027122, 6.4948921
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7811356, 5.7786331
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0051575, 5.0028343
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0302505, 7.0312004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1648

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1570

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6340236, upper bound: 3.6269256
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6289218, upper bound: 3.6320275
time: 5.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3794861, 10.3743706
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2390099, 5.2317257
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2431965, 4.2383423
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3467979, 5.3420563
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0388260, 6.0297356
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1151886, 6.1087112
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4549637, 6.4532471
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8061371, 5.8001385
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2254581, 5.2256336
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0270615, 4.0270195
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2396011, 5.2380733
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9330730, 4.9333496
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2130470, 6.2107277
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2750282, 5.2702827
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2051620, 8.2146378
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8721085, 4.8651886
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4405937, 6.4383583
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1520195, 8.1521797
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3922291, 3.3902874
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6295414, 4.6344147
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.8964386, 4.9025440
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2542114, 4.2590237
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3438702, 4.3434048
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7513981, 4.7590485
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1438789, 7.1500397
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1626720, 4.1697330
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5176125, 6.5211906
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0457668, 4.0470581
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1078453, 6.1172104
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0024719, 5.0038929
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9643421, 4.9728432
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2875748, 6.2917061
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1400433, 4.1360970
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6290913, 6.6354523
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7168999, 5.7210331
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4074135, 5.4114971
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4189310, 3.4176588
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9097328, 5.9149170
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0290375, 6.0282326
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4884300, 6.4904480
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7683640, 5.7714596
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0002899, 4.9963341
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0310669, 7.0316124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1599

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 629

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6261163, upper bound: 3.6262015
time: 12.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6261163, upper bound: 3.6262015
time: 12.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3797455, 10.3741035
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2386665, 5.2320690
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2431126, 4.2384300
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3466988, 5.3421555
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0385361, 6.0300236
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1148376, 6.1090622
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4546394, 6.4535713
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8059883, 5.8002911
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2309704, 5.2201233
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0289841, 4.0250969
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2425537, 5.2351208
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9322319, 4.9341908
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2104530, 6.2133217
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2745171, 5.2707901
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2139206, 8.2058830
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8717442, 4.8655529
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4403687, 6.4385834
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1512718, 8.1529274
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3887539, 3.3937626
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6297016, 4.6342545
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.8970490, 4.9019375
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2542114, 4.2590237
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3404026, 4.3468685
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7516384, 4.7588100
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1445961, 7.1493187
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1627407, 4.1696644
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5174065, 6.5213928
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0454483, 4.0473785
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1082611, 6.1167946
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0019646, 5.0044003
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9651966, 4.9719925
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2877007, 6.2915840
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1391277, 4.1370125
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6292744, 6.6352692
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7188530, 5.7190800
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4075470, 5.4113655
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4182920, 3.4182987
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9098511, 5.9148026
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0289841, 6.0282860
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4901123, 6.4887619
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7705536, 5.7692680
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9986305, 4.9979935
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0310059, 7.0316696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1664

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1570

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6336025, upper bound: 3.6279634
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6284967, upper bound: 3.6330652
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3828201, 10.3755188
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2337494, 5.2259274
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2565765, 4.2515202
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3450661, 5.3402557
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0432014, 6.0341473
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1131020, 6.1068611
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4491997, 6.4488716
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8134880, 5.8048439
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2108822, 5.2020035
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0186043, 4.0168438
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2382221, 5.2352943
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9372215, 4.9381161
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2264099, 6.2278214
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2784424, 5.2763863
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2210388, 8.2145882
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8481712, 4.8413334
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4414101, 6.4410286
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1585083, 8.1530952
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3871918, 3.3871899
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6318932, 4.6359215
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9191475, 4.9213257
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2598610, 4.2628803
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3296833, 4.3305473
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7486629, 4.7553234
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1457367, 7.1497612
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1642284, 4.1696796
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5268135, 6.5285416
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0516739, 4.0505028
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1212502, 6.1284485
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0046082, 5.0047493
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9787483, 4.9838104
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2806435, 6.2846565
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1423836, 4.1428108
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6330986, 6.6408272
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7191849, 5.7239609
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3864174, 5.3932686
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4195061, 3.4221640
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9191284, 5.9257812
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0284653, 6.0276909
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4864998, 6.4908447
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7766666, 5.7788963
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0082245, 5.0092888
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0295677, 7.0298691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1594

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 784

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6301178, upper bound: 3.6306084
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6329019, upper bound: 3.6278221
time: 5.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3777237, 10.3806229
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2282104, 5.2314701
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2539825, 4.2541161
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3432808, 5.3420410
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0390816, 6.0382671
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1102715, 6.1096916
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4494972, 6.4485703
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8083458, 5.8099861
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2057934, 5.2070885
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0163498, 4.0190983
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2350025, 5.2385159
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9365902, 4.9387474
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2264099, 6.2278214
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2781143, 5.2767143
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2132607, 8.2223663
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8440247, 4.8454762
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4393616, 6.4430771
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1541977, 8.1574097
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3869858, 3.3873978
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6338081, 4.6340065
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9206924, 4.9197807
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2607079, 4.2620335
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3298626, 4.3303661
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7515163, 4.7524719
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1471405, 7.1483574
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1672745, 4.1666336
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5271034, 6.5282478
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0496864, 4.0524883
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1248894, 6.1248055
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0035782, 5.0057812
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9799271, 4.9826336
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2827072, 6.2825928
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1430473, 4.1421471
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6396408, 6.6342888
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7231445, 5.7200012
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3919067, 5.3877811
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4222164, 3.4194546
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9229965, 5.9219093
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0290833, 6.0270767
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4907112, 6.4866295
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7782307, 5.7773342
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0091324, 5.0083809
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0302010, 7.0292320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6223029, upper bound: 3.6259372
time: 7.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6223029, upper bound: 3.6327474
time: 6.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3894577, 10.3871994
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2530136, 5.2507820
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2493458, 4.2466908
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3523102, 5.3491631
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0300293, 6.0248165
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1198578, 6.1157265
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4453392, 6.4443855
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8079910, 5.8045921
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2264900, 5.2224503
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0253544, 4.0258293
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2480240, 5.2483921
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9322681, 4.9349823
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2268028, 6.2285881
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2786102, 5.2761536
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2200127, 8.2244110
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8589725, 4.8563099
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4463959, 6.4480782
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1507835, 8.1507568
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3848686, 3.3864212
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6411324, 4.6434555
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9268303, 4.9276371
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2596207, 4.2627068
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3306313, 4.3321838
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7583733, 4.7629547
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1509018, 7.1544189
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1779175, 4.1805573
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5230370, 6.5259666
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0480843, 4.0496712
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1363564, 6.1406670
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9982224, 5.0005760
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9828053, 4.9880199
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2901840, 6.2923241
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1378593, 4.1368141
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6342506, 6.6354980
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7135048, 5.7145157
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4064827, 5.4079094
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4159203, 3.4151974
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9150429, 5.9178619
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0246162, 6.0227776
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4891853, 6.4879112
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7695427, 5.7697029
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0029602, 5.0024529
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0252800, 7.0247459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1594

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 551

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6327581, upper bound: 3.6269961
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6301893, upper bound: 3.6295651
time: 4.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3897858, 10.3872604
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2535248, 5.2507896
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2492809, 4.2467461
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3522301, 5.3492775
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0296783, 6.0248566
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1192627, 6.1165543
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4450111, 6.4457893
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8079720, 5.8046455
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2267570, 5.2225456
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0253201, 4.0258980
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2478981, 5.2487087
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9330921, 4.9342346
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2268562, 6.2286911
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2778053, 5.2768478
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2223892, 8.2217712
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8594456, 4.8562756
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4461594, 6.4490166
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1517105, 8.1502342
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3858395, 3.3854504
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6412086, 4.6435528
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9270020, 4.9273853
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2600040, 4.2621861
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3309727, 4.3319130
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7587585, 4.7623711
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1518097, 7.1535416
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1781960, 4.1803417
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5247765, 6.5245743
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0488358, 4.0489082
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1373024, 6.1400070
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9988403, 4.9999008
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9841290, 4.9867058
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2903328, 6.2929039
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1372414, 4.1382542
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6343880, 6.6357994
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7137451, 5.7142754
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4067116, 5.4077568
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4154606, 3.4158049
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9150963, 5.9179955
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0246353, 6.0228424
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4878998, 6.4893188
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7694473, 5.7706261
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0023155, 5.0035973
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0250816, 7.0253105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1766

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6294614, upper bound: 3.6306917
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6275932, upper bound: 3.6325427
time: 5.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.4038544, 10.3961906
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2690582, 5.2609997
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2481213, 4.2463245
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3484535, 5.3466263
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0187263, 6.0101624
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1165047, 6.1094093
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4345245, 6.4435539
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7919998, 5.7823963
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2195473, 5.2177811
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0359516, 4.0302086
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2710152, 5.2707539
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9278049, 4.9254074
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2205391, 6.2265930
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2737350, 5.2712135
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2317657, 8.2331810
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8598080, 4.8541813
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4775200, 6.4673233
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1294060, 8.1276131
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3913651, 3.3821430
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6381531, 4.6363544
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9323616, 4.9316063
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2696056, 4.2593861
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3319530, 4.3284245
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7639236, 4.7636833
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1551552, 7.1547241
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1832809, 4.1795616
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5281677, 6.5232048
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0504913, 4.0480385
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1418762, 6.1452217
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0054970, 5.0033073
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9831238, 4.9847622
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2910156, 6.2900505
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1260452, 4.1363869
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6177998, 6.6269493
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6941357, 5.7043762
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3969822, 5.4029751
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4067793, 3.4102058
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9102898, 5.9099121
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0033951, 6.0086899
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4867897, 6.4913445
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7635860, 5.7750740
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9945831, 5.0005569
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0114937, 7.0190926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1757

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 788

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6063647, upper bound: 3.6310061
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6113240, upper bound: 3.6260486
time: 6.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3975067, 10.3818054
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2429924, 5.2265968
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2434502, 4.2367039
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3465958, 5.3419914
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0126991, 5.9989986
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1061859, 6.0969238
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4328384, 6.4382210
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7875443, 5.7745304
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2051449, 5.1941357
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0123711, 4.0041790
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2405605, 5.2313728
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9317646, 4.9314480
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2264938, 6.2269478
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2735672, 5.2698021
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2047768, 8.1894188
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8440418, 4.8292046
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4439621, 6.4374466
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1409264, 8.1332855
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3882141, 3.3866634
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6397762, 4.6437073
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9273605, 4.9334030
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2575378, 4.2600422
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3318138, 4.3315964
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7458858, 4.7539940
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1426582, 7.1481400
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1694298, 4.1754112
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5225067, 6.5257988
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0553017, 4.0534420
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1280785, 6.1398697
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0004826, 4.9968586
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9683285, 4.9761314
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2870865, 6.2912560
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1253319, 4.1305676
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.5950775, 6.6123428
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.6703453, 5.6835537
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3689194, 5.3826542
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.3979664, 3.4045076
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9016113, 5.9095421
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0159073, 6.0191765
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4682388, 6.4778214
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7518005, 5.7608433
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9927483, 4.9966393
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0203400, 7.0246582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 757

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 762

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6032465, upper bound: 3.6322802
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6133744, upper bound: 3.6221591
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3324203, 10.3404198
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2176762, 5.2219868
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2199593, 4.2237606
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3231468, 5.3276176
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -5.9864349, 5.9922009
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.0852432, 6.0907249
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4592743, 6.4551163
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7778397, 5.7817020
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.1965313, 5.1989231
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0061913, 4.0108547
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2196445, 5.2244682
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9292793, 4.9277859
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2147293, 6.2117462
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2599564, 5.2633934
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2090492, 8.2141304
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8273754, 4.8322163
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4169159, 6.4242477
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1186142, 8.1254158
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3911877, 3.3928928
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6296616, 4.6280766
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9037399, 4.9005814
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2515869, 4.2497215
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3342133, 4.3359089
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7421112, 4.7403412
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1478691, 7.1474648
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1618900, 4.1604080
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5191689, 6.5190468
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0455036, 4.0468559
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1072617, 6.1044540
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9999466, 5.0011120
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9597168, 4.9588375
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2862968, 6.2841873
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1382465, 4.1347389
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6164398, 6.6121674
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7066803, 5.7052021
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.3979893, 5.3962688
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4131889, 3.4124184
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.8978577, 5.8954201
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0190277, 6.0170517
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4841232, 6.4829178
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7591972, 5.7541828
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0020332, 4.9999466
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0201492, 7.0151329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1767

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 756

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6312862, upper bound: 3.6246979
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6317560, upper bound: 3.6242202
time: 5.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3765564, 10.3717155
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2543640, 5.2489834
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2497787, 4.2472439
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3550949, 5.3526344
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0380630, 6.0331039
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1232910, 6.1201096
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4516411, 6.4528999
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8133240, 5.8093548
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2276917, 5.2250500
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0304050, 4.0291290
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2478485, 5.2455330
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9282055, 4.9272251
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2293854, 6.2288742
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2822571, 5.2804108
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2325439, 8.2265701
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8684788, 4.8620930
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4496193, 6.4483376
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1540375, 8.1502228
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3850479, 3.3832016
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6405220, 4.6427212
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9195747, 4.9227562
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2617531, 4.2627201
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3268204, 4.3261089
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7558556, 4.7586975
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1525536, 7.1542587
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1731586, 4.1756973
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5236664, 6.5243683
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0469227, 4.0457191
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1264610, 6.1299667
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9980507, 4.9955826
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9782333, 4.9800777
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2945862, 6.2975807
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1371651, 4.1392193
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6427631, 6.6474724
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7188263, 5.7208481
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4095917, 5.4132748
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4166632, 3.4182653
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9188194, 5.9224854
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0208664, 6.0233536
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4855499, 6.4901276
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7726593, 5.7755470
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0058441, 5.0076637
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0309830, 7.0326500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 726

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 801

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6133687, upper bound: 3.6299026
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6135899, upper bound: 3.6296773
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3752594, 10.3730125
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2541046, 5.2492447
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2499084, 4.2471161
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3553238, 5.3524055
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0382080, 6.0329590
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1238213, 6.1195793
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4526482, 6.4518929
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8140945, 5.8085842
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2277298, 5.2250099
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0304127, 4.0291214
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2486534, 5.2447300
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9288368, 4.9265938
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2296600, 6.2285957
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2825089, 5.2801590
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2319832, 8.2271309
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8677368, 4.8628349
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4508095, 6.4471474
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1536255, 8.1506348
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3846474, 3.3836021
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6405754, 4.6426678
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9195480, 4.9227829
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2618084, 4.2626648
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3256836, 4.3272457
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7547493, 4.7598095
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1514854, 7.1553307
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1721077, 4.1767502
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5226059, 6.5254288
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0468082, 4.0458336
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1251183, 6.1313057
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9975586, 4.9960766
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9781342, 4.9801788
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2950745, 6.2970924
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1381950, 4.1381855
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6425877, 6.6476479
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7188263, 5.7208519
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4095306, 5.4133358
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4167166, 3.4182148
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9188004, 5.9225006
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0206032, 6.0236206
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4864235, 6.4892540
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7732620, 5.7749443
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0066299, 5.0068779
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0309830, 7.0326500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1423

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1725

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6201536, upper bound: 3.6384954
time: 6.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6221544, upper bound: 3.6367864
time: 6.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3812408, 10.3775787
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2548256, 5.2502251
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2512283, 4.2486725
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3531380, 5.3510017
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0354424, 6.0307808
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1236191, 6.1209488
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4509087, 6.4520988
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8149834, 5.8118496
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2299194, 5.2267075
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0303307, 4.0294456
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2401657, 5.2387543
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9319496, 4.9315948
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2237625, 6.2240753
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2805443, 5.2790718
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2328148, 8.2280426
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8712749, 4.8658104
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4459801, 6.4462814
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1639595, 8.1619377
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3892555, 3.3885460
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6405888, 4.6425591
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9175072, 4.9199333
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2631989, 4.2644119
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3325348, 4.3327179
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7574825, 4.7603321
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1510506, 7.1526489
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1742058, 4.1766624
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5252151, 6.5262146
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0491676, 4.0486221
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1268425, 6.1302452
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0025749, 5.0015106
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9817314, 4.9844093
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2951202, 6.2980194
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1408958, 4.1419315
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6458225, 6.6497993
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7192955, 5.7212372
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4133511, 5.4159088
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4191380, 3.4204264
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9209328, 5.9242973
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0296707, 6.0300331
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4918213, 6.4953461
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7774715, 5.7797928
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0063629, 5.0077591
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0317383, 7.0332489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1458

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 835

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6271414, upper bound: 3.6381675
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6284590, upper bound: 3.6368503
time: 5.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3815460, 10.3772202
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2549629, 5.2500229
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2514267, 4.2485466
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3532791, 5.3509369
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0355606, 6.0307312
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1238441, 6.1207428
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4509315, 6.4521103
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8154411, 5.8112621
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2300339, 5.2264404
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0306168, 4.0291862
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2403793, 5.2389488
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9324150, 4.9313679
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2236328, 6.2241783
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2805901, 5.2791100
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2334442, 8.2276611
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8713169, 4.8658142
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4465218, 6.4460335
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1644173, 8.1616058
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3892574, 3.3885460
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6406002, 4.6425514
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9175529, 4.9199524
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2632198, 4.2643814
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3325539, 4.3327675
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7574902, 4.7603264
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1510506, 7.1526413
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1743641, 4.1767368
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5252800, 6.5262985
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0492363, 4.0484009
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1268616, 6.1302643
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0026855, 5.0013885
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9819679, 4.9842224
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2951927, 6.2980614
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1406670, 4.1421051
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6457348, 6.6502228
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7191467, 5.7213745
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4129238, 5.4162617
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4189510, 3.4206066
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9209900, 5.9244957
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0295410, 6.0302963
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4918022, 6.4956436
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7774601, 5.7799187
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0063744, 5.0077820
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0317535, 7.0332451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 685

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 750

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6174539, upper bound: 3.6404670
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6303889, upper bound: 3.6275263
time: 5.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3642807, 10.3656654
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2333755, 5.2335949
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2326107, 4.2343159
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3276138, 5.3305626
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -5.9956055, 5.9984112
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.0957718, 6.0979843
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4538765, 6.4538536
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7898521, 5.7911472
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.1987381, 5.2020626
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0207977, 4.0216217
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2316284, 5.2307816
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9212360, 4.9168377
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2150002, 6.2133942
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2678223, 5.2681465
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2293663, 8.2293205
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8499393, 4.8504276
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4417038, 6.4400749
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1627808, 8.1608238
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3868084, 3.3859539
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6264114, 4.6237087
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9102325, 4.9100666
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2490635, 4.2456493
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3263397, 4.3258076
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7357674, 4.7338066
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1385918, 7.1388474
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1574020, 4.1562290
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5173836, 6.5176086
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0451660, 4.0452003
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1049652, 6.1040993
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9918137, 4.9891911
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9660378, 4.9652042
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2821617, 6.2795334
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1415844, 4.1423779
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6298714, 6.6299248
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7188835, 5.7211533
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4056568, 5.4074078
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4197903, 3.4206486
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9065819, 5.9057541
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0278778, 6.0301666
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4923248, 6.4938202
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7784729, 5.7805557
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0060310, 5.0061798
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0337830, 7.0320320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 865

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 692

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6288376, upper bound: 3.6248031
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6139010, upper bound: 3.6397368
time: 5.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3689499, 10.3610878
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2383575, 5.2287312
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2369976, 4.2299252
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3336792, 5.3244972
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0043793, 5.9896374
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1022301, 6.0915298
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4530144, 6.4548111
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7956886, 5.7851009
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2059097, 5.1948929
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0229988, 4.0195503
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2335129, 5.2291985
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9176044, 4.9204693
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2126389, 6.2160606
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2709579, 5.2650108
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2296562, 8.2290306
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8554611, 4.8449116
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4423447, 6.4394493
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1629715, 8.1608849
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3864498, 3.3863144
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6218109, 4.6283092
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9073792, 4.9129848
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2445431, 4.2501698
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3249054, 4.3274174
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7292366, 4.7403374
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1349907, 7.1424980
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1522198, 4.1614113
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5152359, 6.5198784
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0441837, 4.0462513
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.0984077, 6.1106606
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9889450, 4.9921379
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9614334, 4.9699669
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2777596, 6.2839355
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1415653, 4.1424274
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6257133, 6.6342583
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7188759, 5.7211628
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4044628, 5.4088860
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4199181, 3.4205837
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9024048, 5.9099274
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0294800, 6.0285759
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4923286, 6.4938469
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7783394, 5.7808037
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0055618, 5.0066490
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0304871, 7.0353317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1364

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 753

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6254393, upper bound: 3.6401112
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6308717, upper bound: 3.6346723
time: 5.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3962021, 10.3969955
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2530136, 5.2520428
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2518539, 4.2505054
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3540993, 5.3510704
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0148697, 6.0120144
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1252289, 6.1237602
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4396973, 6.4374580
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8069305, 5.8056812
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2176380, 5.2176113
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0240097, 4.0254402
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2487183, 5.2503929
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.8995399, 4.9042740
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2300682, 6.2291641
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2764206, 5.2739487
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2164192, 8.2246513
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8449135, 4.8438301
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4300385, 6.4318848
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1447716, 8.1494026
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3809967, 3.3818665
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6368504, 4.6410332
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9278526, 4.9299240
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2459793, 4.2522240
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3308792, 4.3325748
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7454300, 4.7519112
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1472473, 7.1518135
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1803284, 4.1837463
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5331154, 6.5362053
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0472775, 4.0493355
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1321182, 6.1372299
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9964046, 5.0011368
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9623184, 4.9672432
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2942200, 6.2976456
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1388664, 4.1361084
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6371651, 6.6362419
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7106819, 5.7084446
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4040985, 5.4030647
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4133873, 3.4106035
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9164391, 5.9185715
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0163879, 6.0128098
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4811096, 6.4804459
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7715206, 5.7699718
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0013199, 4.9986038
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0223007, 7.0211258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 571

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 685

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6367887, upper bound: 3.6220350
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6367884, upper bound: 3.6220355
time: 4.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3978577, 10.3953400
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2532310, 5.2518253
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2524109, 4.2499542
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3541412, 5.3510323
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0166245, 6.0102596
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1267471, 6.1222420
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4394417, 6.4377136
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8077087, 5.8048992
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2195950, 5.2156544
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0243530, 4.0250950
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2488594, 5.2502518
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.8974724, 4.9063416
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2293091, 6.2299232
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2772064, 5.2731628
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2166977, 8.2243690
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8466835, 4.8420620
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4291916, 6.4327316
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1460037, 8.1481705
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3796291, 3.3832340
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6370029, 4.6408787
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9276657, 4.9301109
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2457733, 4.2524300
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3316879, 4.3317680
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7451591, 4.7521839
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1469574, 7.1521072
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1808472, 4.1832275
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5320282, 6.5372963
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0467033, 4.0499115
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1313858, 6.1379623
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -4.9969845, 5.0005608
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9597168, 4.9698448
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2952080, 6.2966576
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1386948, 4.1362801
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6369057, 6.6365013
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7089233, 5.7102032
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4030609, 5.4041023
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4131641, 3.4108276
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9162903, 5.9187164
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0166626, 6.0125389
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4828224, 6.4787369
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7714596, 5.7700329
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0004921, 4.9994316
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0224915, 7.0209312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 928

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6388325, upper bound: 3.6199795
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6398500, upper bound: 3.6189631
time: 5.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3887177, 10.3882332
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2541046, 5.2530479
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2491455, 4.2472153
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3548317, 5.3519211
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0234222, 6.0190163
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1240005, 6.1211472
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4435310, 6.4414368
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8078766, 5.8058090
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2286301, 5.2266369
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0260735, 4.0275192
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2489319, 5.2506847
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9203262, 4.9258747
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2312622, 6.2311287
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2809105, 5.2776871
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2150192, 8.2230110
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8546581, 4.8521919
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4413300, 6.4439964
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1442909, 8.1475029
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3853970, 3.3876419
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6377716, 4.6411381
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9245834, 4.9266548
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2529297, 4.2582951
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3315735, 4.3325310
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7515221, 4.7578068
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1515274, 7.1561432
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1782227, 4.1810284
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5241165, 6.5282173
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0483704, 4.0510654
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1303482, 6.1358566
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0005608, 5.0043659
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9763222, 4.9832592
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2906075, 6.2928047
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1382179, 4.1354923
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6366463, 6.6357613
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7150383, 5.7141666
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4073753, 5.4072762
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4133682, 3.4106531
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9173203, 5.9195251
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0217514, 6.0180550
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4845734, 6.4820786
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7714157, 5.7696609
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0046539, 5.0027695
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0209236, 7.0195694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1767

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1596

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6381659, upper bound: 3.6161250
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6329360, upper bound: 3.6213562
time: 4.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3886414, 10.3883018
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2539673, 5.2531872
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2490311, 4.2473259
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3546600, 5.3520927
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0232887, 6.0191498
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1238823, 6.1212616
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4435997, 6.4413719
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.8076630, 5.8060226
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2284584, 5.2268066
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0260563, 4.0275364
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2488174, 5.2508030
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9205360, 4.9256649
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2313042, 6.2310867
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2809067, 5.2776909
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2148285, 8.2232018
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8546238, 4.8522263
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4414635, 6.4438629
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1440086, 8.1477852
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3854046, 3.3876343
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6378670, 4.6410427
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9245872, 4.9266510
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2529831, 4.2582397
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3316517, 4.3324509
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7516174, 4.7577095
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1515427, 7.1561241
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1783714, 4.1808796
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5241280, 6.5282059
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0484390, 4.0509949
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1304245, 6.1357803
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0005722, 5.0043545
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9764481, 4.9831333
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2907257, 6.2926865
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1382103, 4.1355000
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6367760, 6.6356316
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7149544, 5.7142487
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4074211, 5.4072285
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4133167, 3.4107037
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9174805, 5.9193649
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0217018, 6.0181046
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4845886, 6.4820633
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7714195, 5.7696552
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0046577, 5.0027657
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0209541, 7.0195427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2041
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 726

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1442

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6372499, upper bound: 3.6232827
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6375263, upper bound: 3.6230051
time: 5.15 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 12.27 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6118103, upper bound: 3.6313903
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6138003, upper bound: 3.6297038
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6331733, upper bound: 3.6275323
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6336477, upper bound: 3.6270527
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6161506, upper bound: 3.6213848
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6161160, upper bound: 3.6214192
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6263227, upper bound: 3.6263629
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6263227, upper bound: 3.6263629
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6340236, upper bound: 3.6269256
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6289218, upper bound: 3.6320275
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6261163, upper bound: 3.6262015
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6261163, upper bound: 3.6262015
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6336025, upper bound: 3.6279634
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6284967, upper bound: 3.6330652
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6301178, upper bound: 3.6306084
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6329019, upper bound: 3.6278221
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6223029, upper bound: 3.6259372
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6223029, upper bound: 3.6327474
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6327581, upper bound: 3.6269961
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6301893, upper bound: 3.6295651
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6294614, upper bound: 3.6306917
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6275932, upper bound: 3.6325427
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6063647, upper bound: 3.6310061
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6113240, upper bound: 3.6260486
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6032465, upper bound: 3.6322802
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6133744, upper bound: 3.6221591
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6312862, upper bound: 3.6246979
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6317560, upper bound: 3.6242202
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6133687, upper bound: 3.6299026
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6135899, upper bound: 3.6296773
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6201536, upper bound: 3.6384954
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6221544, upper bound: 3.6367864
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6271414, upper bound: 3.6381675
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6284590, upper bound: 3.6368503
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6174539, upper bound: 3.6404670
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6303889, upper bound: 3.6275263
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6288376, upper bound: 3.6248031
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6139010, upper bound: 3.6397368
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6254393, upper bound: 3.6401112
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6308717, upper bound: 3.6346723
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6367887, upper bound: 3.6220350
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6367884, upper bound: 3.6220355
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6388325, upper bound: 3.6199795
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6398500, upper bound: 3.6189631
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6381659, upper bound: 3.6161250
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6329360, upper bound: 3.6213562
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6372499, upper bound: 3.6232827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 12.27
Output dim: 38, lower bound: -3.6375263, upper bound: 3.6230051
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6349812, upper bound: 3.6210780
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6374192, upper bound: 3.6186646
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6345790, upper bound: 3.6184534
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6110155, upper bound: 3.6359119
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6338945, upper bound: 3.6276597
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6341732, upper bound: 3.6273818
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6308696, upper bound: 3.6335613
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6308407, upper bound: 3.6335900
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6325327, upper bound: 3.6322070
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6273223, upper bound: 3.6374184
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6199538, upper bound: 3.6346855
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6386374, upper bound: 3.6161522
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6352173, upper bound: 3.6299162
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6380065, upper bound: 3.6271271
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6380306, upper bound: 3.6244828
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6332516, upper bound: 3.6292629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6324651, upper bound: 3.6216559
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6324651, upper bound: 3.6216559
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6290669, upper bound: 3.6345360
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6274806, upper bound: 3.6361238
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6319314, upper bound: 3.6334149
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6319385, upper bound: 3.6334078
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6327787, upper bound: 3.6293352
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6285542, upper bound: 3.6376594
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6312081, upper bound: 3.6350053
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6200739, upper bound: 3.6376516
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6335312, upper bound: 3.6168012
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6338914, upper bound: 3.6193500
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6395300, upper bound: 3.6260918
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6394602, upper bound: 3.6261729
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6379416, upper bound: 3.6276795
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6378715, upper bound: 3.6277606
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6348088, upper bound: 3.6242072
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6347793, upper bound: 3.6242367
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6323410, upper bound: 3.6199419
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6272393, upper bound: 3.6332502
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6254047, upper bound: 3.6351043
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6206713, upper bound: 3.6378726
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6252692, upper bound: 3.6386268
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6263518, upper bound: 3.6375446
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6210688, upper bound: 3.6360642
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6192358, upper bound: 3.6379158
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6255287, upper bound: 3.6333238
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 12.27
Output dim: 38, lower bound: -3.6263233, upper bound: 3.6325258

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 18.91 + 1788.28 = 1807.20 seconds
