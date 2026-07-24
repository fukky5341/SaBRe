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
execution time: IAR + RelationalAnalysis = 2.30 + 16.70 = 19.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 38, lower bound: -3.6426783, upper bound: 3.6426783

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1711

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6420764, upper bound: 3.6368974
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6368974, upper bound: 3.6420765
time: 6.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.79 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.79
Output dim: 38, lower bound: -3.6420764, upper bound: 3.6368974
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.79
Output dim: 38, lower bound: -3.6368974, upper bound: 3.6420765

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3448181, 10.3567734
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2326927, 5.2418842
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2295456, 4.2364960
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3329926, 5.3407707
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0011139, 6.0129890
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.0969849, 6.1063385
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4597626, 6.4535484
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7877769, 5.7962322
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2067490, 5.2144642
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0093346, 4.0151730
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2275276, 5.2333546
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9615784, 4.9589806
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2115364, 6.2062950
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2700577, 5.2756577
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2209091, 8.2268448
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8502331, 4.8600559
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4177780, 6.4242973
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1380577, 8.1468353
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3971710, 3.3986778
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6404209, 4.6367416
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9112282, 4.9061852
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2635841, 4.2601280
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3368893, 4.3376255
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7620583, 4.7564335
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1627846, 7.1596222
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1768074, 4.1720161
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5294762, 6.5277710
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0582256, 4.0579605
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1263084, 6.1189690
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0053062, 5.0053806
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9927406, 4.9870472
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2961769, 6.2915001
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1438446, 4.1394024
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6341019, 6.6259155
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7155552, 5.7120590
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4149799, 5.4108849
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4201012, 3.4188156
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9133987, 5.9073410
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0274124, 6.0271988
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4950447, 6.4924469
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7623291, 5.7550259
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0038719, 5.0007629
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0228691, 7.0163689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1711

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6329052, upper bound: 3.6276676
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6320047, upper bound: 3.6285919
time: 5.89 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3567734, 10.3448181
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2418861, 5.2326927
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2364960, 4.2295456
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3407707, 5.3329926
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0129890, 6.0011158
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1063385, 6.0969849
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4535484, 6.4597626
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7962341, 5.7877789
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2144661, 5.2067509
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0151730, 4.0093346
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2333565, 5.2275276
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9589806, 4.9615784
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2062950, 6.2115364
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2756577, 5.2700577
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2268448, 8.2209129
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8600559, 4.8502331
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4242973, 6.4177780
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1468353, 8.1380577
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3986778, 3.3971710
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6367397, 4.6404228
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9061890, 4.9112282
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2601280, 4.2635841
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3376255, 4.3368874
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7564354, 4.7620564
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1596260, 7.1627884
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1720161, 4.1768074
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5277710, 6.5294762
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0579605, 4.0582256
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1189690, 6.1263084
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0053825, 5.0053043
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9870491, 4.9927406
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2915001, 6.2961769
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1394005, 4.1438446
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6259155, 6.6341019
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7120571, 5.7155571
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4108829, 5.4149799
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4188156, 3.4201012
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9073410, 5.9133987
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0271988, 6.0274124
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4924469, 6.4950447
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7550278, 5.7623272
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0007629, 5.0038719
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0163689, 7.0228729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1711

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6285919, upper bound: 3.6320047
time: 9.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6276676, upper bound: 3.6329052
time: 6.29 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.83 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.83
Output dim: 38, lower bound: -3.6329052, upper bound: 3.6276676
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.83
Output dim: 38, lower bound: -3.6320047, upper bound: 3.6285919
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.83
Output dim: 38, lower bound: -3.6285919, upper bound: 3.6320047
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.83
Output dim: 38, lower bound: -3.6276676, upper bound: 3.6329052

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3423462, 10.3658409
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2308388, 5.2484970
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2283401, 4.2408981
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3320885, 5.3448715
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -5.9992447, 6.0204048
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.0955963, 6.1121292
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4636269, 6.4527168
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7863503, 5.8019123
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2053947, 5.2201176
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0081825, 4.0196915
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2262955, 5.2376747
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9627838, 4.9589005
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2146378, 6.2056236
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2692642, 5.2791252
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2195549, 8.2313881
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8486080, 4.8666878
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4166412, 6.4282913
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1366348, 8.1516190
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3971157, 3.3990173
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6424026, 4.6365128
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9143524, 4.9053154
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2657528, 4.2599182
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3368378, 4.3378868
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7654419, 4.7558289
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1651878, 7.1594009
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1792812, 4.1712971
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5310822, 6.5273361
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0582218, 4.0579166
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1310081, 6.1176605
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0050545, 5.0055656
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9976578, 4.9865513
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2993393, 6.2912712
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1470661, 4.1386490
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6405602, 6.6242447
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7199135, 5.7109165
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4194527, 5.4097366
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4217463, 3.4183407
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9171028, 5.9063644
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0275269, 6.0270691
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4976387, 6.4917755
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7671089, 5.7537212
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0058784, 5.0002594
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0260468, 7.0154877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1711

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6322264, upper bound: 3.6251104
time: 5.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6304090, upper bound: 3.6270097
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3448181, 10.3542976
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2326927, 5.2400303
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2295456, 4.2352886
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3329926, 5.3398666
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0011139, 6.0111179
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.0969849, 6.1049500
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4589348, 6.4535484
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7877769, 5.7948055
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2067490, 5.2131081
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0093346, 4.0140209
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2275276, 5.2321205
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9614964, 4.9589806
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2108650, 6.2062950
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2700577, 5.2748642
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2209091, 8.2254868
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8502331, 4.8584309
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4177780, 6.4231644
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1380577, 8.1454163
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3971710, 3.3986244
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6401939, 4.6367416
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9103546, 4.9061852
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2633762, 4.2601280
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3368893, 4.3375740
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7614517, 4.7564335
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1625633, 7.1596222
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1760883, 4.1720161
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5290413, 6.5277710
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0582256, 4.0579567
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1249962, 6.1189690
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0053062, 5.0051327
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9922409, 4.9870472
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2959518, 6.2915001
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1430912, 4.1394024
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6324272, 6.6259155
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7144127, 5.7120590
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4138336, 5.4108849
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4196272, 3.4188156
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9124222, 5.9073410
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0272827, 6.0271988
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4943695, 6.4924469
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7610207, 5.7550259
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0033646, 5.0007629
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0219879, 7.0163689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1711

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6313258, upper bound: 3.6260360
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6295138, upper bound: 3.6279344
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3542938, 10.3538857
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2400322, 5.2393036
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2352905, 4.2339478
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3398666, 5.3370934
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0111160, 6.0085316
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1049500, 6.1027756
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4574127, 6.4589348
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7948074, 5.7934589
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2131081, 5.2124004
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0140209, 4.0138531
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2321205, 5.2318497
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9601078, 4.9614964
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2093964, 6.2108650
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2748642, 5.2735252
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2254868, 8.2255936
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8584309, 4.8568630
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4231644, 6.4218140
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1454163, 8.1430550
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3986244, 3.3974476
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6387253, 4.6401939
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9093094, 4.9103584
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2622967, 4.2633762
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3375740, 4.3371506
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7598190, 4.7614517
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1620293, 7.1625633
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1744900, 4.1760883
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5293732, 6.5290413
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0579567, 4.0581894
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1236687, 6.1249962
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0051308, 5.0054569
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9919624, 4.9922428
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2946625, 6.2959518
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1426258, 4.1430931
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6323814, 6.6324310
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7164116, 5.7144146
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4153595, 5.4138336
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4204912, 3.4196272
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9110451, 5.9124222
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0272713, 6.0272827
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4951019, 6.4943695
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7598076, 5.7610226
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0027733, 5.0033646
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0195389, 7.0219917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1711

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6279343, upper bound: 3.6295138
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6260360, upper bound: 3.6313258
time: 5.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3567734, 10.3423424
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2418861, 5.2308388
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2364960, 4.2283401
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3407707, 5.3320885
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0129890, 5.9992428
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1063385, 6.0955963
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4527168, 6.4597626
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7962341, 5.7863522
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2144661, 5.2053928
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0151730, 4.0081806
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2333565, 5.2262955
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9588985, 4.9615784
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2056236, 6.2115364
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2756577, 5.2692642
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2268448, 8.2195549
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8600559, 4.8486080
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4242973, 6.4166412
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1468353, 8.1366348
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3986778, 3.3971157
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6365128, 4.6404228
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9053154, 4.9112282
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2599182, 4.2635841
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3376255, 4.3368378
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7558289, 4.7620564
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1594048, 7.1627884
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1712971, 4.1768074
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5273361, 6.5294762
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0579605, 4.0582218
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1176605, 6.1263084
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0053825, 5.0050564
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9865494, 4.9927406
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2912712, 6.2961769
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1386509, 4.1438446
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6242409, 6.6341019
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7109146, 5.7155571
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4097366, 5.4149799
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4183416, 3.4201012
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9063683, 5.9133987
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0270691, 6.0274124
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4917755, 6.4950447
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7537193, 5.7623272
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0002594, 5.0038719
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0154877, 7.0228729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1711

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1755

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6270097, upper bound: 3.6304090
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6251104, upper bound: 3.6322264
time: 5.46 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 12.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.97
Output dim: 38, lower bound: -3.6322264, upper bound: 3.6251104
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 12.97
Output dim: 38, lower bound: -3.6304090, upper bound: 3.6270097
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 12.97
Output dim: 38, lower bound: -3.6313258, upper bound: 3.6260360
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 12.97
Output dim: 38, lower bound: -3.6295138, upper bound: 3.6279344
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 12.97
Output dim: 38, lower bound: -3.6279343, upper bound: 3.6295138
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 12.97
Output dim: 38, lower bound: -3.6260360, upper bound: 3.6313258
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 12.97
Output dim: 38, lower bound: -3.6270097, upper bound: 3.6304090
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.97
Output dim: 38, lower bound: -3.6251104, upper bound: 3.6322264

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3429489, 10.3663826
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2234650, 5.2423897
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2210770, 4.2347775
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3200035, 5.3354874
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -5.9980392, 6.0193367
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.0889854, 6.1063728
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4648399, 6.4541397
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7713814, 5.7893353
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2064457, 5.2190914
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0095482, 4.0207653
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2285576, 5.2395668
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9449921, 4.9444962
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2202606, 6.2113647
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2688217, 5.2786789
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2195816, 8.2298241
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8453064, 4.8631039
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.3921700, 6.4077797
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1256752, 8.1428680
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3939648, 3.4004192
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6391716, 4.6331863
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9146538, 4.9056358
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2674160, 4.2620735
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3357220, 4.3369122
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7667465, 4.7572556
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1649399, 7.1592560
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1723232, 4.1630650
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5399017, 6.5389366
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0576687, 4.0590763
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1327896, 6.1195831
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0048676, 5.0054455
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -5.0002232, 4.9910164
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2959785, 6.2867470
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1528797, 4.1440353
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6318188, 6.6135941
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7174797, 5.7081947
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4207287, 5.4110718
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4198179, 3.4171276
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9156914, 5.9032974
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0252838, 6.0243835
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4954262, 6.4868813
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7647820, 5.7496262
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0048294, 4.9993401
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0314636, 7.0207062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1711

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 750

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6203836, upper bound: 3.6246848
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6318010, upper bound: 3.6135761
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3573074, 10.3429527
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2357788, 5.2234650
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2303772, 4.2210751
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3313828, 5.3200035
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0119171, 5.9980392
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.1005859, 6.0889854
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4541397, 6.4609718
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7836494, 5.7713833
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2134418, 5.2064476
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0162487, 4.0095501
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2352448, 5.2285557
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9444962, 4.9437866
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2113647, 6.2171593
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2752113, 5.2688217
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2252846, 8.2195816
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8564682, 4.8453064
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4037857, 6.3921700
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1380806, 8.1256752
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.4000778, 3.3939648
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6331863, 4.6371880
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9056358, 4.9115257
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2620735, 4.2652493
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3366508, 4.3357220
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7572556, 4.7633667
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1592560, 7.1625290
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1630650, 4.1698532
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5389366, 6.5382957
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0591202, 4.0576687
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1195831, 6.1280899
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0052567, 5.0048676
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9910145, 4.9953041
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2867470, 6.2928123
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1440372, 4.1496582
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6135998, 6.6253548
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7081947, 5.7131310
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4110737, 5.4162540
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4171286, 3.4181728
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9032974, 5.9119835
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0243874, 6.0251617
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4868813, 6.4928322
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7496262, 5.7599945
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9993401, 5.0028229
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0207062, 7.0282898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1711

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 750

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6135761, upper bound: 3.6318010
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6246848, upper bound: 3.6203837
time: 4.76 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 11.33 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 11.33
Output dim: 38, lower bound: -3.6203836, upper bound: 3.6246848
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.33
Output dim: 38, lower bound: -3.6318010, upper bound: 3.6135761
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.33
Output dim: 38, lower bound: -3.6135761, upper bound: 3.6318010
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 11.33
Output dim: 38, lower bound: -3.6246848, upper bound: 3.6203837

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3331070, 10.3617821
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2170143, 5.2393799
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2173195, 4.2330227
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3171501, 5.3341560
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -5.9911346, 6.0161152
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.0836563, 6.1038857
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4621468, 6.4483681
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7675514, 5.7875462
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2034645, 5.2176991
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0044823, 4.0184002
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2225037, 5.2367420
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9458084, 4.9443741
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2195625, 6.2098732
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2641144, 5.2764816
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2134857, 8.2269783
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8368053, 4.8591347
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.3876534, 6.4056702
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1163712, 8.1385193
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3927021, 3.3998299
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6391563, 4.6331749
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9124680, 4.9009953
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2670937, 4.2618408
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3312187, 4.3348408
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7654877, 4.7551804
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1650085, 7.1585312
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1717587, 4.1623211
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5385246, 6.5367851
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0570393, 4.0588074
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1306877, 6.1151505
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0006924, 5.0034981
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -5.0005970, 4.9886055
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2958794, 6.2865410
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1508884, 4.1397667
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6294842, 6.6086006
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7148285, 5.7025127
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4188423, 5.4070358
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4185133, 3.4143353
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9143562, 5.9004326
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0238495, 6.0213089
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4943771, 6.4846382
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7611122, 5.7417641
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -5.0034981, 4.9964867
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0294266, 7.0163574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1711

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 751

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6242211, upper bound: 3.6130722
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6316188, upper bound: 3.6083804
time: 5.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -21.5377674, -8.4728136, -21.5377674, -8.4728136, -10.3527069, 10.3331108
1: -21.4293251, -12.2464428, -21.4293251, -12.2464428, -5.2327728, 5.2170143
2: -12.3901634, -5.7797737, -12.3901634, -5.7797737, -4.2286186, 4.2173176
3: -12.0021515, -4.1719904, -12.0021515, -4.1719904, -5.3300476, 5.3171501
4: -10.2855673, -0.0076582, -10.2855673, -0.0076582, -6.0086937, 5.9911346
5: -13.5574436, -4.0553980, -13.5574436, -4.0553980, -6.0980988, 6.0836563
6: -8.3219557, 0.5313072, -8.3219557, 0.5313072, -6.4483681, 6.4582787
7: -32.1583405, -22.0876503, -32.1583405, -22.0876503, -5.7818565, 5.7675495
8: -18.8087330, -9.1081448, -18.8087330, -9.1081448, -5.2120514, 5.2034645
9: -5.3156528, 1.3933949, -5.3156528, 1.3933949, -4.0138836, 4.0044823
10: -36.1374550, -27.7813396, -36.1374550, -27.7813396, -5.2324181, 5.2225037
11: -55.1378746, -44.8436050, -55.1378746, -44.8436050, -4.9443741, 4.9446030
12: -11.5788040, -4.6013403, -11.5788040, -4.6013403, -6.2098732, 6.2164612
13: 0.8942345, 8.0055618, 0.8942345, 8.0055618, -5.2730179, 5.2641144
14: -71.0848846, -57.9703903, -71.0848846, -57.9703903, -8.2224388, 8.2134857
15: -8.9140129, 0.8986573, -8.9140129, 0.8986573, -4.8524990, 4.8368053
16: -33.5499344, -23.9971142, -33.5499344, -23.9971142, -6.4016800, 6.3876534
17: -88.6848145, -72.4758148, -88.6848145, -72.4758148, -8.1337357, 8.1163712
18: -4.1732812, 1.0497780, -4.1732812, 1.0497780, -3.3994884, 3.3927021
19: -30.5248871, -23.2230377, -30.5248871, -23.2230377, -4.6331749, 4.6371765
20: -11.1716137, -5.1662931, -11.1716137, -5.1662931, -4.9009933, 4.9093380
21: -43.5484772, -35.0822601, -43.5484772, -35.0822601, -4.2618408, 4.2649097
22: -27.0070724, -19.5644722, -27.0070724, -19.5644722, -4.3345795, 4.3312187
23: -20.8362694, -12.5192890, -20.8362694, -12.5192890, -4.7551804, 4.7620373
24: -16.8516159, -7.6452918, -16.8516159, -7.6452918, -7.1585312, 7.1625366
25: -14.6275959, -6.9724197, -14.6275959, -6.9724197, -4.1623211, 4.1692638
26: -14.6166592, -7.8205671, -14.6166592, -7.8205671, -6.5367851, 6.5368652
27: -14.6325216, -9.5606995, -14.6325216, -9.5606995, -4.0588531, 4.0570393
28: -10.0171633, -1.4245698, -10.0171633, -1.4245698, -6.1151505, 6.1259346
29: -45.5862808, -36.8613014, -45.5862808, -36.8613014, -5.0033092, 5.0006924
30: -32.1891670, -23.0364208, -32.1891670, -23.0364208, -4.9886036, 4.9956799
31: -32.2342300, -23.5404835, -32.2342300, -23.5404835, -6.2865448, 6.2926979
32: 7.7214622, 13.6710949, 7.7214622, 13.6710949, -4.1397648, 4.1476650
33: 4.6460185, 16.3143883, 4.6460185, 16.3143883, -6.6086025, 6.6230202
34: 20.5859108, 30.9924107, 20.5859108, 30.9924107, -5.7025146, 5.7104778
35: 16.5536060, 26.8696270, 16.5536060, 26.8696270, -5.4070358, 5.4143696
36: 28.8363075, 35.1272850, 28.8363075, 35.1272850, -3.4143343, 3.4168701
37: 11.0544949, 20.1148396, 11.0544949, 20.1148396, -5.9004326, 5.9106445
38: 34.9073639, 43.6897011, 34.9073639, 43.6897011, -6.0213127, 6.0237312
39: 9.0230293, 18.5087013, 9.0230293, 18.5087013, -6.4846382, 6.4917870
40: 15.8109369, 25.1263657, 15.8109369, 25.1263657, -5.7417641, 5.7563267
41: 6.7387195, 13.2257462, 6.7387195, 13.2257462, -4.9964905, 5.0014915
42: -12.3792200, -3.4640219, -12.3792200, -3.4640219, -7.0163574, 7.0262604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=81, inp2_unstable=81, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 629
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 731
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 692
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1442
type: RSZ, layer: 1, pos: 1364
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 643
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 685
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 928
type: RSZ, layer: 1, pos: 1112
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 1711

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 751

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6083804, upper bound: 3.6316188
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6130722, upper bound: 3.6242211
time: 5.50 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 12.56 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.56
Output dim: 38, lower bound: -3.6242211, upper bound: 3.6130722
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.56
Output dim: 38, lower bound: -3.6316188, upper bound: 3.6083804
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 12.56
Output dim: 38, lower bound: -3.6083804, upper bound: 3.6316188
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 12.56
Output dim: 38, lower bound: -3.6130722, upper bound: 3.6242211

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 19.00 + 140.67 = 159.66 seconds
