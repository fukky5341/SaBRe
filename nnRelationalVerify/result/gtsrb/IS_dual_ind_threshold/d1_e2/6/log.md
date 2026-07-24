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
execution time: IAR + RelationalAnalysis = 2.33 + 16.62 = 18.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 38, lower bound: -3.6426783, upper bound: 3.6426783

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 692

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6416022, upper bound: 3.6266615
time: 6.36 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6416022, upper bound: 3.6416020
time: 4.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.26 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.26
Output dim: 38, lower bound: -3.6416022, upper bound: 3.6266615
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.26
Output dim: 38, lower bound: -3.6416022, upper bound: 3.6416020

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -21.5362453, -8.4730873, -21.5373707, -8.4728947, -10.3938751, 10.3948593
1: -21.4272423, -12.2468815, -21.4288006, -12.2465734, -5.2701378, 5.2708683
2: -12.3899384, -5.7831583, -12.3901005, -5.7806692, -4.2583046, 4.2556286
3: -12.0020103, -4.1808934, -12.0021124, -4.1745362, -5.3629074, 5.3578262
4: -10.2850008, -0.0126140, -10.2854261, -0.0089840, -6.0520248, 6.0492706
5: -13.5570068, -4.0579147, -13.5573330, -4.0560613, -6.1352997, 6.1336746
6: -8.3137436, 0.5311160, -8.3198214, 0.5312614, -6.4488792, 6.4536514
7: -32.1519623, -22.0882874, -32.1566467, -22.0878277, -5.8180771, 5.8199368
8: -18.8085136, -9.1141634, -18.8086910, -9.1097040, -5.2372589, 5.2344437
9: -5.3148308, 1.3930820, -5.3154268, 1.3933115, -4.0321560, 4.0323067
10: -36.1279373, -27.7823029, -36.1349106, -27.7815971, -5.2427864, 5.2477036
11: -55.1175461, -44.8439598, -55.1324043, -44.8436813, -4.9366531, 4.9510918
12: -11.5726166, -4.6022344, -11.5771065, -4.6016045, -6.2197609, 6.2224846
13: 0.8946202, 7.9990358, 0.8943234, 8.0038366, -5.2900810, 5.2851868
14: -71.0821457, -57.9713974, -71.0841293, -57.9706535, -8.2436676, 8.2433395
15: -8.9133196, 0.8922167, -8.9138165, 0.8969436, -4.8903427, 4.8855762
16: -33.5344238, -23.9973183, -33.5458832, -23.9971714, -6.4333916, 6.4407310
17: -88.6730804, -72.4774933, -88.6817703, -72.4763184, -8.1662331, 8.1705322
18: -4.1699486, 1.0496061, -4.1724153, 1.0497348, -3.3896809, 3.3918800
19: -30.5104866, -23.2230511, -30.5210114, -23.2230301, -4.6380215, 4.6467896
20: -11.1652012, -5.1664972, -11.1698627, -5.1663365, -4.9200211, 4.9238091
21: -43.5301247, -35.0823975, -43.5435371, -35.0823174, -4.2511749, 4.2660828
22: -27.0002861, -19.5646133, -27.0053215, -19.5644836, -4.3274212, 4.3313885
23: -20.8285313, -12.5197620, -20.8341904, -12.5194111, -4.7715759, 4.7762566
24: -16.8464546, -7.6455970, -16.8502007, -7.6453695, -7.1634750, 7.1664963
25: -14.6176910, -6.9727492, -14.6248932, -6.9724846, -4.1809616, 4.1881580
26: -14.6142197, -7.8210859, -14.6160069, -7.8207207, -6.5323677, 6.5335960
27: -14.6225805, -9.5609894, -14.6298504, -9.5607662, -4.0484581, 4.0553684
28: -10.0141392, -1.4248854, -10.0163612, -1.4246401, -6.1447868, 6.1457405
29: -45.5726738, -36.8614960, -45.5827179, -36.8613739, -4.9949398, 5.0019932
30: -32.1776314, -23.0372295, -32.1860199, -23.0366268, -4.9988804, 5.0050850
31: -32.2171974, -23.5405636, -32.2294807, -23.5404778, -6.2878304, 6.2992897
32: 7.7219276, 13.6703815, 7.7215834, 13.6709051, -4.1474514, 4.1473923
33: 4.6478882, 16.3109760, 4.6465216, 16.3135052, -6.6582108, 6.6577415
34: 20.5865078, 30.9733810, 20.5860481, 30.9873695, -5.7203178, 5.7104359
35: 16.5546913, 26.8561783, 16.5539036, 26.8660488, -5.4220562, 5.4141598
36: 28.8370438, 35.1180954, 28.8365173, 35.1248741, -3.4209414, 3.4165611
37: 11.0585785, 20.1146832, 11.0556278, 20.1147842, -5.9295845, 5.9326363
38: 34.9086418, 43.6751633, 34.9077110, 43.6858826, -6.0247993, 6.0190163
39: 9.0247860, 18.5081978, 9.0234776, 18.5085526, -6.5022926, 6.5034103
40: 15.8130198, 25.1246185, 15.8114777, 25.1258888, -5.7810135, 5.7818565
41: 6.7415347, 13.2255993, 6.7394676, 13.2256880, -5.0056915, 5.0073624
42: -12.3702555, -3.4643779, -12.3767757, -3.4640806, -7.0283241, 7.0326424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=81, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1563

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6263223, upper bound: 3.6261630
time: 6.09 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6410987, upper bound: 3.6261585
time: 34.01 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -21.5407333, -8.4723320, -21.5375271, -8.4728241, -10.3980408, 10.3953514
1: -21.4305763, -12.2434044, -21.4291439, -12.2464867, -5.2723541, 5.2725277
2: -12.3953476, -5.7811899, -12.3901138, -5.7803936, -4.2660217, 4.2572517
3: -12.0145264, -4.1714897, -12.0021267, -4.1723537, -5.3746262, 5.3639336
4: -10.2939434, -0.0095351, -10.2855339, -0.0084162, -6.0608864, 6.0522709
5: -13.5592966, -4.0547123, -13.5573864, -4.0559902, -6.1382904, 6.1370163
6: -8.3191786, 0.5404682, -8.3205776, 0.5313147, -6.4546013, 6.4614944
7: -32.1582031, -22.0800762, -32.1580849, -22.0876808, -5.8247299, 5.8254604
8: -18.8164635, -9.1086359, -18.8087292, -9.1084433, -5.2426243, 5.2378693
9: -5.3226748, 1.3954701, -5.3155928, 1.3933557, -4.0439968, 4.0331573
10: -36.1369514, -27.7685318, -36.1371918, -27.7813911, -5.2487793, 5.2607822
11: -55.1369247, -44.8084030, -55.1371689, -44.8435974, -4.9491730, 4.9943695
12: -11.5798349, -4.5929899, -11.5785646, -4.6014209, -6.2248802, 6.2275391
13: 0.8848710, 8.0036268, 0.8942598, 8.0045261, -5.3023415, 5.2896423
14: -71.0849304, -57.9686584, -71.0844727, -57.9704666, -8.2478104, 8.2440681
15: -8.9204702, 0.8946304, -8.9139652, 0.8971896, -4.9025726, 4.8882275
16: -33.5544281, -23.9746609, -33.5492401, -23.9971352, -6.4511299, 6.4623833
17: -88.6840668, -72.4589157, -88.6845551, -72.4759674, -8.1740036, 8.1861725
18: -4.1728926, 1.0546954, -4.1729641, 1.0497813, -3.3923264, 3.3966446
19: -30.5242863, -23.2043133, -30.5243587, -23.2230473, -4.6472893, 4.6664791
20: -11.1713552, -5.1568174, -11.1712885, -5.1663280, -4.9246788, 4.9336624
21: -43.5474205, -35.0555954, -43.5477028, -35.0822754, -4.2629223, 4.3007870
22: -27.0068588, -19.5543633, -27.0068951, -19.5644760, -4.3317852, 4.3401661
23: -20.8355408, -12.5057421, -20.8357201, -12.5193310, -4.7768669, 4.7917652
24: -16.8502693, -7.6357203, -16.8508606, -7.6452870, -7.1661644, 7.1735077
25: -14.6270123, -6.9563551, -14.6272697, -6.9724350, -4.1873894, 4.2081394
26: -14.6163912, -7.8174291, -14.6165113, -7.8206596, -6.5337563, 6.5370407
27: -14.6325092, -9.5454493, -14.6322346, -9.5607128, -4.0552788, 4.0725117
28: -10.0171099, -1.4219282, -10.0169201, -1.4246231, -6.1487312, 6.1479340
29: -45.5855179, -36.8383179, -45.5859222, -36.8613205, -5.0025082, 5.0262756
30: -32.1886292, -23.0156593, -32.1887779, -23.0365047, -5.0060539, 5.0284386
31: -32.2324600, -23.5187321, -32.2328262, -23.5404930, -6.2986259, 6.3251419
32: 7.7183433, 13.6709423, 7.7215071, 13.6709776, -4.1506462, 4.1482277
33: 4.6362534, 16.3142204, 4.6461124, 16.3141079, -6.6706676, 6.6618271
34: 20.5590897, 30.9919510, 20.5859451, 30.9918079, -5.7489319, 5.7243347
35: 16.5322647, 26.8688011, 16.5536537, 26.8692932, -5.4480648, 5.4237671
36: 28.8233261, 35.1266632, 28.8363380, 35.1270523, -3.4335670, 3.4225874
37: 11.0520487, 20.1185684, 11.0555639, 20.1148052, -5.9379463, 5.9383621
38: 34.8874512, 43.6904297, 34.9074249, 43.6893349, -6.0403328, 6.0310478
39: 9.0154705, 18.5090408, 9.0231113, 18.5086517, -6.5106430, 6.5066299
40: 15.8019543, 25.1264172, 15.8110933, 25.1262932, -5.7907734, 5.7828560
41: 6.7385178, 13.2285461, 6.7393556, 13.2257242, -5.0088234, 5.0097961
42: -12.3784533, -3.4515028, -12.3788548, -3.4640305, -7.0339127, 7.0424309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=81, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1563

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6263223, upper bound: 3.6411033
time: 5.83 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6410987, upper bound: 3.6410988
time: 4.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 12.57 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 12.57
Output dim: 38, lower bound: -3.6263223, upper bound: 3.6261630
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 12.57
Output dim: 38, lower bound: -3.6410987, upper bound: 3.6261585
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 12.57
Output dim: 38, lower bound: -3.6263223, upper bound: 3.6411033
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 12.57
Output dim: 38, lower bound: -3.6410987, upper bound: 3.6410988

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -21.5360279, -8.4735260, -21.5645885, -8.4734230, -10.3927231, 10.4233665
1: -21.4263191, -12.2469101, -21.4286690, -12.2388763, -5.2667732, 5.2820282
2: -12.3894405, -5.7831979, -12.3906240, -5.7756348, -4.2629585, 4.2562466
3: -12.0015144, -4.1810007, -12.0018864, -4.1683183, -5.3691139, 5.3589859
4: -10.2844906, -0.0127865, -10.2866249, 0.0079007, -6.0696945, 6.0498581
5: -13.5563669, -4.0579987, -13.5580654, -4.0432940, -6.1477966, 6.1360283
6: -8.3134003, 0.5309465, -8.3215227, 0.5359011, -6.4430847, 6.4695930
7: -32.1512642, -22.0882988, -32.1560669, -22.0615101, -5.8441696, 5.8196220
8: -18.8084888, -9.1143694, -18.8106499, -9.1046562, -5.2431393, 5.2363052
9: -5.3147445, 1.3926970, -5.3288617, 1.3931165, -4.0322094, 4.0472908
10: -36.1274605, -27.7823601, -36.1373520, -27.7762527, -5.2404289, 5.2600040
11: -55.1161957, -44.8439445, -55.1301346, -44.8231277, -4.9558926, 4.9526672
12: -11.5720997, -4.6024156, -11.5769987, -4.5940781, -6.2140350, 6.2394714
13: 0.8952199, 7.9989557, 0.8922700, 8.0159407, -5.3035316, 5.2869415
14: -71.0812225, -57.9715118, -71.0857391, -57.9588242, -8.2518463, 8.2492180
15: -8.9126654, 0.8921175, -8.9147034, 0.9060998, -4.9001770, 4.8887634
16: -33.5335236, -23.9973373, -33.5509872, -23.9879074, -6.4337921, 6.4577942
17: -88.6718750, -72.4776382, -88.6794052, -72.4325943, -8.2138710, 8.1703758
18: -4.1694098, 1.0495605, -4.1790562, 1.0597272, -3.3980999, 3.4002037
19: -30.5100327, -23.2230854, -30.5239182, -23.2185936, -4.6437359, 4.6484146
20: -11.1650543, -5.1665401, -11.1715288, -5.1627913, -4.9283257, 4.9224014
21: -43.5296974, -35.0824203, -43.5452118, -35.0770683, -4.2506561, 4.2761574
22: -26.9999352, -19.5646744, -27.0071106, -19.5488358, -4.3421021, 4.3339844
23: -20.8284569, -12.5203323, -20.8527641, -12.5176268, -4.7743416, 4.7934723
24: -16.8463707, -7.6462631, -16.8624363, -7.6447396, -7.1649170, 7.1767769
25: -14.6176500, -6.9732270, -14.6373005, -6.9715929, -4.1829700, 4.1998863
26: -14.6141109, -7.8215184, -14.6225338, -7.8181152, -6.5353699, 6.5398788
27: -14.6221752, -9.5610523, -14.6310549, -9.5450382, -4.0609360, 4.0603523
28: -10.0140724, -1.4255586, -10.0239811, -1.4249240, -6.1565704, 6.1420517
29: -45.5719299, -36.8615456, -45.5832939, -36.8436584, -5.0044651, 5.0083561
30: -32.1771393, -23.0372658, -32.1861382, -23.0268688, -5.0077724, 5.0067234
31: -32.2165985, -23.5406380, -32.2362595, -23.5324364, -6.2952881, 6.3065987
32: 7.7224150, 13.6702366, 7.7203832, 13.6763134, -4.1491356, 4.1539478
33: 4.6479664, 16.3103352, 4.6171665, 16.3123970, -6.6576195, 6.6885529
34: 20.5865822, 30.9725628, 20.5598106, 30.9860897, -5.7197952, 5.7368584
35: 16.5547523, 26.8554230, 16.5233212, 26.8640594, -5.4215183, 5.4451752
36: 28.8371105, 35.1175461, 28.8212280, 35.1244698, -3.4214230, 3.4330883
37: 11.0586977, 20.1140556, 11.0258045, 20.1134682, -5.9286537, 5.9630470
38: 34.9087639, 43.6742325, 34.8737221, 43.6853714, -6.0257797, 6.0544281
39: 9.0248833, 18.5074577, 8.9954662, 18.5077820, -6.5022507, 6.5322876
40: 15.8130999, 25.1240997, 15.7983818, 25.1273079, -5.7829742, 5.7952099
41: 6.7416887, 13.2249098, 6.7318435, 13.2250175, -5.0035324, 5.0191154
42: -12.3701401, -3.4655957, -12.3862514, -3.4660602, -7.0256691, 7.0497551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1769

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6397618, upper bound: 3.6136447
time: 5.15 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6397618, upper bound: 3.6248171
time: 5.08 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -21.5407333, -8.4723320, -21.5349827, -8.4775658, -10.3923111, 10.3938293
1: -21.4305763, -12.2434044, -21.4261208, -12.2465963, -5.2705822, 5.2716198
2: -12.3953476, -5.7811899, -12.3881674, -5.7807775, -4.2655487, 4.2552929
3: -12.0145264, -4.1714897, -11.9995918, -4.1730924, -5.3739166, 5.3617363
4: -10.2939434, -0.0095351, -10.2798367, -0.0099595, -6.0603409, 6.0463562
5: -13.5592966, -4.0547123, -13.5535526, -4.0564847, -6.1377869, 6.1335144
6: -8.3191786, 0.5404682, -8.3182087, 0.5296122, -6.4487267, 6.4596558
7: -32.1582031, -22.0800762, -32.1498985, -22.0879230, -5.8244781, 5.8170719
8: -18.8164635, -9.1086359, -18.8079491, -9.1093864, -5.2419815, 5.2369862
9: -5.3226748, 1.3954701, -5.3148642, 1.3916485, -4.0420246, 4.0326614
10: -36.1369514, -27.7685318, -36.1357231, -27.7821636, -5.2457047, 5.2598648
11: -55.1369247, -44.8084030, -55.1306686, -44.8439865, -4.9488201, 4.9891415
12: -11.5798349, -4.5929899, -11.5768871, -4.6032119, -6.2199402, 6.2262993
13: 0.8848710, 8.0036268, 0.8962187, 8.0038033, -5.3017349, 5.2867889
14: -71.0849304, -57.9686584, -71.0799255, -57.9710464, -8.2471008, 8.2399254
15: -8.9204702, 0.8946304, -8.9102926, 0.8964529, -4.9018784, 4.8851223
16: -33.5544281, -23.9746609, -33.5447197, -23.9972992, -6.4494896, 6.4604301
17: -88.6840668, -72.4589157, -88.6705627, -72.4775391, -8.1726913, 8.1717529
18: -4.1728926, 1.0546954, -4.1690412, 1.0493577, -3.3919888, 3.3935184
19: -30.5242863, -23.2043133, -30.5219345, -23.2235966, -4.6468906, 4.6636868
20: -11.1713552, -5.1568174, -11.1696882, -5.1669607, -4.9242096, 4.9304352
21: -43.5474205, -35.0555954, -43.5444031, -35.0823631, -4.2622108, 4.2991772
22: -27.0068588, -19.5543633, -27.0017796, -19.5649948, -4.3314419, 4.3359337
23: -20.8355408, -12.5057421, -20.8346825, -12.5233192, -4.7732086, 4.7907486
24: -16.8502693, -7.6357203, -16.8499680, -7.6488848, -7.1623611, 7.1722221
25: -14.6270123, -6.9563551, -14.6267328, -6.9748955, -4.1851215, 4.2076187
26: -14.6163912, -7.8174291, -14.6154070, -7.8226523, -6.5318871, 6.5356178
27: -14.6325092, -9.5454493, -14.6275692, -9.5612411, -4.0545807, 4.0695801
28: -10.0171099, -1.4219282, -10.0160770, -1.4263124, -6.1473312, 6.1436729
29: -45.5855179, -36.8383179, -45.5801239, -36.8618622, -5.0018158, 5.0234509
30: -32.1886292, -23.0156593, -32.1863022, -23.0372448, -5.0054417, 5.0263958
31: -32.2324600, -23.5187321, -32.2299309, -23.5412140, -6.2977715, 6.3223724
32: 7.7183433, 13.6709423, 7.7230186, 13.6694431, -4.1482849, 4.1468925
33: 4.6362534, 16.3142204, 4.6471705, 16.3064880, -6.6629086, 6.6611023
34: 20.5590897, 30.9919510, 20.5869045, 30.9821568, -5.7391834, 5.7236404
35: 16.5322647, 26.8688011, 16.5542526, 26.8604088, -5.4392090, 5.4232330
36: 28.8233261, 35.1266632, 28.8369675, 35.1226082, -3.4294806, 3.4220839
37: 11.0520487, 20.1185684, 11.0567074, 20.1072102, -5.9302635, 5.9374313
38: 34.8874512, 43.6904297, 34.9085999, 43.6785736, -6.0296974, 6.0302734
39: 9.0154705, 18.5090408, 9.0240307, 18.5003319, -6.5025482, 6.5056458
40: 15.8019543, 25.1264172, 15.8120565, 25.1235428, -5.7884941, 5.7822514
41: 6.7385178, 13.2285461, 6.7406673, 13.2214622, -5.0032234, 5.0086937
42: -12.3784533, -3.4515028, -12.3778133, -3.4690402, -7.0266151, 7.0418587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1563

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6263223, upper bound: 3.6263221
time: 5.83 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6263223, upper bound: 3.6410988
time: 7.75 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -21.5404930, -8.4728003, -21.5647488, -8.4733658, -10.3968964, 10.4238586
1: -21.4295979, -12.2434235, -21.4290276, -12.2388210, -5.2689800, 5.2836876
2: -12.3948727, -5.7812133, -12.3906364, -5.7753696, -4.2706738, 4.2578754
3: -12.0140553, -4.1715803, -12.0019226, -4.1661272, -5.3808365, 5.3650780
4: -10.2934523, -0.0096688, -10.2867079, 0.0084705, -6.0785522, 6.0528526
5: -13.5586195, -4.0547500, -13.5581341, -4.0431757, -6.1507835, 6.1393738
6: -8.3188305, 0.5403216, -8.3223019, 0.5359385, -6.4488182, 6.4774513
7: -32.1575317, -22.0801163, -32.1575089, -22.0613937, -5.8508263, 5.8251534
8: -18.8163643, -9.1088305, -18.8107033, -9.1033764, -5.2485180, 5.2397461
9: -5.3225889, 1.3951170, -5.3290138, 1.3931916, -4.0440464, 4.0481205
10: -36.1364441, -27.7686081, -36.1396141, -27.7760410, -5.2464447, 5.2730770
11: -55.1355553, -44.8084450, -55.1349182, -44.8230057, -4.9684105, 4.9959316
12: -11.5793304, -4.5931444, -11.5784512, -4.5938892, -6.2191925, 6.2445068
13: 0.8854943, 8.0035753, 0.8922254, 8.0166225, -5.3158035, 5.2914124
14: -71.0839539, -57.9687653, -71.0861053, -57.9586105, -8.2559853, 8.2499352
15: -8.9198265, 0.8945312, -8.9148293, 0.9063320, -4.9124050, 4.8914108
16: -33.5535278, -23.9746876, -33.5543518, -23.9878654, -6.4515381, 6.4794235
17: -88.6828308, -72.4590149, -88.6822052, -72.4322433, -8.2216377, 8.1860008
18: -4.1723628, 1.0546458, -4.1796160, 1.0597713, -3.4007511, 3.4049702
19: -30.5238171, -23.2043381, -30.5273075, -23.2185783, -4.6530151, 4.6681137
20: -11.1711979, -5.1568747, -11.1729689, -5.1627727, -4.9329758, 4.9322662
21: -43.5470047, -35.0556259, -43.5493851, -35.0770645, -4.2624092, 4.3108559
22: -27.0064697, -19.5544243, -27.0087147, -19.5487785, -4.3464642, 4.3427601
23: -20.8354206, -12.5063238, -20.8542519, -12.5175438, -4.7796364, 4.8089809
24: -16.8501968, -7.6363678, -16.8630714, -7.6446977, -7.1675911, 7.1837730
25: -14.6269436, -6.9568071, -14.6396809, -6.9715376, -4.1893959, 4.2198811
26: -14.6163082, -7.8178372, -14.6230354, -7.8180647, -6.5367622, 6.5433502
27: -14.6321068, -9.5454969, -14.6334591, -9.5449638, -4.0677567, 4.0774956
28: -10.0170221, -1.4226522, -10.0245438, -1.4248848, -6.1605301, 6.1442757
29: -45.5848083, -36.8383789, -45.5864868, -36.8436356, -5.0120277, 5.0326405
30: -32.1880951, -23.0157433, -32.1889038, -23.0267563, -5.0149231, 5.0300808
31: -32.2318459, -23.5188332, -32.2395744, -23.5323868, -6.3061142, 6.3324623
32: 7.7188406, 13.6707993, 7.7203283, 13.6763897, -4.1523132, 4.1547756
33: 4.6363139, 16.3135719, 4.6167040, 16.3130550, -6.6700745, 6.6926270
34: 20.5591908, 30.9911213, 20.5596771, 30.9905167, -5.7484283, 5.7507553
35: 16.5323067, 26.8680534, 16.5230503, 26.8672943, -5.4475212, 5.4547825
36: 28.8234005, 35.1261215, 28.8210564, 35.1266518, -3.4340572, 3.4391155
37: 11.0521488, 20.1179504, 11.0257616, 20.1134834, -5.9370117, 5.9687500
38: 34.8875504, 43.6894913, 34.8734398, 43.6888275, -6.0413094, 6.0664520
39: 9.0155573, 18.5083389, 8.9950781, 18.5078716, -6.5106430, 6.5355148
40: 15.8020344, 25.1258640, 15.7979813, 25.1277122, -5.7927513, 5.7961807
41: 6.7386551, 13.2278423, 6.7317104, 13.2250538, -5.0066643, 5.0215454
42: -12.3783455, -3.4527192, -12.3883381, -3.4660101, -7.0312462, 7.0595207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=80, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1769

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6397618, upper bound: 3.6285796
time: 5.07 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6397618, upper bound: 3.6397619
time: 4.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 11.70 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 11.70
Output dim: 38, lower bound: -3.6397618, upper bound: 3.6136447
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 11.70
Output dim: 38, lower bound: -3.6397618, upper bound: 3.6248171
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 11.70
Output dim: 38, lower bound: -3.6263223, upper bound: 3.6263221
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 11.70
Output dim: 38, lower bound: -3.6263223, upper bound: 3.6410988
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 11.70
Output dim: 38, lower bound: -3.6397618, upper bound: 3.6285796
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 11.70
Output dim: 38, lower bound: -3.6397618, upper bound: 3.6397619

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -21.5168629, -8.4787636, -21.5590916, -8.4735270, -10.3694229, 10.3913231
1: -21.4225864, -12.2498150, -21.4281120, -12.2399616, -5.2623367, 5.2704620
2: -12.3861561, -5.7853503, -12.3902483, -5.7764254, -4.2611961, 4.2466183
3: -11.9950304, -4.1944752, -12.0017738, -4.1725817, -5.3580589, 5.3455582
4: -10.2692146, -0.0368247, -10.2858906, -0.0005536, -6.0437050, 6.0244217
5: -13.5493774, -4.0663948, -13.5574627, -4.0459189, -6.1434059, 6.1290207
6: -8.2934685, 0.5219058, -8.3150368, 0.5357869, -6.4232025, 6.4533958
7: -32.1495590, -22.0934715, -32.1558876, -22.0632362, -5.8424377, 5.8129578
8: -18.7896919, -9.1573267, -18.8104305, -9.1197414, -5.2080765, 5.1934319
9: -5.3066053, 1.3811710, -5.3284016, 1.3893094, -4.0194550, 4.0350666
10: -36.1209564, -27.7876682, -36.1362228, -27.7777519, -5.2345276, 5.2543411
11: -55.0860443, -44.8612099, -55.1192322, -44.8233833, -4.9249287, 4.9234295
12: -11.5651493, -4.6105967, -11.5746765, -4.5948224, -6.2071762, 6.2294006
13: 0.9057541, 7.9740725, 0.8923968, 8.0078564, -5.2847290, 5.2617722
14: -71.0777130, -57.9749222, -71.0851364, -57.9602127, -8.2458038, 8.2330170
15: -8.8971624, 0.8709650, -8.9133625, 0.8987546, -4.8750992, 4.8656406
16: -33.5062294, -24.0108356, -33.5420265, -23.9879456, -6.4072914, 6.4361954
17: -88.6691055, -72.4868164, -88.6785049, -72.4337311, -8.2080650, 8.1595230
18: -4.1424274, 1.0411448, -4.1694112, 1.0596864, -3.3708630, 3.3821716
19: -30.4978848, -23.2283421, -30.5199776, -23.2186203, -4.6330261, 4.6393833
20: -11.1639338, -5.1698656, -11.1712780, -5.1633286, -4.9199524, 4.9192657
21: -43.5118942, -35.0926361, -43.5391579, -35.0773544, -4.2318535, 4.2584934
22: -26.9923134, -19.5675983, -27.0047054, -19.5488586, -4.3348389, 4.3278065
23: -20.8044262, -12.5351582, -20.8445091, -12.5185318, -4.7491398, 4.7690487
24: -16.8169384, -7.6615887, -16.8526611, -7.6448388, -7.1356049, 7.1515999
25: -14.6058960, -6.9806719, -14.6336145, -6.9720201, -4.1702423, 4.1876068
26: -14.6119547, -7.8269777, -14.6220665, -7.8188915, -6.5165405, 6.5338326
27: -14.6083794, -9.5691404, -14.6269331, -9.5453873, -4.0468750, 4.0481415
28: -10.0033226, -1.4362209, -10.0204039, -1.4258026, -6.1479340, 6.1383018
29: -45.5512543, -36.8722229, -45.5763855, -36.8436890, -4.9839764, 4.9910507
30: -32.1459923, -23.0576019, -32.1751785, -23.0284462, -4.9738655, 4.9720306
31: -32.1882477, -23.5508881, -32.2270088, -23.5324287, -6.2679901, 6.2871819
32: 7.7285614, 13.6673527, 7.7221999, 13.6762257, -4.1426735, 4.1475277
33: 4.6616445, 16.3046341, 4.6212945, 16.3123131, -6.6464710, 6.6798630
34: 20.6041927, 30.9637947, 20.5655785, 30.9860344, -5.7019882, 5.7222576
35: 16.5748138, 26.8455544, 16.5297623, 26.8640137, -5.4014969, 5.4288273
36: 28.8409157, 35.1150246, 28.8221111, 35.1242294, -3.4181767, 3.4303293
37: 11.0766010, 20.1070175, 11.0313187, 20.1133308, -5.9100800, 5.9498863
38: 34.9221306, 43.6592026, 34.8747635, 43.6803055, -6.0064774, 6.0380058
39: 9.0348616, 18.5020332, 8.9960566, 18.5058899, -6.4920502, 6.5281868
40: 15.8280935, 25.1147137, 15.8018026, 25.1242828, -5.7634048, 5.7814980
41: 6.7565951, 13.2172480, 6.7364674, 13.2249002, -4.9886627, 5.0067787
42: -12.3689375, -3.4687302, -12.3858404, -3.4665947, -7.0238495, 7.0468636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 692

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6324149, upper bound: 3.6136447
time: 5.25 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6324149, upper bound: 3.6136447
time: 4.95 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -21.5335960, -8.4735823, -21.5639782, -8.4734087, -10.3890076, 10.4435234
1: -21.4260540, -12.2469978, -21.4286079, -12.2389145, -5.2652378, 5.2918224
2: -12.3893890, -5.7832685, -12.3906126, -5.7756691, -4.2613754, 4.2667179
3: -12.0014791, -4.1814680, -12.0018930, -4.1684384, -5.3690071, 5.3522263
4: -10.2844048, -0.0138226, -10.2865915, 0.0076419, -6.0696220, 6.0258179
5: -13.5562534, -4.0582852, -13.5580387, -4.0433455, -6.1446686, 6.1400185
6: -8.3126459, 0.5309341, -8.3213282, 0.5358995, -6.4381409, 6.4694099
7: -32.1511841, -22.0887947, -32.1560669, -22.0616493, -5.8425293, 5.8233871
8: -18.8084431, -9.1157551, -18.8106346, -9.1050682, -5.2428570, 5.2066479
9: -5.3147187, 1.3922573, -5.3288345, 1.3930272, -4.0321045, 4.0373421
10: -36.1273499, -27.7825489, -36.1373215, -27.7762928, -5.2395039, 5.2613564
11: -55.1155319, -44.8439941, -55.1300163, -44.8231506, -4.9284267, 4.9524937
12: -11.5719662, -4.6025586, -11.5769558, -4.5941434, -6.2078819, 6.2392006
13: 0.8952706, 7.9980717, 0.8922865, 8.0157204, -5.3032608, 5.2788544
14: -71.0810852, -57.9716301, -71.0856934, -57.9588470, -8.2493515, 8.2592201
15: -8.9125156, 0.8912406, -8.9146500, 0.9058857, -4.8999348, 4.8698730
16: -33.5329361, -23.9973469, -33.5508499, -23.9879169, -6.4229088, 6.4575386
17: -88.6709290, -72.4777451, -88.6791687, -72.4326172, -8.2115822, 8.1679344
18: -4.1687427, 1.0495377, -4.1788597, 1.0597219, -3.3839855, 3.3999271
19: -30.5096588, -23.2230911, -30.5238228, -23.2185974, -4.6359367, 4.6483288
20: -11.1649885, -5.1666107, -11.1715260, -5.1628108, -4.9346962, 4.9209766
21: -43.5293083, -35.0824661, -43.5451279, -35.0770607, -4.2341671, 4.2761116
22: -26.9994698, -19.5646935, -27.0070000, -19.5488243, -4.3369389, 4.3338509
23: -20.8274479, -12.5204048, -20.8525276, -12.5176458, -4.7570953, 4.7932415
24: -16.8451824, -7.6463199, -16.8621693, -7.6447573, -7.1533051, 7.1764565
25: -14.6172056, -6.9732475, -14.6372147, -6.9715900, -4.1749706, 4.1997871
26: -14.6140413, -7.8219118, -14.6225395, -7.8182034, -6.5505104, 6.5374985
27: -14.6216927, -9.5610828, -14.6309271, -9.5450268, -4.0546741, 4.0601959
28: -10.0137863, -1.4256409, -10.0239143, -1.4249516, -6.1659431, 6.1385651
29: -45.5714951, -36.8615417, -45.5831757, -36.8436584, -4.9939632, 5.0081444
30: -32.1757889, -23.0374489, -32.1858444, -23.0269337, -4.9789314, 5.0064030
31: -32.2156258, -23.5406399, -32.2359810, -23.5324364, -6.2901688, 6.3063202
32: 7.7226205, 13.6702166, 7.7204351, 13.6763306, -4.1486168, 4.1556454
33: 4.6484704, 16.3102951, 4.6172800, 16.3123856, -6.6522064, 6.6879845
34: 20.5872726, 30.9725266, 20.5599651, 30.9860992, -5.7068787, 5.7366619
35: 16.5555687, 26.8554230, 16.5234947, 26.8640556, -5.4068413, 5.4449844
36: 28.8375168, 35.1175156, 28.8213367, 35.1244583, -3.4203720, 3.4327612
37: 11.0593052, 20.1139793, 11.0260010, 20.1134567, -5.9201927, 5.9628983
38: 34.9088516, 43.6736565, 34.8737564, 43.6852036, -6.0255814, 6.0425644
39: 9.0249081, 18.5070686, 8.9954853, 18.5076828, -6.5015373, 6.5298691
40: 15.8133955, 25.1233101, 15.7984743, 25.1271210, -5.7825890, 5.7918625
41: 6.7422552, 13.2248526, 6.7319560, 13.2250118, -4.9985390, 5.0189514
42: -12.3699703, -3.4656718, -12.3861904, -3.4660873, -7.0256691, 7.0493622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 692

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6324149, upper bound: 3.6248171
time: 4.51 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6324149, upper bound: 3.6248171
time: 4.59 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.5679474, -8.4728537, -21.5349827, -8.4775658, -10.4213028, 10.3929977
1: -21.4304237, -12.2357283, -21.4261208, -12.2465963, -5.2680836, 5.2717342
2: -12.3958588, -5.7761455, -12.3881674, -5.7807775, -4.2662544, 4.2603359
3: -12.0143356, -4.1652670, -11.9995918, -4.1730924, -5.3745880, 5.3684006
4: -10.2951298, 0.0073671, -10.2798367, -0.0099595, -6.0612335, 6.0645409
5: -13.5600185, -4.0418997, -13.5535526, -4.0564847, -6.1385956, 6.1468048
6: -8.3208990, 0.5451291, -8.3182087, 0.5296122, -6.4489632, 6.4572334
7: -32.1576462, -22.0537567, -32.1498985, -22.0879230, -5.8235397, 5.8438892
8: -18.8184052, -9.1035748, -18.8079491, -9.1093864, -5.2441864, 5.2421303
9: -5.3360901, 1.3952842, -5.3148642, 1.3916485, -4.0572243, 4.0329819
10: -36.1393700, -27.7631569, -36.1357231, -27.7821636, -5.2492714, 5.2600212
11: -55.1346359, -44.7878494, -55.1306686, -44.8439865, -4.9471989, 5.0099220
12: -11.5797243, -4.5854759, -11.5768871, -4.6032119, -6.2190247, 6.2249908
13: 0.8828372, 8.0157375, 0.8962187, 8.0038033, -5.3037376, 5.2999420
14: -71.0865479, -57.9567757, -71.0799255, -57.9710464, -8.2460938, 8.2500992
15: -8.9213018, 0.9037843, -8.9102926, 0.8964529, -4.9041443, 4.8955040
16: -33.5595284, -23.9654160, -33.5447197, -23.9972992, -6.4535294, 6.4642868
17: -88.6817017, -72.4151993, -88.6705627, -72.4775391, -8.1732368, 8.2197075
18: -4.1795597, 1.0646925, -4.1690412, 1.0493577, -3.3981361, 3.4028568
19: -30.5272064, -23.1998501, -30.5219345, -23.2235966, -4.6488361, 4.6690388
20: -11.1730270, -5.1532726, -11.1696882, -5.1669607, -4.9239807, 4.9347382
21: -43.5491066, -35.0503845, -43.5444031, -35.0823631, -4.2640476, 4.3009949
22: -27.0086765, -19.5386868, -27.0017796, -19.5649948, -4.3325462, 4.3514862
23: -20.8540897, -12.5039806, -20.8346825, -12.5233192, -4.7911758, 4.7922153
24: -16.8625031, -7.6350837, -16.8499680, -7.6488848, -7.1733170, 7.1724739
25: -14.6394348, -6.9554529, -14.6267328, -6.9748955, -4.1974564, 4.2088966
26: -14.6229258, -7.8148222, -14.6154070, -7.8226523, -6.5387115, 6.5380936
27: -14.6337366, -9.5296726, -14.6275692, -9.5612411, -4.0549736, 4.0835209
28: -10.0247364, -1.4222350, -10.0160770, -1.4263124, -6.1467667, 6.1427155
29: -45.5860825, -36.8206100, -45.5801239, -36.8618622, -4.9989128, 5.0357494
30: -32.1887589, -23.0059319, -32.1863022, -23.0372448, -5.0046101, 5.0361023
31: -32.2392578, -23.5106621, -32.2299309, -23.5412140, -6.3040314, 6.3306732
32: 7.7171478, 13.6763506, 7.7230186, 13.6694431, -4.1492691, 4.1500607
33: 4.6068058, 16.3131580, 4.6471705, 16.3064880, -6.6944180, 6.6598358
34: 20.5328541, 30.9906540, 20.5869045, 30.9821568, -5.7664280, 5.7223759
35: 16.5016556, 26.8667946, 16.5542526, 26.8604088, -5.4709797, 5.4220791
36: 28.8080597, 35.1262894, 28.8369675, 35.1226082, -3.4463415, 3.4223576
37: 11.0222206, 20.1172714, 11.0567074, 20.1072102, -5.9613190, 5.9359283
38: 34.8534851, 43.6899033, 34.9085999, 43.6785736, -6.0656815, 6.0315361
39: 8.9874249, 18.5082664, 9.0240307, 18.5003319, -6.5319519, 6.5057831
40: 15.7888088, 25.1277905, 15.8120565, 25.1235428, -5.8024940, 5.7833271
41: 6.7308702, 13.2278528, 6.7406673, 13.2214622, -5.0127068, 5.0073891
42: -12.3879204, -3.4534540, -12.3778133, -3.4690402, -7.0390816, 7.0405998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=80, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6138000, upper bound: 3.6397664
time: 5.92 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6249885, upper bound: 3.6397665
time: 5.06 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5213089, -8.4779768, -21.5592861, -8.4734545, -10.3735962, 10.3917961
1: -21.4259071, -12.2463541, -21.4284630, -12.2398834, -5.2645531, 5.2721157
2: -12.3915691, -5.7833719, -12.3902779, -5.7761707, -4.2689152, 4.2482357
3: -12.0075550, -4.1850634, -12.0017872, -4.1703849, -5.3697586, 5.3516693
4: -10.2781658, -0.0337025, -10.2860041, 0.0000312, -6.0525589, 6.0274239
5: -13.5516863, -4.0631890, -13.5574923, -4.0457773, -6.1463890, 6.1323853
6: -8.2989216, 0.5312831, -8.3158188, 0.5358554, -6.4289627, 6.4612503
7: -32.1558189, -22.0852318, -32.1573257, -22.0631409, -5.8490944, 5.8185024
8: -18.7976074, -9.1518097, -18.8104591, -9.1184893, -5.2134514, 5.1968670
9: -5.3144760, 1.3835889, -5.3285575, 1.3893924, -4.0313015, 4.0359001
10: -36.1299667, -27.7738953, -36.1385002, -27.7775230, -5.2405376, 5.2674236
11: -55.1054535, -44.8256454, -55.1239929, -44.8232880, -4.9374523, 4.9666939
12: -11.5723801, -4.6013298, -11.5761452, -4.5946326, -6.2123413, 6.2344666
13: 0.8960335, 7.9786615, 0.8923295, 8.0085392, -5.2969894, 5.2662392
14: -71.0804749, -57.9721756, -71.0855331, -57.9600716, -8.2499428, 8.2337456
15: -8.9043407, 0.8733330, -8.9134741, 0.8989859, -4.8873577, 4.8682804
16: -33.5262375, -23.9882011, -33.5453873, -23.9879189, -6.4250336, 6.4578133
17: -88.6800613, -72.4682465, -88.6812897, -72.4333801, -8.2158356, 8.1751785
18: -4.1453462, 1.0462530, -4.1699729, 1.0597267, -3.3734760, 3.3869324
19: -30.5116730, -23.2095871, -30.5233727, -23.2186069, -4.6422844, 4.6590881
20: -11.1700821, -5.1601968, -11.1727066, -5.1633062, -4.9246216, 4.9291344
21: -43.5292130, -35.0658493, -43.5433197, -35.0773392, -4.2435913, 4.2932014
22: -26.9988613, -19.5573711, -27.0062904, -19.5488472, -4.3391933, 4.3365955
23: -20.8114014, -12.5211506, -20.8460083, -12.5184317, -4.7544117, 4.7845612
24: -16.8207474, -7.6516590, -16.8533058, -7.6447663, -7.1382751, 7.1586189
25: -14.6152191, -6.9642582, -14.6359730, -6.9719639, -4.1766682, 4.2076035
26: -14.6141300, -7.8233123, -14.6225691, -7.8188100, -6.5179443, 6.5372887
27: -14.6183262, -9.5535860, -14.6293554, -9.5453281, -4.0536919, 4.0652905
28: -10.0062666, -1.4332821, -10.0209675, -1.4257627, -6.1518745, 6.1405144
29: -45.5641327, -36.8490524, -45.5796013, -36.8436432, -4.9915371, 5.0153351
30: -32.1569557, -23.0360603, -32.1779518, -23.0283165, -4.9810104, 4.9954185
31: -32.2034988, -23.5290794, -32.2303467, -23.5324059, -6.2787857, 6.3130455
32: 7.7249870, 13.6678991, 7.7221375, 13.6762991, -4.1458740, 4.1483612
33: 4.6499443, 16.3079014, 4.6209078, 16.3129578, -6.6589432, 6.6839409
34: 20.5767822, 30.9823647, 20.5654659, 30.9904537, -5.7306023, 5.7361450
35: 16.5523834, 26.8581848, 16.5294914, 26.8672600, -5.4274979, 5.4384441
36: 28.8271904, 35.1236000, 28.8219452, 35.1264114, -3.4308052, 3.4363623
37: 11.0700665, 20.1109200, 11.0312586, 20.1133518, -5.9184418, 5.9556007
38: 34.9009323, 43.6744728, 34.8744812, 43.6837769, -6.0220375, 6.0500450
39: 9.0255632, 18.5028839, 8.9956608, 18.5059891, -6.5004082, 6.5314102
40: 15.8169727, 25.1164799, 15.8014565, 25.1246929, -5.7732124, 5.7824764
41: 6.7535591, 13.2201853, 6.7363572, 13.2249289, -4.9918213, 5.0091820
42: -12.3771276, -3.4558194, -12.3879271, -3.4665074, -7.0294304, 7.0566597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6283009, upper bound: 3.6278570
time: 4.87 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6390400, upper bound: 3.6278570
time: 4.72 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5380974, -8.4728279, -21.5641689, -8.4733534, -10.3931580, 10.4440079
1: -21.4293518, -12.2435198, -21.4289589, -12.2388506, -5.2674446, 5.2934723
2: -12.3947906, -5.7812767, -12.3906174, -5.7753963, -4.2691078, 4.2683430
3: -12.0140038, -4.1720681, -12.0019064, -4.1662254, -5.3807106, 5.3583260
4: -10.2933865, -0.0107057, -10.2867041, 0.0082493, -6.0784760, 6.0288181
5: -13.5585051, -4.0550966, -13.5580921, -4.0432587, -6.1476479, 6.1433716
6: -8.3180666, 0.5403080, -8.3221149, 0.5359217, -6.4438858, 6.4772339
7: -32.1574554, -22.0805969, -32.1575165, -22.0615253, -5.8491783, 5.8289242
8: -18.8163471, -9.1102457, -18.8106861, -9.1037893, -5.2482262, 5.2100792
9: -5.3225613, 1.3946368, -5.3290014, 1.3930941, -4.0439491, 4.0381737
10: -36.1363525, -27.7687950, -36.1395988, -27.7760773, -5.2455006, 5.2744484
11: -55.1348877, -44.8084946, -55.1347847, -44.8230438, -4.9409485, 4.9957695
12: -11.5791931, -4.5932984, -11.5784235, -4.5939388, -6.2130470, 6.2442322
13: 0.8855148, 8.0026646, 0.8922193, 8.0164080, -5.3155403, 5.2832870
14: -71.0838623, -57.9688911, -71.0860825, -57.9586029, -8.2534981, 8.2599373
15: -8.9197025, 0.8936372, -8.9148045, 0.9061027, -4.9121666, 4.8725090
16: -33.5529289, -23.9747047, -33.5542145, -23.9878597, -6.4406509, 6.4791603
17: -88.6819000, -72.4591904, -88.6819534, -72.4322815, -8.2193413, 8.1835709
18: -4.1716924, 1.0546489, -4.1794143, 1.0597694, -3.3866348, 3.4046860
19: -30.5234299, -23.2043324, -30.5271873, -23.2185898, -4.6451912, 4.6680279
20: -11.1711235, -5.1569347, -11.1729498, -5.1627846, -4.9393463, 4.9308243
21: -43.5466003, -35.0556221, -43.5492897, -35.0770569, -4.2458973, 4.3108101
22: -27.0060425, -19.5544319, -27.0085716, -19.5487804, -4.3413048, 4.3426304
23: -20.8344402, -12.5063896, -20.8540154, -12.5175858, -4.7623825, 4.8087482
24: -16.8489838, -7.6363945, -16.8628101, -7.6447001, -7.1559753, 7.1834717
25: -14.6265030, -6.9568434, -14.6395569, -6.9715424, -4.1813946, 4.2197838
26: -14.6162357, -7.8182297, -14.6230173, -7.8181334, -6.5518684, 6.5409546
27: -14.6316147, -9.5455322, -14.6333380, -9.5449762, -4.0614891, 4.0773392
28: -10.0167561, -1.4227362, -10.0244846, -1.4249275, -6.1698990, 6.1407852
29: -45.5843544, -36.8383904, -45.5863762, -36.8436165, -5.0015221, 5.0324306
30: -32.1867485, -23.0158920, -32.1885986, -23.0267906, -4.9860859, 5.0297604
31: -32.2308807, -23.5188408, -32.2393265, -23.5324059, -6.3009682, 6.3321724
32: 7.7190447, 13.6707697, 7.7203732, 13.6763849, -4.1517944, 4.1564770
33: 4.6367965, 16.3135490, 4.6168346, 16.3130302, -6.6646671, 6.6920853
34: 20.5598621, 30.9910774, 20.5598488, 30.9905033, -5.7354813, 5.7505589
35: 16.5330772, 26.8680382, 16.5232410, 26.8672924, -5.4328423, 5.4546013
36: 28.8238106, 35.1261139, 28.8211555, 35.1266518, -3.4329929, 3.4387836
37: 11.0527668, 20.1178799, 11.0259361, 20.1134796, -5.9285583, 5.9685974
38: 34.8876686, 43.6889000, 34.8734703, 43.6886635, -6.0411110, 6.0545921
39: 9.0156326, 18.5079346, 8.9951000, 18.5077934, -6.5098991, 6.5330772
40: 15.8023291, 25.1250801, 15.7980995, 25.1275177, -5.7923470, 5.7928562
41: 6.7392364, 13.2277927, 6.7318611, 13.2250547, -5.0016785, 5.0213699
42: -12.3781681, -3.4528108, -12.3882771, -3.4660535, -7.0312614, 7.0591469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=80, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1597

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6283009, upper bound: 3.6390402
time: 5.15 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6390400, upper bound: 3.6390402
time: 5.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 13.08 seconds
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.08
Output dim: 38, lower bound: -3.6324149, upper bound: 3.6136447
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.08
Output dim: 38, lower bound: -3.6324149, upper bound: 3.6136447
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.08
Output dim: 38, lower bound: -3.6324149, upper bound: 3.6248171
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.08
Output dim: 38, lower bound: -3.6324149, upper bound: 3.6248171
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.08
Output dim: 38, lower bound: -3.6138000, upper bound: 3.6397664
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.08
Output dim: 38, lower bound: -3.6249885, upper bound: 3.6397665
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 13.08
Output dim: 38, lower bound: -3.6283009, upper bound: 3.6278570
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.08
Output dim: 38, lower bound: -3.6390400, upper bound: 3.6278570
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.08
Output dim: 38, lower bound: -3.6283009, upper bound: 3.6390402
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.08
Output dim: 38, lower bound: -3.6390400, upper bound: 3.6390402

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -21.5168629, -8.4787636, -21.5579891, -8.4737301, -10.3688126, 10.3897133
1: -21.4225864, -12.2498150, -21.4265709, -12.2402706, -5.2621441, 5.2695560
2: -12.3861561, -5.7853503, -12.3900814, -5.7789254, -4.2583542, 4.2464466
3: -11.9950304, -4.1944752, -12.0016584, -4.1789584, -5.3528404, 5.3454247
4: -10.2692146, -0.0368247, -10.2855053, -0.0042294, -6.0403328, 6.0238113
5: -13.5493774, -4.0663948, -13.5571671, -4.0477452, -6.1414909, 6.1287270
6: -8.2934685, 0.5219058, -8.3089628, 0.5356274, -6.4230042, 6.4484138
7: -32.1495590, -22.0934715, -32.1512146, -22.0637169, -5.8410034, 5.8096676
8: -18.7896919, -9.1573267, -18.8102474, -9.1241875, -5.2050362, 5.1932144
9: -5.3066053, 1.3811710, -5.3277960, 1.3890753, -4.0187969, 4.0342598
10: -36.1209564, -27.7876682, -36.1292534, -27.7784309, -5.2338848, 5.2487812
11: -55.0860443, -44.8612099, -55.1043930, -44.8236656, -4.9246426, 4.9087048
12: -11.5651493, -4.6105967, -11.5701637, -4.5954504, -6.2065849, 6.2260628
13: 0.9057541, 7.9740725, 0.8926972, 8.0030241, -5.2795219, 5.2614326
14: -71.0777130, -57.9749222, -71.0832062, -57.9609833, -8.2441406, 8.2316742
15: -8.8971624, 0.8709650, -8.9128056, 0.8940053, -4.8694763, 4.8647823
16: -33.5062294, -24.0108356, -33.5305786, -23.9880905, -6.4071693, 6.4287415
17: -88.6691055, -72.4868164, -88.6698151, -72.4349518, -8.2061768, 8.1533508
18: -4.1424274, 1.0411448, -4.1669617, 1.0595417, -3.3707047, 3.3798180
19: -30.4978848, -23.2283421, -30.5094643, -23.2186184, -4.6327038, 4.6303043
20: -11.1639338, -5.1698656, -11.1666203, -5.1634684, -4.9190140, 4.9145393
21: -43.5118942, -35.0926361, -43.5257492, -35.0774765, -4.2316093, 4.2433395
22: -26.9923134, -19.5675983, -26.9996967, -19.5489807, -4.3347187, 4.3237286
23: -20.8044262, -12.5351582, -20.8388577, -12.5188732, -4.7486248, 4.7638550
24: -16.8169384, -7.6615887, -16.8488808, -7.6450906, -7.1353302, 7.1483307
25: -14.6058960, -6.9806719, -14.6264210, -6.9722729, -4.1699715, 4.1801491
26: -14.6119547, -7.8269777, -14.6202908, -7.8192320, -6.5161896, 6.5322838
27: -14.6083794, -9.5691404, -14.6196613, -9.5456190, -4.0466290, 4.0409908
28: -10.0033226, -1.4362209, -10.0182028, -1.4260259, -6.1465836, 6.1359978
29: -45.5512543, -36.8722229, -45.5663223, -36.8437920, -4.9838142, 4.9838486
30: -32.1459923, -23.0576019, -32.1668129, -23.0290375, -4.9731503, 4.9651299
31: -32.1882477, -23.5508881, -32.2147713, -23.5325089, -6.2678833, 6.2755966
32: 7.7285614, 13.6673527, 7.7225561, 13.6757069, -4.1422882, 4.1471863
33: 4.6616445, 16.3046341, 4.6226387, 16.3098259, -6.6447525, 6.6786194
34: 20.6041927, 30.9637947, 20.5660381, 30.9719543, -5.6907768, 5.7209206
35: 16.5748138, 26.8455544, 16.5305443, 26.8541126, -5.3925190, 5.4277573
36: 28.8409157, 35.1150246, 28.8226433, 35.1174469, -3.4127369, 3.4292746
37: 11.0766010, 20.1070175, 11.0342846, 20.1132317, -5.9099884, 5.9467278
38: 34.9221306, 43.6592026, 34.8756943, 43.6696129, -5.9990921, 6.0363960
39: 9.0348616, 18.5020332, 8.9973106, 18.5055466, -6.4912300, 6.5262413
40: 15.8280935, 25.1147137, 15.8033533, 25.1229782, -5.7628536, 5.7800751
41: 6.7565951, 13.2172480, 6.7385492, 13.2248135, -4.9885902, 5.0049973
42: -12.3689375, -3.4687302, -12.3793125, -3.4668787, -7.0235443, 7.0422516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6021840
time: 5.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6129224
time: 5.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.5168629, -8.4787636, -21.5624542, -8.4729576, -10.3693314, 10.3938446
1: -21.4225864, -12.2498150, -21.4298611, -12.2368126, -5.2638931, 5.2712193
2: -12.3861561, -5.7853503, -12.3954735, -5.7769561, -4.2618027, 4.2533779
3: -11.9950304, -4.1944752, -12.0142117, -4.1695313, -5.3595657, 5.3559570
4: -10.2692146, -0.0368247, -10.2944317, -0.0011063, -6.0440178, 6.0319538
5: -13.5493774, -4.0663948, -13.5594215, -4.0445385, -6.1450424, 6.1315231
6: -8.2934685, 0.5219058, -8.3143930, 0.5450232, -6.4305382, 6.4523811
7: -32.1495590, -22.0934715, -32.1574669, -22.0555000, -5.8453865, 5.8132820
8: -18.7896919, -9.1573267, -18.8181534, -9.1186619, -5.2083759, 5.1978931
9: -5.3066053, 1.3811710, -5.3356285, 1.3914773, -4.0199986, 4.0422192
10: -36.1209564, -27.7876682, -36.1382675, -27.7646751, -5.2449493, 5.2563591
11: -55.0860443, -44.8612099, -55.1237793, -44.7881317, -4.9621258, 4.9292984
12: -11.5651493, -4.6105967, -11.5773869, -4.5861816, -6.2113533, 6.2304916
13: 0.9057541, 7.9740725, 0.8829433, 8.0076275, -5.2863007, 5.2720757
14: -71.0777130, -57.9749222, -71.0859756, -57.9582825, -8.2451210, 8.2323189
15: -8.8971624, 0.8709650, -8.9199829, 0.8964386, -4.8773670, 4.8747902
16: -33.5062294, -24.0108356, -33.5505753, -23.9654465, -6.4257812, 6.4440651
17: -88.6691055, -72.4868164, -88.6807861, -72.4163742, -8.2195091, 8.1614265
18: -4.1424274, 1.0411448, -4.1699185, 1.0646598, -3.3750248, 3.3824005
19: -30.4978848, -23.2283421, -30.5232391, -23.1998672, -4.6495132, 4.6423206
20: -11.1639338, -5.1698656, -11.1727781, -5.1538057, -4.9271736, 4.9196739
21: -43.5118942, -35.0926361, -43.5430489, -35.0506821, -4.2609959, 4.2641258
22: -26.9923134, -19.5675983, -27.0062599, -19.5387287, -4.3427696, 4.3279476
23: -20.8044262, -12.5351582, -20.8458443, -12.5048733, -4.7620697, 4.7713165
24: -16.8169384, -7.6615887, -16.8527222, -7.6351585, -7.1423683, 7.1505890
25: -14.6058960, -6.9806719, -14.6357040, -6.9558754, -4.1872959, 4.1902561
26: -14.6119547, -7.8269777, -14.6224575, -7.8156013, -6.5193863, 6.5337944
27: -14.6083794, -9.5691404, -14.6296043, -9.5300503, -4.0609760, 4.0511456
28: -10.0033226, -1.4362209, -10.0211363, -1.4230995, -6.1485596, 6.1375084
29: -45.5512543, -36.8722229, -45.5791855, -36.8206444, -5.0048676, 4.9941635
30: -32.1459923, -23.0576019, -32.1777916, -23.0074501, -4.9940987, 4.9743767
31: -32.1882477, -23.5508881, -32.2299995, -23.5106735, -6.2891197, 6.2921143
32: 7.7285614, 13.6673527, 7.7189751, 13.6762772, -4.1423874, 4.1504211
33: 4.6616445, 16.3046341, 4.6110044, 16.3130531, -6.6453400, 6.6911736
34: 20.6041927, 30.9637947, 20.5386314, 30.9905548, -5.7061691, 5.7445164
35: 16.5748138, 26.8455544, 16.5081272, 26.8667831, -5.4042435, 5.4501686
36: 28.8409157, 35.1150246, 28.8089447, 35.1260376, -3.4201832, 3.4396410
37: 11.0766010, 20.1070175, 11.0277414, 20.1171265, -5.9150848, 5.9557419
38: 34.9221306, 43.6592026, 34.8545227, 43.6848602, -6.0079803, 6.0500107
39: 9.0348616, 18.5020332, 8.9880447, 18.5064011, -6.4916916, 6.5347061
40: 15.8280935, 25.1147137, 15.7922583, 25.1247883, -5.7624741, 5.7901402
41: 6.7565951, 13.2172480, 6.7355266, 13.2277508, -4.9911156, 5.0079269
42: -12.3689375, -3.4687302, -12.3874989, -3.4539931, -7.0322723, 7.0469627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6021840
time: 5.28 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6129224
time: 5.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5335960, -8.4735823, -21.5628529, -8.4736452, -10.3883743, 10.4419022
1: -21.4260540, -12.2469978, -21.4270782, -12.2392302, -5.2650414, 5.2909164
2: -12.3893890, -5.7832685, -12.3904295, -5.7781506, -4.2585373, 4.2665520
3: -12.0014791, -4.1814680, -12.0018005, -4.1747837, -5.3637924, 5.3520927
4: -10.2844048, -0.0138226, -10.2861500, 0.0040040, -6.0662613, 6.0252037
5: -13.5562534, -4.0582852, -13.5577593, -4.0451884, -6.1427574, 6.1397209
6: -8.3126459, 0.5309341, -8.3152781, 0.5357314, -6.4379425, 6.4644318
7: -32.1511841, -22.0887947, -32.1513901, -22.0620995, -5.8410988, 5.8200989
8: -18.8084431, -9.1157551, -18.8104801, -9.1095209, -5.2398167, 5.2064323
9: -5.3147187, 1.3922573, -5.3282328, 1.3928072, -4.0314541, 4.0365334
10: -36.1273499, -27.7825489, -36.1303406, -27.7769890, -5.2388496, 5.2557945
11: -55.1155319, -44.8439941, -55.1151199, -44.8233948, -4.9281387, 4.9377689
12: -11.5719662, -4.6025586, -11.5724564, -4.5947604, -6.2072983, 6.2358589
13: 0.8952706, 7.9980717, 0.8925644, 8.0109158, -5.2980499, 5.2785149
14: -71.0810852, -57.9716301, -71.0837402, -57.9595947, -8.2476997, 8.2578812
15: -8.9125156, 0.8912406, -8.9141445, 0.9011512, -4.8943138, 4.8690205
16: -33.5329361, -23.9973469, -33.5393791, -23.9880543, -6.4227829, 6.4500732
17: -88.6709290, -72.4777451, -88.6705170, -72.4338379, -8.2096901, 8.1617622
18: -4.1687427, 1.0495377, -4.1764007, 1.0595772, -3.3838291, 3.3975563
19: -30.5096588, -23.2230911, -30.5132923, -23.2186012, -4.6356144, 4.6392441
20: -11.1649885, -5.1666107, -11.1668491, -5.1629543, -4.9337540, 4.9162388
21: -43.5293083, -35.0824661, -43.5317039, -35.0772057, -4.2339230, 4.2609596
22: -26.9994698, -19.5646935, -27.0019684, -19.5489502, -4.3368244, 4.3297653
23: -20.8274479, -12.5204048, -20.8468666, -12.5180187, -4.7565823, 4.7880554
24: -16.8451824, -7.6463199, -16.8584061, -7.6450009, -7.1530228, 7.1731720
25: -14.6172056, -6.9732475, -14.6300259, -6.9718661, -4.1746960, 4.1923275
26: -14.6140413, -7.8219118, -14.6207285, -7.8185945, -6.5501442, 6.5359383
27: -14.6216927, -9.5610828, -14.6236668, -9.5452509, -4.0544300, 4.0530510
28: -10.0137863, -1.4256409, -10.0217056, -1.4251826, -6.1645927, 6.1362648
29: -45.5714951, -36.8615417, -45.5731354, -36.8437958, -4.9938011, 5.0009384
30: -32.1757889, -23.0374489, -32.1774712, -23.0275249, -4.9782143, 4.9995003
31: -32.2156258, -23.5406399, -32.2237244, -23.5325012, -6.2900734, 6.2947159
32: 7.7226205, 13.6702166, 7.7207699, 13.6757851, -4.1482182, 4.1552982
33: 4.6484704, 16.3102951, 4.6186004, 16.3098984, -6.6504898, 6.6867371
34: 20.5872726, 30.9725266, 20.5604057, 30.9720345, -5.6956635, 5.7353268
35: 16.5555687, 26.8554230, 16.5242805, 26.8541603, -5.3978672, 5.4439125
36: 28.8375168, 35.1175156, 28.8218498, 35.1176682, -3.4149303, 3.4317007
37: 11.0593052, 20.1139793, 11.0289516, 20.1133518, -5.9200974, 5.9597397
38: 34.9088516, 43.6736565, 34.8746872, 43.6744919, -6.0181961, 6.0409393
39: 9.0249081, 18.5070686, 8.9967442, 18.5073223, -6.5007172, 6.5279160
40: 15.8133955, 25.1233101, 15.8000097, 25.1258392, -5.7820358, 5.7904453
41: 6.7422552, 13.2248526, 6.7340412, 13.2249250, -4.9984550, 5.0171738
42: -12.3699703, -3.4656718, -12.3796778, -3.4663649, -7.0253525, 7.0447578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6133573
time: 5.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6240949
time: 5.25 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5335960, -8.4735823, -21.5673809, -8.4728556, -10.3888931, 10.4460335
1: -21.4260540, -12.2469978, -21.4303684, -12.2357578, -5.2667885, 5.2925777
2: -12.3893890, -5.7832685, -12.3958473, -5.7761631, -4.2619953, 4.2734776
3: -12.0014791, -4.1814680, -12.0143089, -4.1653881, -5.3705063, 5.3626289
4: -10.2844048, -0.0138226, -10.2951117, 0.0071352, -6.0699387, 6.0333481
5: -13.5562534, -4.0582852, -13.5599976, -4.0419989, -6.1462975, 6.1425209
6: -8.3126459, 0.5309341, -8.3206921, 0.5451202, -6.4454727, 6.4683952
7: -32.1511841, -22.0887947, -32.1576385, -22.0539093, -5.8454819, 5.8237038
8: -18.8084431, -9.1157551, -18.8183880, -9.1039915, -5.2431736, 5.2111244
9: -5.3147187, 1.3922573, -5.3360853, 1.3951790, -4.0326519, 4.0444927
10: -36.1273499, -27.7825489, -36.1393585, -27.7631989, -5.2499199, 5.2633724
11: -55.1155319, -44.8439941, -55.1344986, -44.7878532, -4.9656200, 4.9583549
12: -11.5719662, -4.6025586, -11.5797014, -4.5855045, -6.2120590, 6.2402840
13: 0.8952706, 7.9980717, 0.8828471, 8.0155029, -5.3048477, 5.2891464
14: -71.0810852, -57.9716301, -71.0865173, -57.9568710, -8.2486610, 8.2585220
15: -8.9125156, 0.8912406, -8.9212809, 0.9035616, -4.9022045, 4.8790188
16: -33.5329361, -23.9973469, -33.5593910, -23.9654236, -6.4413948, 6.4654083
17: -88.6709290, -72.4777451, -88.6814728, -72.4152374, -8.2230034, 8.1698341
18: -4.1687427, 1.0495377, -4.1793766, 1.0646970, -3.3881512, 3.4001503
19: -30.5096588, -23.2230911, -30.5270691, -23.1998634, -4.6524162, 4.6512756
20: -11.1649885, -5.1666107, -11.1730042, -5.1532845, -4.9418983, 4.9213715
21: -43.5293083, -35.0824661, -43.5490036, -35.0504112, -4.2633076, 4.2817421
22: -26.9994698, -19.5646935, -27.0085449, -19.5386906, -4.3448772, 4.3339920
23: -20.8274479, -12.5204048, -20.8538589, -12.5040054, -4.7700195, 4.7955170
24: -16.8451824, -7.6463199, -16.8622322, -7.6350894, -7.1600647, 7.1754494
25: -14.6172056, -6.9732475, -14.6393414, -6.9554410, -4.1920300, 4.2024403
26: -14.6140413, -7.8219118, -14.6229076, -7.8149238, -6.5533371, 6.5374451
27: -14.6216927, -9.5610828, -14.6336031, -9.5296965, -4.0687790, 4.0632076
28: -10.0137863, -1.4256409, -10.0246840, -1.4222549, -6.1665878, 6.1377716
29: -45.5714951, -36.8615417, -45.5859833, -36.8205948, -5.0148468, 5.0112686
30: -32.1757889, -23.0374489, -32.1884651, -23.0059490, -4.9991531, 5.0087452
31: -32.2156258, -23.5406399, -32.2389832, -23.5106621, -6.3113213, 6.3112526
32: 7.7226205, 13.6702166, 7.7172108, 13.6763563, -4.1483212, 4.1585255
33: 4.6484704, 16.3102951, 4.6069369, 16.3131485, -6.6510773, 6.6992836
34: 20.5872726, 30.9725266, 20.5330029, 30.9906311, -5.7110538, 5.7589149
35: 16.5555687, 26.8554230, 16.5018425, 26.8668022, -5.4095917, 5.4663258
36: 28.8375168, 35.1175156, 28.8081627, 35.1262741, -3.4223824, 3.4420776
37: 11.0593052, 20.1139793, 11.0223980, 20.1172295, -5.9251976, 5.9687691
38: 34.9088516, 43.6736565, 34.8535309, 43.6897278, -6.0270920, 6.0545311
39: 9.0249081, 18.5070686, 8.9874611, 18.5081730, -6.5011787, 6.5363617
40: 15.8133955, 25.1233101, 15.7889147, 25.1276188, -5.7816620, 5.8005066
41: 6.7422552, 13.2248526, 6.7310100, 13.2278471, -5.0009766, 5.0201302
42: -12.3699703, -3.4656718, -12.3878756, -3.4535041, -7.0340958, 7.0494614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6133573
time: 5.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6240949
time: 4.97 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -21.5624542, -8.4729576, -21.5158043, -8.4827681, -10.3892593, 10.3696671
1: -21.4298611, -12.2368126, -21.4224052, -12.2495413, -5.2564983, 5.2672977
2: -12.3954735, -5.7769561, -12.3848925, -5.7829604, -4.2566299, 4.2585812
3: -12.0142117, -4.1695313, -11.9931116, -4.1865726, -5.3611641, 5.3573418
4: -10.2944317, -0.0011063, -10.2645473, -0.0340023, -6.0357819, 6.0385246
5: -13.5594215, -4.0445385, -13.5465946, -4.0648685, -6.1315918, 6.1424026
6: -8.3143930, 0.5450232, -8.2983370, 0.5205963, -6.4327621, 6.4373589
7: -32.1574669, -22.0555000, -32.1481667, -22.0930958, -5.8168793, 5.8421612
8: -18.8181534, -9.1186619, -18.7891388, -9.1523466, -5.2012749, 5.2070770
9: -5.3356285, 1.3914773, -5.3067150, 1.3800933, -4.0450001, 4.0202351
10: -36.1382675, -27.7646751, -36.1292496, -27.7874451, -5.2436218, 5.2540894
11: -55.1237793, -44.7881317, -55.1005249, -44.8612175, -4.9179573, 4.9789600
12: -11.5773869, -4.5861816, -11.5699205, -4.6113911, -6.2089462, 6.2181091
13: 0.8829433, 8.0076275, 0.9067791, 7.9789166, -5.2785683, 5.2811394
14: -71.0859756, -57.9582825, -71.0763702, -57.9744644, -8.2299080, 8.2440605
15: -8.9199829, 0.8964386, -8.8948135, 0.8752904, -4.8810368, 4.8703938
16: -33.5505753, -23.9654465, -33.5174179, -24.0108185, -6.4319344, 6.4377747
17: -88.6807861, -72.4163742, -88.6677856, -72.4867401, -8.1623993, 8.2139168
18: -4.1699185, 1.0646598, -4.1420212, 1.0409420, -3.3801365, 3.3755322
19: -30.5232391, -23.1998672, -30.5097504, -23.2288570, -4.6398087, 4.6583118
20: -11.1727781, -5.1538057, -11.1685629, -5.1702642, -4.9208488, 4.9263554
21: -43.5430489, -35.0506821, -43.5265732, -35.0925751, -4.2463856, 4.2821903
22: -27.0062599, -19.5387287, -26.9941635, -19.5679531, -4.3263779, 4.3441963
23: -20.8458443, -12.5048733, -20.8106518, -12.5381031, -4.7667465, 4.7670040
24: -16.8527222, -7.6351585, -16.8205471, -7.6642041, -7.1481552, 7.1431618
25: -14.6357040, -6.9558754, -14.6149817, -6.9823456, -4.1851692, 4.1961708
26: -14.6224575, -7.8156013, -14.6132555, -7.8281240, -6.5326462, 6.5192528
27: -14.6296043, -9.5300503, -14.6137772, -9.5693331, -4.0427723, 4.0694427
28: -10.0211363, -1.4230995, -10.0053196, -1.4369662, -6.1430206, 6.1340675
29: -45.5791855, -36.8206444, -45.5594406, -36.8725281, -4.9816189, 5.0152683
30: -32.1777916, -23.0074501, -32.1551514, -23.0575542, -4.9699173, 5.0022011
31: -32.2299995, -23.5106735, -32.2015724, -23.5514526, -6.2846565, 6.3032913
32: 7.7189751, 13.6762772, 7.7291837, 13.6665688, -4.1428680, 4.1436195
33: 4.6110044, 16.3130531, 4.6608257, 16.3008499, -6.6857376, 6.6486816
34: 20.5386314, 30.9905548, 20.6045208, 30.9734135, -5.7518101, 5.7045517
35: 16.5081272, 26.8667831, 16.5743370, 26.8505173, -5.4546413, 5.4020424
36: 28.8089447, 35.1260376, 28.8407555, 35.1200638, -3.4435749, 3.4191074
37: 11.0277414, 20.1171265, 11.0745983, 20.1001778, -5.9481468, 5.9173431
38: 34.8545227, 43.6848602, 34.9219589, 43.6635361, -6.0492783, 6.0122337
39: 8.9880447, 18.5064011, 9.0340385, 18.4949074, -6.5278549, 6.4955597
40: 15.7922583, 25.1247883, 15.8269987, 25.1141224, -5.7888069, 5.7637711
41: 6.7355266, 13.2277508, 6.7555623, 13.2138157, -5.0003319, 4.9925461
42: -12.3874989, -3.4539931, -12.3766098, -3.4721537, -7.0362282, 7.0387726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6130751, upper bound: 3.6283052
time: 5.38 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6130751, upper bound: 3.6390443
time: 6.11 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -21.5673809, -8.4728556, -21.5325966, -8.4775763, -10.4414444, 10.3892708
1: -21.4303684, -12.2357578, -21.4258938, -12.2467070, -5.2778702, 5.2701950
2: -12.3958473, -5.7761631, -12.3880930, -5.7808471, -4.2767277, 4.2587738
3: -12.0143089, -4.1653881, -11.9995565, -4.1735687, -5.3678513, 5.3682632
4: -10.2951117, 0.0071352, -10.2797337, -0.0109830, -6.0371819, 6.0644550
5: -13.5599976, -4.0419989, -13.5534239, -4.0568056, -6.1425781, 6.1436653
6: -8.3206921, 0.5451202, -8.3174534, 0.5295904, -6.4487534, 6.4522781
7: -32.1576385, -22.0539093, -32.1497803, -22.0883942, -5.8273087, 5.8422470
8: -18.8183880, -9.1039915, -18.8079014, -9.1107597, -5.2145309, 5.2418480
9: -5.3360853, 1.3951790, -5.3148155, 1.3911710, -4.0472794, 4.0328846
10: -36.1393585, -27.7631989, -36.1356506, -27.7823524, -5.2506332, 5.2590675
11: -55.1344986, -44.7878532, -55.1300011, -44.8440094, -4.9470177, 4.9824619
12: -11.5797014, -4.5855045, -11.5767345, -4.6033592, -6.2187157, 6.2188187
13: 0.8828471, 8.0155029, 0.8962387, 8.0028982, -5.2956276, 5.2996902
14: -71.0865173, -57.9568710, -71.0797653, -57.9711380, -8.2561302, 8.2475967
15: -8.9212809, 0.9035616, -8.9101562, 0.8955789, -4.8852482, 4.8952599
16: -33.5593910, -23.9654236, -33.5441437, -23.9973335, -6.4532738, 6.4534035
17: -88.6814728, -72.4152374, -88.6696320, -72.4776535, -8.1707916, 8.2174034
18: -4.1793766, 1.0646970, -4.1683693, 1.0493391, -3.3978634, 3.3887367
19: -30.5270691, -23.1998634, -30.5215302, -23.2235947, -4.6487465, 4.6612301
20: -11.1730042, -5.1532845, -11.1696167, -5.1669965, -4.9225388, 4.9410782
21: -43.5490036, -35.0504112, -43.5439987, -35.0824089, -4.2639866, 4.2844944
22: -27.0085449, -19.5386906, -27.0013332, -19.5649948, -4.3324165, 4.3463211
23: -20.8538589, -12.5040054, -20.8336830, -12.5233755, -4.7909508, 4.7749615
24: -16.8622322, -7.6350894, -16.8487968, -7.6489325, -7.1730042, 7.1608582
25: -14.6393414, -6.9554410, -14.6262798, -6.9749308, -4.1973495, 4.2008915
26: -14.6229076, -7.8149238, -14.6153412, -7.8230448, -6.5363121, 6.5532074
27: -14.6336031, -9.5296965, -14.6270695, -9.5612755, -4.0548248, 4.0772419
28: -10.0246840, -1.4222549, -10.0157776, -1.4264010, -6.1432724, 6.1520920
29: -45.5859833, -36.8205948, -45.5796700, -36.8618660, -4.9987183, 5.0252514
30: -32.1884651, -23.0059490, -32.1849632, -23.0373802, -5.0042858, 5.0072651
31: -32.2389832, -23.5106621, -32.2289543, -23.5412197, -6.3037758, 6.3255386
32: 7.7172108, 13.6763563, 7.7232361, 13.6694441, -4.1509628, 4.1495457
33: 4.6069369, 16.3131485, 4.6476722, 16.3064404, -6.6938591, 6.6544075
34: 20.5330029, 30.9906311, 20.5875988, 30.9821358, -5.7662182, 5.7094421
35: 16.5018425, 26.8668022, 16.5550327, 26.8603859, -5.4707947, 5.4073982
36: 28.8081627, 35.1262741, 28.8373795, 35.1225815, -3.4460192, 3.4212980
37: 11.0223980, 20.1172295, 11.0572872, 20.1071568, -5.9611588, 5.9274597
38: 34.8535309, 43.6897278, 34.9086876, 43.6780014, -6.0537987, 6.0313377
39: 8.9874611, 18.5081730, 9.0241375, 18.4999409, -6.5295067, 6.5050430
40: 15.7889147, 25.1276188, 15.8123350, 25.1227226, -5.7991581, 5.7829418
41: 6.7310100, 13.2278471, 6.7412415, 13.2214203, -5.0125313, 5.0024109
42: -12.3878756, -3.4535041, -12.3776121, -3.4691253, -7.0387268, 7.0406113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6242638, upper bound: 3.6283052
time: 4.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6242638, upper bound: 3.6390443
time: 5.20 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.5209255, -8.4790382, -21.5742702, -8.4760332, -10.3653564, 10.4170456
1: -21.4249039, -12.2463913, -21.4298115, -12.2344933, -5.2649136, 5.2727089
2: -12.3914089, -5.7838726, -12.3923368, -5.7771916, -4.2677917, 4.2499657
3: -12.0069466, -4.1851888, -12.0015163, -4.1676588, -5.3711739, 5.3515472
4: -10.2775316, -0.0338100, -10.2880573, 0.0103753, -6.0654335, 6.0266209
5: -13.5514526, -4.0638809, -13.5609798, -4.0437541, -6.1483498, 6.1386108
6: -8.2987728, 0.5308048, -8.3355074, 0.5353110, -6.4251976, 6.4808540
7: -32.1549644, -22.0852928, -32.1571808, -22.0579567, -5.8541107, 5.8169823
8: -18.7968731, -9.1518965, -18.8098507, -9.1133995, -5.2202339, 5.1957512
9: -5.3136692, 1.3834990, -5.3389111, 1.3923056, -4.0336533, 4.0502110
10: -36.1290665, -27.7740593, -36.1374207, -27.7660809, -5.2466888, 5.2655392
11: -55.1049194, -44.8257675, -55.1231232, -44.8054047, -4.9549141, 4.9638309
12: -11.5721416, -4.6014886, -11.5778589, -4.5903192, -6.2098846, 6.2373276
13: 0.8967322, 7.9784746, 0.8873084, 8.0120316, -5.3016357, 5.2705498
14: -71.0791168, -57.9722672, -71.0850449, -57.9463272, -8.2574615, 8.2314606
15: -8.9037256, 0.8732553, -8.9136467, 0.9119773, -4.9033337, 4.8649387
16: -33.5259171, -23.9886093, -33.5556068, -23.9884987, -6.4177208, 6.4810219
17: -88.6788330, -72.4685135, -88.6785202, -72.4017868, -8.2487335, 8.1692848
18: -4.1449451, 1.0461981, -4.1698904, 1.0697668, -3.3833141, 3.3865051
19: -30.5112839, -23.2096386, -30.5232410, -23.2184563, -4.6423683, 4.6589565
20: -11.1699457, -5.1607399, -11.1731577, -5.1612396, -4.9273605, 4.9271297
21: -43.5283890, -35.0658798, -43.5420227, -35.0749626, -4.2433338, 4.2933140
22: -26.9982147, -19.5574112, -27.0052967, -19.5369835, -4.3518734, 4.3345966
23: -20.8112068, -12.5219250, -20.8504810, -12.5160561, -4.7570229, 4.7916260
24: -16.8205795, -7.6527328, -16.8564720, -7.6425881, -7.1396217, 7.1610336
25: -14.6150856, -6.9647675, -14.6390581, -6.9693112, -4.1795654, 4.2114697
26: -14.6135035, -7.8236098, -14.6211119, -7.8013487, -6.5361938, 6.5352058
27: -14.6179247, -9.5536499, -14.6291676, -9.5386095, -4.0586948, 4.0650291
28: -10.0058718, -1.4337013, -10.0213585, -1.4264121, -6.1526527, 6.1392174
29: -45.5632248, -36.8490906, -45.5786819, -36.8301430, -5.0033531, 5.0139275
30: -32.1563263, -23.0361061, -32.1773224, -23.0192814, -4.9888649, 4.9941654
31: -32.2031784, -23.5298843, -32.2358589, -23.5316010, -6.2793007, 6.3210907
32: 7.7250977, 13.6674366, 7.7048979, 13.6754122, -4.1423073, 4.1661034
33: 4.6501694, 16.3070393, 4.5998607, 16.3121223, -6.6559029, 6.7104149
34: 20.5769596, 30.9816074, 20.5573387, 30.9899406, -5.7289906, 5.7460022
35: 16.5525589, 26.8573380, 16.5156441, 26.8655968, -5.4247665, 5.4547157
36: 28.8273048, 35.1231384, 28.8092728, 35.1254501, -3.4284439, 3.4493752
37: 11.0702553, 20.1104870, 11.0130358, 20.1130066, -5.9168015, 5.9764671
38: 34.9010735, 43.6738052, 34.8613358, 43.6850891, -6.0220413, 6.0631981
39: 9.0257702, 18.5022755, 8.9717083, 18.5062637, -6.4984360, 6.5555534
40: 15.8170919, 25.1158485, 15.7825947, 25.1244984, -5.7706795, 5.8020973
41: 6.7536774, 13.2196398, 6.7203603, 13.2237968, -4.9895973, 5.0260010
42: -12.3770437, -3.4564080, -12.3994246, -3.4673042, -7.0275879, 7.0675888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 759

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6267343, upper bound: 3.6275216
time: 4.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6387520, upper bound: 3.6275715
time: 5.01 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5369644, -8.4739532, -21.5590496, -8.4782925, -10.3779373, 10.4374542
1: -21.4277534, -12.2435551, -21.4218826, -12.2389507, -5.2624512, 5.2849808
2: -12.3943310, -5.7815557, -12.3885736, -5.7766786, -4.2654991, 4.2658215
3: -12.0132160, -4.1724563, -11.9984398, -4.1679735, -5.3779068, 5.3548737
4: -10.2914238, -0.0111372, -10.2782192, 0.0063627, -6.0755959, 6.0191593
5: -13.5579529, -4.0557041, -13.5555763, -4.0459180, -6.1423378, 6.1401634
6: -8.3174925, 0.5370787, -8.3194990, 0.5216308, -6.4289055, 6.4724007
7: -32.1559944, -22.0807095, -32.1510468, -22.0621033, -5.8468246, 5.8216305
8: -18.8152065, -9.1105556, -18.8055077, -9.1052818, -5.2459431, 5.2042236
9: -5.3218436, 1.3945014, -5.3258176, 1.3923953, -4.0425873, 4.0353851
10: -36.1340904, -27.7689705, -36.1295700, -27.7769146, -5.2422676, 5.2644730
11: -55.1331482, -44.8086700, -55.1268768, -44.8239288, -4.9384212, 4.9874878
12: -11.5781345, -4.5934949, -11.5737305, -4.5947828, -6.2101288, 6.2411804
13: 0.8866342, 8.0022860, 0.8970149, 8.0147600, -5.3130455, 5.2775002
14: -71.0799026, -57.9689407, -71.0691223, -57.9589310, -8.2471428, 8.2426491
15: -8.9169559, 0.8934383, -8.9028349, 0.9051876, -4.9085655, 4.8594646
16: -33.5518913, -23.9749146, -33.5496445, -23.9888039, -6.4317169, 6.4740791
17: -88.6763687, -72.4598312, -88.6577301, -72.4350586, -8.2114220, 8.1586723
18: -4.1706972, 1.0544829, -4.1751370, 1.0590236, -3.3850098, 3.3999062
19: -30.5230846, -23.2044601, -30.5256653, -23.2192192, -4.6441994, 4.6657372
20: -11.1710033, -5.1576509, -11.1723700, -5.1657996, -4.9358025, 4.9273548
21: -43.5458984, -35.0557480, -43.5462494, -35.0776176, -4.2427750, 4.3065109
22: -27.0041695, -19.5545845, -27.0002880, -19.5494614, -4.3389053, 4.3338337
23: -20.8340702, -12.5071507, -20.8524017, -12.5209618, -4.7575989, 4.8064461
24: -16.8485641, -7.6374373, -16.8608665, -7.6493826, -7.1504250, 7.1802673
25: -14.6262360, -6.9575129, -14.6384916, -6.9742975, -4.1781311, 4.2180595
26: -14.6140556, -7.8190656, -14.6132946, -7.8217912, -6.5477753, 6.5302963
27: -14.6309071, -9.5457020, -14.6301670, -9.5457764, -4.0592079, 4.0735855
28: -10.0164242, -1.4233111, -10.0230980, -1.4275064, -6.1670418, 6.1359520
29: -45.5822067, -36.8385391, -45.5770569, -36.8443832, -4.9979477, 5.0225143
30: -32.1857872, -23.0161247, -32.1843033, -23.0279083, -4.9836693, 5.0250397
31: -32.2303238, -23.5202999, -32.2369690, -23.5381966, -6.2933311, 6.3286209
32: 7.7193985, 13.6685333, 7.7219267, 13.6664915, -4.1414852, 4.1531773
33: 4.6373215, 16.3112621, 4.6193590, 16.3031864, -6.6535378, 6.6880302
34: 20.5603142, 30.9892311, 20.5618210, 30.9825172, -5.7264328, 5.7471313
35: 16.5335140, 26.8660507, 16.5250626, 26.8584251, -5.4228840, 5.4508533
36: 28.8240242, 35.1240616, 28.8220673, 35.1175652, -3.4236813, 3.4358234
37: 11.0533352, 20.1160126, 11.0284748, 20.1054897, -5.9197159, 5.9646797
38: 34.8879471, 43.6862411, 34.8747177, 43.6769600, -6.0292587, 6.0510063
39: 9.0161800, 18.5054741, 8.9974041, 18.4972095, -6.4977493, 6.5280685
40: 15.8026934, 25.1231346, 15.7997589, 25.1189270, -5.7831726, 5.7895603
41: 6.7396655, 13.2254362, 6.7338572, 13.2144871, -4.9907265, 5.0174294
42: -12.3779230, -3.4550793, -12.3872652, -3.4760780, -7.0209961, 7.0565186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 759

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6159988, upper bound: 3.6387005
time: 5.50 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6280130, upper bound: 3.6387524
time: 5.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5376930, -8.4738283, -21.5791798, -8.4759617, -10.3849335, 10.4692535
1: -21.4283791, -12.2435322, -21.4302902, -12.2334652, -5.2678165, 5.2940674
2: -12.3946476, -5.7817926, -12.3926821, -5.7764344, -4.2679729, 4.2700729
3: -12.0133810, -4.1721830, -12.0016203, -4.1635203, -5.3821144, 5.3582230
4: -10.2927446, -0.0108049, -10.2887335, 0.0186105, -6.0913734, 6.0280094
5: -13.5583248, -4.0558357, -13.5615406, -4.0412192, -6.1496048, 6.1496162
6: -8.3178835, 0.5397983, -8.3418102, 0.5354081, -6.4401321, 6.4968376
7: -32.1566162, -22.0806408, -32.1573639, -22.0563660, -5.8541985, 5.8273830
8: -18.8156052, -9.1103334, -18.8100395, -9.0987377, -5.2550163, 5.2089767
9: -5.3217597, 1.3945855, -5.3393564, 1.3960212, -4.0462990, 4.0524807
10: -36.1354828, -27.7689552, -36.1385117, -27.7646179, -5.2516670, 5.2725525
11: -55.1343498, -44.8085480, -55.1339035, -44.8051910, -4.9584141, 4.9928837
12: -11.5789347, -4.5934706, -11.5801754, -4.5896416, -6.2105789, 6.2470970
13: 0.8862158, 8.0024757, 0.8871970, 8.0199232, -5.3201981, 5.2876091
14: -71.0824890, -57.9689560, -71.0855560, -57.9449425, -8.2610168, 8.2576447
15: -8.9191055, 0.8935528, -8.9149857, 0.9191327, -4.9281464, 4.8691692
16: -33.5526047, -23.9751205, -33.5644188, -23.9884758, -6.4333153, 6.5023537
17: -88.6806717, -72.4594727, -88.6792145, -72.4006500, -8.2522469, 8.1776619
18: -4.1712856, 1.0545788, -4.1793242, 1.0698075, -3.3964748, 3.4042530
19: -30.5230598, -23.2043571, -30.5270519, -23.2184639, -4.6452866, 4.6678963
20: -11.1709919, -5.1574793, -11.1733780, -5.1607294, -4.9420853, 4.9287968
21: -43.5457878, -35.0556488, -43.5479927, -35.0746384, -4.2456436, 4.3109207
22: -27.0054054, -19.5545044, -27.0075684, -19.5369225, -4.3539944, 4.3406391
23: -20.8342247, -12.5071869, -20.8584976, -12.5152225, -4.7649803, 4.8157978
24: -16.8487968, -7.6374607, -16.8659725, -7.6424880, -7.1573334, 7.1858826
25: -14.6263714, -6.9573689, -14.6426620, -6.9688892, -4.1842918, 4.2236500
26: -14.6155939, -7.8185163, -14.6215439, -7.8006697, -6.5701408, 6.5388565
27: -14.6312275, -9.5455866, -14.6331615, -9.5382452, -4.0664902, 4.0770779
28: -10.0163546, -1.4231263, -10.0248804, -1.4255323, -6.1707001, 6.1394691
29: -45.5834579, -36.8384247, -45.5854874, -36.8300743, -5.0133400, 5.0310307
30: -32.1861305, -23.0159416, -32.1879768, -23.0177689, -4.9939270, 5.0285034
31: -32.2305870, -23.5196495, -32.2448158, -23.5315704, -6.3014717, 6.3402290
32: 7.7191496, 13.6702995, 7.7031369, 13.6754942, -4.1482201, 4.1742153
33: 4.6369739, 16.3126640, 4.5958161, 16.3122063, -6.6615982, 6.7185364
34: 20.5600052, 30.9902821, 20.5517426, 30.9900036, -5.7338848, 5.7603989
35: 16.5332756, 26.8671913, 16.5093403, 26.8656330, -5.4301147, 5.4708729
36: 28.8239136, 35.1256332, 28.8084946, 35.1256714, -3.4306355, 3.4518108
37: 11.0529766, 20.1174698, 11.0076971, 20.1131325, -5.9269180, 5.9894753
38: 34.8878136, 43.6882324, 34.8603287, 43.6899681, -6.0411110, 6.0677414
39: 9.0158615, 18.5072956, 8.9711199, 18.5080376, -6.5079231, 6.5572205
40: 15.8024483, 25.1244373, 15.7792702, 25.1273232, -5.7898293, 5.8124790
41: 6.7393456, 13.2272472, 6.7158370, 13.2239141, -4.9994583, 5.0381851
42: -12.3781147, -3.4533587, -12.3997955, -3.4667718, -7.0294037, 7.0700531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=79, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 759

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6267343, upper bound: 3.6387005
time: 5.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6387520, upper bound: 3.6387524
time: 5.02 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 12.49 seconds
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6021840
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6129224
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6021840
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6129224
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6133573
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6240949
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6133573
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6316929, upper bound: 3.6240949
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6130751, upper bound: 3.6283052
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6130751, upper bound: 3.6390443
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6242638, upper bound: 3.6283052
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6242638, upper bound: 3.6390443
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6267343, upper bound: 3.6275216
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6387520, upper bound: 3.6275715
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6159988, upper bound: 3.6387005
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6280130, upper bound: 3.6387524
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6267343, upper bound: 3.6387005
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 12.49
Output dim: 38, lower bound: -3.6387520, upper bound: 3.6387524

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.5774193, -8.4755783, -21.5154114, -8.4837847, -10.4144974, 10.3614464
1: -21.4311619, -12.2314386, -21.4214211, -12.2495384, -5.2570915, 5.2676678
2: -12.3975639, -5.7779851, -12.3847408, -5.7834535, -4.2583580, 4.2574520
3: -12.0139265, -4.1668234, -11.9924841, -4.1866894, -5.3610344, 5.3587494
4: -10.2964830, 0.0092409, -10.2638893, -0.0341135, -6.0350189, 6.0514011
5: -13.5628777, -4.0424929, -13.5463676, -4.0656023, -6.1378136, 6.1443672
6: -8.3340788, 0.5444812, -8.2981749, 0.5200965, -6.4523735, 6.4335632
7: -32.1573143, -22.0503159, -32.1473579, -22.0931263, -5.8153496, 5.8471565
8: -18.8175259, -9.1136036, -18.7884064, -9.1524715, -5.2001629, 5.2138672
9: -5.3459353, 1.3943951, -5.3059196, 1.3800271, -4.0594196, 4.0225964
10: -36.1371956, -27.7532139, -36.1283493, -27.7876282, -5.2417355, 5.2602806
11: -55.1228943, -44.7702217, -55.0999908, -44.8612709, -4.9150753, 4.9964600
12: -11.5791569, -4.5818920, -11.5697069, -4.6115670, -6.2118073, 6.2156830
13: 0.8779631, 8.0111399, 0.9074551, 7.9787378, -5.2828712, 5.2858009
14: -71.0854568, -57.9445953, -71.0750504, -57.9745102, -8.2276039, 8.2515831
15: -8.9201355, 0.9094138, -8.8942223, 0.8751903, -4.8776894, 4.8863697
16: -33.5607605, -23.9660263, -33.5170746, -24.0112267, -6.4552307, 6.4304581
17: -88.6780167, -72.3847580, -88.6665344, -72.4869843, -8.1565094, 8.2468338
18: -4.1698179, 1.0747108, -4.1416025, 1.0408785, -3.3796902, 3.3853912
19: -30.5231094, -23.1997147, -30.5093842, -23.2288895, -4.6396732, 4.6583939
20: -11.1732283, -5.1517363, -11.1684160, -5.1708040, -4.9188576, 4.9291115
21: -43.5417519, -35.0482483, -43.5257683, -35.0926208, -4.2464848, 4.2819252
22: -27.0052662, -19.5268726, -26.9935017, -19.5679760, -4.3243771, 4.3569050
23: -20.8503113, -12.5024872, -20.8104439, -12.5388880, -4.7738075, 4.7696114
24: -16.8559036, -7.6329646, -16.8203392, -7.6652737, -7.1505661, 7.1445236
25: -14.6388073, -6.9532137, -14.6148500, -6.9828634, -4.1890526, 4.1990528
26: -14.6209946, -7.7981339, -14.6126299, -7.8283939, -6.5305786, 6.5375061
27: -14.6294298, -9.5233297, -14.6133680, -9.5693932, -4.0424862, 4.0745029
28: -10.0215321, -1.4237309, -10.0049229, -1.4373504, -6.1417274, 6.1348495
29: -45.5783005, -36.8071060, -45.5585594, -36.8725586, -4.9801903, 5.0271435
30: -32.1771812, -22.9984207, -32.1545334, -23.0576191, -4.9686508, 5.0100956
31: -32.2355270, -23.5098953, -32.2012711, -23.5523033, -6.2926941, 6.3038101
32: 7.7017322, 13.6753922, 7.7292819, 13.6660862, -4.1606026, 4.1400547
33: 4.5898843, 16.3122139, 4.6610146, 16.2999611, -6.7122841, 6.6456223
34: 20.5305176, 30.9900703, 20.6046715, 30.9726505, -5.7616577, 5.7029552
35: 16.4941978, 26.8651066, 16.5745029, 26.8496761, -5.4709778, 5.3993225
36: 28.7962513, 35.1250916, 28.8408699, 35.1195908, -3.4566231, 3.4167480
37: 11.0095043, 20.1167870, 11.0747852, 20.0997677, -5.9690781, 5.9156990
38: 34.8413696, 43.6861763, 34.9221001, 43.6628647, -6.0624199, 6.0122681
39: 8.9640541, 18.5066299, 9.0342474, 18.4942799, -6.5520401, 6.4935913
40: 15.7734003, 25.1245918, 15.8271027, 25.1135063, -5.8085060, 5.7612457
41: 6.7195225, 13.2266045, 6.7557077, 13.2132721, -5.0171585, 4.9903183
42: -12.3990030, -3.4547563, -12.3765345, -3.4727020, -7.0471497, 7.0369492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=79, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 692

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5981451, upper bound: 3.6390442
time: 5.72 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5981451, upper bound: 3.6390445
time: 5.53 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5823517, -8.4754782, -21.5322227, -8.4786081, -10.4666977, 10.3810310
1: -21.4316921, -12.2303839, -21.4248695, -12.2467203, -5.2784500, 5.2705784
2: -12.3978977, -5.7771926, -12.3879433, -5.7813721, -4.2784672, 4.2576447
3: -12.0140562, -4.1626692, -11.9989443, -4.1737208, -5.3677216, 5.3697128
4: -10.2971478, 0.0175240, -10.2791195, -0.0111005, -6.0363922, 6.0773277
5: -13.5634670, -4.0399451, -13.5532446, -4.0575395, -6.1488266, 6.1456223
6: -8.3403673, 0.5445777, -8.3173161, 0.5290911, -6.4683800, 6.4485245
7: -32.1574860, -22.0487480, -32.1489792, -22.0884457, -5.8257866, 5.8472557
8: -18.8177528, -9.0988894, -18.8071651, -9.1108751, -5.2134190, 5.2486362
9: -5.3463721, 1.3981118, -5.3140244, 1.3911289, -4.0616894, 4.0352459
10: -36.1382904, -27.7517624, -36.1347656, -27.7825069, -5.2487488, 5.2652454
11: -55.1336555, -44.7699356, -55.1294479, -44.8440933, -4.9441280, 4.9999523
12: -11.5814342, -4.5812030, -11.5765228, -4.6035242, -6.2215843, 6.2163734
13: 0.8778406, 8.0190048, 0.8969458, 8.0027332, -5.2999573, 5.3043518
14: -71.0859375, -57.9431458, -71.0784454, -57.9712143, -8.2537994, 8.2551231
15: -8.9214668, 0.9165831, -8.9095573, 0.8954868, -4.8819084, 4.9112473
16: -33.5695686, -23.9660187, -33.5437889, -23.9977322, -6.4765511, 6.4460945
17: -88.6787109, -72.3836136, -88.6683655, -72.4779510, -8.1648941, 8.2503548
18: -4.1792688, 1.0747476, -4.1679564, 1.0492873, -3.3974190, 3.3985939
19: -30.5269508, -23.1997128, -30.5211697, -23.2236271, -4.6486187, 4.6613274
20: -11.1734610, -5.1512146, -11.1694775, -5.1675525, -4.9205246, 4.9438515
21: -43.5477142, -35.0479736, -43.5432014, -35.0824432, -4.2640991, 4.2842369
22: -27.0075207, -19.5268173, -27.0006866, -19.5650787, -4.3304157, 4.3590336
23: -20.8583431, -12.5016270, -20.8334656, -12.5241470, -4.7980118, 4.7775764
24: -16.8653946, -7.6328974, -16.8485985, -7.6499810, -7.1754227, 7.1622276
25: -14.6424198, -6.9527712, -14.6261282, -6.9754505, -4.2012367, 4.2037830
26: -14.6214590, -7.7974482, -14.6147337, -7.8233356, -6.5342369, 6.5714493
27: -14.6334267, -9.5229588, -14.6266670, -9.5613165, -4.0545387, 4.0822926
28: -10.0250711, -1.4228759, -10.0153913, -1.4267882, -6.1419830, 6.1528740
29: -45.5851059, -36.8070641, -45.5788078, -36.8619080, -4.9972763, 5.0371304
30: -32.1878395, -22.9968910, -32.1843643, -23.0374603, -5.0030193, 5.0151577
31: -32.2444916, -23.5098534, -32.2286644, -23.5420475, -6.3118019, 6.3260498
32: 7.6999569, 13.6754742, 7.7233324, 13.6689510, -4.1687069, 4.1459770
33: 4.5858126, 16.3123379, 4.6478586, 16.3055534, -6.7204037, 6.6513405
34: 20.5249023, 30.9901314, 20.5877304, 30.9813499, -5.7760792, 5.7078514
35: 16.4879417, 26.8651485, 16.5551949, 26.8595543, -5.4871349, 5.4046650
36: 28.7954788, 35.1252975, 28.8374710, 35.1221008, -3.4590626, 3.4189396
37: 11.0041695, 20.1168900, 11.0574627, 20.1067410, -5.9820824, 5.9258194
38: 34.8403511, 43.6910400, 34.9088440, 43.6773491, -6.0669632, 6.0313797
39: 8.9634571, 18.5084496, 9.0243187, 18.4993172, -6.5537109, 6.5030746
40: 15.7700653, 25.1274223, 15.8124666, 25.1221275, -5.8188572, 5.7804146
41: 6.7150149, 13.2267151, 6.7413602, 13.2208605, -5.0293541, 5.0001831
42: -12.3993721, -3.4542549, -12.3775606, -3.4696922, -7.0496178, 7.0387497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=79, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 692

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6093243, upper bound: 3.6390442
time: 4.92 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6093243, upper bound: 3.6390445
time: 5.28 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -21.5277729, -8.4791651, -21.5739384, -8.4762478, -10.3951645, 10.4046364
1: -21.4255257, -12.2443829, -21.4297314, -12.2348127, -5.2705765, 5.2712975
2: -12.3922672, -5.7810388, -12.3922424, -5.7772055, -4.2762489, 4.2457352
3: -12.0072718, -4.1775889, -12.0013494, -4.1677294, -5.3712959, 5.3590584
4: -10.2786503, -0.0133619, -10.2876682, 0.0103471, -6.0657349, 6.0467644
5: -13.5533447, -4.0576468, -13.5607939, -4.0438042, -6.1552696, 6.1324654
6: -8.3121138, 0.5306234, -8.3354206, 0.5350853, -6.4384766, 6.4803276
7: -32.1554108, -22.0873737, -32.1571388, -22.0597572, -5.8566475, 5.8189926
8: -18.7964077, -9.1169167, -18.8093643, -9.1134300, -5.2194290, 5.2301865
9: -5.3141351, 1.3922377, -5.3387394, 1.3922557, -4.0340233, 4.0586472
10: -36.1313972, -27.7692013, -36.1372986, -27.7661667, -5.2499771, 5.2657242
11: -55.1395721, -44.8267136, -55.1231232, -44.8059959, -4.9893532, 4.9623222
12: -11.5739994, -4.5985098, -11.5778484, -4.5905252, -6.2084846, 6.2395973
13: 0.8970466, 7.9955969, 0.8875777, 8.0119381, -5.3011513, 5.2878876
14: -71.0816803, -57.9691544, -71.0849380, -57.9466209, -8.2695618, 8.2299957
15: -8.9066086, 0.8943667, -8.9132538, 0.9119458, -4.9057903, 4.8857307
16: -33.5479546, -23.9895477, -33.5555267, -23.9888763, -6.4397697, 6.4789963
17: -88.6786804, -72.4691925, -88.6784210, -72.4033661, -8.2465553, 8.1692696
18: -4.1674995, 1.0455945, -4.1698422, 1.0695345, -3.4054546, 3.3857517
19: -30.5185642, -23.2101288, -30.5231590, -23.2186623, -4.6498299, 4.6578789
20: -11.1707935, -5.1620684, -11.1730700, -5.1621928, -4.9243202, 4.9303226
21: -43.5440063, -35.0659370, -43.5419540, -35.0752220, -4.2586708, 4.2931232
22: -26.9995594, -19.5569420, -27.0049362, -19.5369949, -4.3532982, 4.3332844
23: -20.8293667, -12.5206909, -20.8504143, -12.5163660, -4.7752876, 4.7933311
24: -16.8385201, -7.6535902, -16.8563786, -7.6429176, -7.1574516, 7.1600266
25: -14.6173496, -6.9639912, -14.6389856, -6.9693871, -4.1822796, 4.2121620
26: -14.6142750, -7.8227863, -14.6209040, -7.8018942, -6.5276527, 6.5518646
27: -14.6270723, -9.5533352, -14.6290674, -9.5388050, -4.0668602, 4.0650215
28: -10.0189753, -1.4325309, -10.0212727, -1.4266708, -6.1503677, 6.1421013
29: -45.5752792, -36.8496742, -45.5786400, -36.8303299, -5.0155525, 5.0123787
30: -32.1880188, -23.0337257, -32.1772575, -23.0197811, -5.0201797, 4.9968834
31: -32.2197418, -23.5305481, -32.2357140, -23.5317993, -6.2972221, 6.3190804
32: 7.7207012, 13.6675434, 7.7049236, 13.6753330, -4.1468163, 4.1660519
33: 4.6491394, 16.3075733, 4.6007371, 16.3120670, -6.6587086, 6.7108612
34: 20.5660400, 30.9813690, 20.5573845, 30.9897461, -5.7397213, 5.7455769
35: 16.5405788, 26.8566761, 16.5157051, 26.8653603, -5.4364281, 5.4538841
36: 28.8266106, 35.1235809, 28.8093376, 35.1253853, -3.4277229, 3.4495335
37: 11.0626106, 20.1109428, 11.0131302, 20.1129093, -5.9245491, 5.9765320
38: 34.8991852, 43.6870308, 34.8616829, 43.6850471, -6.0240593, 6.0761871
39: 9.0260601, 18.5136757, 8.9723339, 18.5062275, -6.4981079, 6.5638924
40: 15.8090611, 25.1198540, 15.7827301, 25.1244202, -5.7768822, 5.8093071
41: 6.7459598, 13.2198524, 6.7204127, 13.2236252, -4.9972954, 5.0260925
42: -12.3763733, -3.4545541, -12.3991203, -3.4673467, -7.0249825, 7.0711746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 692

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6238066, upper bound: 3.6275713
time: 6.17 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6238066, upper bound: 3.6275717
time: 8.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.5345383, -8.4768724, -21.5588398, -8.4785185, -10.3759003, 10.4274597
1: -21.4266586, -12.2439270, -21.4218006, -12.2389641, -5.2618370, 5.2824173
2: -12.3921747, -5.7815905, -12.3884106, -5.7766824, -4.2646255, 4.2627754
3: -12.0095558, -4.1736736, -11.9981718, -4.1680717, -5.3740807, 5.3535385
4: -10.2818174, -0.0115393, -10.2774391, 0.0063066, -6.0656509, 6.0179424
5: -13.5533333, -4.0567513, -13.5552254, -4.0459938, -6.1396179, 6.1357994
6: -8.3164396, 0.5315919, -8.3194170, 0.5212067, -6.4275513, 6.4663811
7: -32.1554337, -22.0813980, -32.1510086, -22.0621567, -5.8463364, 5.8190746
8: -18.8031425, -9.1112747, -18.8045387, -9.1053104, -5.2337856, 5.2028770
9: -5.3177662, 1.3936148, -5.3255043, 1.3923299, -4.0386200, 4.0341663
10: -36.1309204, -27.7705936, -36.1293106, -27.7770500, -5.2397614, 5.2626534
11: -55.1326218, -44.8232040, -55.1268616, -44.8251114, -4.9369202, 4.9727230
12: -11.5779095, -4.5980229, -11.5737267, -4.5951328, -6.2077980, 6.2365837
13: 0.8929712, 8.0001097, 0.8975192, 8.0145922, -5.3064117, 5.2750359
14: -71.0783997, -57.9727173, -71.0689545, -57.9592514, -8.2456169, 8.2359047
15: -8.9068413, 0.8929195, -8.9020157, 0.9051480, -4.8980198, 4.8582726
16: -33.5505142, -23.9842873, -33.5495262, -23.9895630, -6.4300957, 6.4642906
17: -88.6759262, -72.4635162, -88.6576996, -72.4353333, -8.2091026, 8.1535034
18: -4.1697173, 1.0489702, -4.1750507, 1.0585871, -3.3834286, 3.3943481
19: -30.5213833, -23.2079086, -30.5255318, -23.2194939, -4.6421871, 4.6621265
20: -11.1697483, -5.1601815, -11.1722479, -5.1659966, -4.9330635, 4.9262714
21: -43.5446892, -35.0626183, -43.5461540, -35.0781708, -4.2409973, 4.2995892
22: -27.0020714, -19.5547619, -27.0000820, -19.5494766, -4.3364201, 4.3325253
23: -20.8327370, -12.5149822, -20.8522987, -12.5215769, -4.7560272, 4.7985153
24: -16.8466263, -7.6457438, -16.8607025, -7.6500683, -7.1481285, 7.1718445
25: -14.6245155, -6.9594183, -14.6383371, -6.9744420, -4.1762867, 4.2159176
26: -14.6104231, -7.8216486, -14.6130056, -7.8219995, -6.5390549, 6.5285263
27: -14.6286659, -9.5506878, -14.6299801, -9.5461693, -4.0563908, 4.0682316
28: -10.0147409, -1.4306835, -10.0229597, -1.4280888, -6.1645470, 6.1311951
29: -45.5812645, -36.8433838, -45.5769806, -36.8447571, -4.9971046, 5.0176430
30: -32.1851959, -23.0294609, -32.1842461, -23.0289650, -4.9820938, 5.0117970
31: -32.2275772, -23.5253029, -32.2367363, -23.5386009, -6.2906227, 6.3227348
32: 7.7199526, 13.6664362, 7.7219481, 13.6663036, -4.1408310, 4.1509953
33: 4.6390786, 16.3105469, 4.6194725, 16.3031425, -6.6506634, 6.6872330
34: 20.5616398, 30.9840603, 20.5619202, 30.9821472, -5.7247276, 5.7421989
35: 16.5353966, 26.8602962, 16.5252037, 26.8579674, -5.4205322, 5.4449921
36: 28.8254147, 35.1228485, 28.8221664, 35.1174698, -3.4209290, 3.4344778
37: 11.0551834, 20.1129417, 11.0286064, 20.1052284, -5.9175911, 5.9615288
38: 34.8968620, 43.6855736, 34.8754425, 43.6769066, -6.0202217, 6.0497856
39: 9.0257816, 18.5051003, 8.9981813, 18.4971561, -6.4882889, 6.5269775
40: 15.8056822, 25.1223125, 15.8000183, 25.1188850, -5.7792969, 5.7886829
41: 6.7408323, 13.2213011, 6.7339487, 13.2141600, -4.9894524, 5.0132217
42: -12.3772259, -3.4559534, -12.3872070, -3.4761665, -7.0182304, 7.0542984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 692

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6010589, upper bound: 3.6387003
time: 5.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6010589, upper bound: 3.6387007
time: 4.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.5438328, -8.4740868, -21.5587425, -8.4784651, -10.4077682, 10.4250412
1: -21.4284210, -12.2416077, -21.4218254, -12.2392330, -5.2680969, 5.2835560
2: -12.3951683, -5.7787285, -12.3884659, -5.7766848, -4.2739487, 4.2615776
3: -12.0135393, -4.1648407, -11.9982824, -4.1680274, -5.3780327, 5.3623734
4: -10.2925167, 0.0093410, -10.2778234, 0.0063276, -6.0758934, 6.0393047
5: -13.5598183, -4.0494342, -13.5553799, -4.0459661, -6.1492386, 6.1340103
6: -8.3307953, 0.5369194, -8.3194370, 0.5214353, -6.4421997, 6.4718781
7: -32.1564026, -22.0827904, -32.1510048, -22.0639114, -5.8493538, 5.8236256
8: -18.8147297, -9.0755329, -18.8050289, -9.1052895, -5.2451286, 5.2387352
9: -5.3223023, 1.4032469, -5.3256531, 1.3923633, -4.0429440, 4.0438271
10: -36.1363983, -27.7641029, -36.1294212, -27.7770004, -5.2454872, 5.2646465
11: -55.1678085, -44.8096695, -55.1268539, -44.8244934, -4.9728546, 4.9859734
12: -11.5799837, -4.5905080, -11.5737143, -4.5949602, -6.2086906, 6.2434273
13: 0.8869425, 8.0194578, 0.8972666, 8.0146790, -5.3125496, 5.2948761
14: -71.0824738, -57.9658546, -71.0690079, -57.9591789, -8.2592049, 8.2411118
15: -8.9197760, 0.9144998, -8.9024172, 0.9051495, -4.9110088, 4.8802528
16: -33.5739479, -23.9758511, -33.5495491, -23.9891968, -6.4537811, 6.4720573
17: -88.6762390, -72.4605179, -88.6576691, -72.4365997, -8.2092514, 8.1586227
18: -4.1933627, 1.0538881, -4.1751013, 1.0587928, -3.4072800, 3.3991470
19: -30.5304012, -23.2049828, -30.5255947, -23.2194099, -4.6517181, 4.6646633
20: -11.1718483, -5.1589832, -11.1722870, -5.1667690, -4.9327621, 4.9305553
21: -43.5615158, -35.0558472, -43.5461655, -35.0778885, -4.2581158, 4.3063126
22: -27.0055256, -19.5541363, -26.9999275, -19.5494804, -4.3403168, 4.3325291
23: -20.8522472, -12.5059528, -20.8523159, -12.5212593, -4.7758656, 4.8081093
24: -16.8665333, -7.6382618, -16.8607826, -7.6497402, -7.1682434, 7.1792755
25: -14.6285715, -6.9567747, -14.6384068, -6.9743786, -4.1808567, 4.2187366
26: -14.6148262, -7.8182378, -14.6130896, -7.8223605, -6.5392189, 6.5469208
27: -14.6400490, -9.5454121, -14.6300573, -9.5459747, -4.0673828, 4.0735626
28: -10.0295343, -1.4221704, -10.0229998, -1.4277856, -6.1647148, 6.1387749
29: -45.5942459, -36.8391380, -45.5769958, -36.8445587, -5.0101395, 5.0209694
30: -32.2174835, -23.0137482, -32.1842575, -23.0284424, -5.0149803, 5.0277348
31: -32.2469139, -23.5209351, -32.2368240, -23.5383949, -6.3113098, 6.3266182
32: 7.7150021, 13.6686230, 7.7219367, 13.6663971, -4.1459961, 4.1531315
33: 4.6363187, 16.3117867, 4.6202269, 16.3031616, -6.6563129, 6.6884689
34: 20.5493851, 30.9890480, 20.5618725, 30.9823303, -5.7371521, 5.7467251
35: 16.5215511, 26.8654633, 16.5251503, 26.8581791, -5.4345474, 5.4500217
36: 28.8233185, 35.1245041, 28.8221283, 35.1174965, -3.4229622, 3.4359598
37: 11.0457001, 20.1164455, 11.0285521, 20.1053314, -5.9275131, 5.9647522
38: 34.8861008, 43.6995010, 34.8750839, 43.6769066, -6.0312424, 6.0640030
39: 9.0164528, 18.5168571, 8.9980259, 18.4971504, -6.4974289, 6.5363960
40: 15.7948036, 25.1271667, 15.7999210, 25.1188889, -5.7893105, 5.7967606
41: 6.7319651, 13.2256451, 6.7339144, 13.2143307, -4.9984207, 5.0175323
42: -12.3772430, -3.4532442, -12.3869543, -3.4761412, -7.0184174, 7.0601425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 692

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6130691, upper bound: 3.6387523
time: 4.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6130691, upper bound: 3.6387526
time: 4.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5353146, -8.4767933, -21.5789490, -8.4762316, -10.3829193, 10.4592285
1: -21.4272881, -12.2439003, -21.4302216, -12.2334919, -5.2672195, 5.2914982
2: -12.3924866, -5.7818208, -12.3925095, -5.7764239, -4.2670956, 4.2670364
3: -12.0097122, -4.1733861, -12.0013351, -4.1636434, -5.3782730, 5.3568802
4: -10.2830963, -0.0112605, -10.2879534, 0.0185672, -6.0814095, 6.0268097
5: -13.5536747, -4.0568838, -13.5611811, -4.0412560, -6.1468811, 6.1452599
6: -8.3168516, 0.5342963, -8.3417082, 0.5349692, -6.4387512, 6.4908218
7: -32.1560593, -22.0813255, -32.1573257, -22.0564098, -5.8537140, 5.8248577
8: -18.8035469, -9.1110649, -18.8090630, -9.0987835, -5.2428551, 5.2076111
9: -5.3176966, 1.3937058, -5.3390346, 1.3959172, -4.0423317, 4.0512638
10: -36.1322746, -27.7705536, -36.1382751, -27.7647781, -5.2491570, 5.2707310
11: -55.1338501, -44.8230743, -55.1338654, -44.8063507, -4.9569168, 4.9781284
12: -11.5787373, -4.5980024, -11.5801601, -4.5899982, -6.2082710, 6.2424927
13: 0.8925767, 8.0003023, 0.8877168, 8.0197363, -5.3135376, 5.2851334
14: -71.0809784, -57.9727592, -71.0854340, -57.9452362, -8.2594986, 8.2509003
15: -8.9089851, 0.8930793, -8.9141827, 0.9191141, -4.9175930, 4.8679829
16: -33.5512314, -23.9844704, -33.5643158, -23.9892273, -6.4316978, 6.4925728
17: -88.6802063, -72.4632111, -88.6791611, -72.4009247, -8.2499313, 8.1725044
18: -4.1702862, 1.0490849, -4.1792436, 1.0693593, -3.3948956, 3.3987007
19: -30.5213509, -23.2077847, -30.5269356, -23.2187386, -4.6432686, 4.6642876
20: -11.1697426, -5.1600046, -11.1732922, -5.1609373, -4.9393539, 4.9277306
21: -43.5445709, -35.0625229, -43.5478897, -35.0751839, -4.2438698, 4.3040066
22: -27.0032635, -19.5546761, -27.0073967, -19.5369301, -4.3515053, 4.3393230
23: -20.8328896, -12.5150146, -20.8583813, -12.5158310, -4.7634029, 4.8078880
24: -16.8468723, -7.6457615, -16.8658028, -7.6431603, -7.1550217, 7.1774445
25: -14.6246252, -6.9592524, -14.6425447, -6.9690337, -4.1824493, 4.2215080
26: -14.6119671, -7.8211107, -14.6212482, -7.8008909, -6.5613937, 6.5371170
27: -14.6289959, -9.5505581, -14.6329861, -9.5386496, -4.0636635, 4.0717278
28: -10.0146828, -1.4305075, -10.0247583, -1.4261032, -6.1682281, 6.1347313
29: -45.5825424, -36.8432465, -45.5854187, -36.8304672, -5.0125065, 5.0261517
30: -32.1855431, -23.0292664, -32.1879272, -23.0188408, -4.9923668, 5.0152702
31: -32.2277985, -23.5246544, -32.2446175, -23.5319824, -6.2987747, 6.3343353
32: 7.7197127, 13.6681967, 7.7031798, 13.6753407, -4.1475639, 4.1720390
33: 4.6387043, 16.3119335, 4.5959463, 16.3121414, -6.6587315, 6.7177544
34: 20.5613174, 30.9850807, 20.5518188, 30.9896049, -5.7321815, 5.7554626
35: 16.5351677, 26.8614769, 16.5094833, 26.8651733, -5.4277668, 5.4650097
36: 28.8253136, 35.1244392, 28.8086014, 35.1255798, -3.4278831, 3.4504528
37: 11.0547905, 20.1143970, 11.0078173, 20.1128883, -5.9247971, 5.9863281
38: 34.8967209, 43.6875801, 34.8610001, 43.6899033, -6.0320892, 6.0665283
39: 9.0254383, 18.5069637, 8.9718971, 18.5080242, -6.4984703, 6.5561066
40: 15.8054256, 25.1236000, 15.7794914, 25.1272430, -5.7859516, 5.8116112
41: 6.7404895, 13.2231178, 6.7159405, 13.2235718, -4.9981842, 5.0340004
42: -12.3774185, -3.4542308, -12.3997412, -3.4668765, -7.0266266, 7.0678062

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 692

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6117935, upper bound: 3.6387003
time: 5.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6117935, upper bound: 3.6387007
time: 6.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5445709, -8.4739914, -21.5788364, -8.4761477, -10.4147797, 10.4567947
1: -21.4290485, -12.2415981, -21.4302464, -12.2337809, -5.2734795, 5.2926521
2: -12.3954668, -5.7789497, -12.3925924, -5.7764411, -4.2764244, 4.2658234
3: -12.0137177, -4.1645889, -12.0014753, -4.1635675, -5.3822403, 5.3657265
4: -10.2938023, 0.0096242, -10.2883358, 0.0185593, -6.0916557, 6.0481701
5: -13.5601473, -4.0495539, -13.5613480, -4.0412598, -6.1564980, 6.1434441
6: -8.3312378, 0.5396384, -8.3417463, 0.5351982, -6.4534416, 6.4963264
7: -32.1570435, -22.0827141, -32.1573105, -22.0581741, -5.8567429, 5.8294010
8: -18.8151493, -9.0752993, -18.8095531, -9.0987616, -5.2542038, 5.2434731
9: -5.3222303, 1.4033334, -5.3391724, 1.3959734, -4.0466576, 4.0609226
10: -36.1377678, -27.7641125, -36.1383896, -27.7647209, -5.2548752, 5.2727356
11: -55.1690178, -44.8095512, -55.1338654, -44.8057365, -4.9928570, 4.9913712
12: -11.5808086, -4.5905018, -11.5801506, -4.5898290, -6.2091827, 6.2493553
13: 0.8865424, 8.0196543, 0.8874736, 8.0198164, -5.3196945, 5.3049622
14: -71.0850601, -57.9658928, -71.0854797, -57.9451675, -8.2731018, 8.2561150
15: -8.9219294, 0.9146500, -8.9145422, 0.9191003, -4.9305992, 4.8899651
16: -33.5746613, -23.9760342, -33.5643387, -23.9888229, -6.4553833, 6.5003433
17: -88.6805115, -72.4601822, -88.6791153, -72.4022141, -8.2500763, 8.1776009
18: -4.1939507, 1.0539906, -4.1792841, 1.0695794, -3.4187489, 3.4034977
19: -30.5303860, -23.2048721, -30.5269947, -23.2186470, -4.6527939, 4.6668205
20: -11.1718378, -5.1587934, -11.1733112, -5.1616983, -4.9390488, 4.9320145
21: -43.5613937, -35.0557594, -43.5479164, -35.0749130, -4.2609692, 4.3107300
22: -27.0067329, -19.5540409, -27.0072079, -19.5369492, -4.3554058, 4.3393250
23: -20.8523865, -12.5059910, -20.8584232, -12.5155077, -4.7832489, 4.8174801
24: -16.8667717, -7.6382875, -16.8658829, -7.6428485, -7.1751518, 7.1848679
25: -14.6286345, -6.9565859, -14.6425915, -6.9689493, -4.1870155, 4.2243271
26: -14.6163616, -7.8176880, -14.6213617, -7.8012357, -6.5615692, 6.5555153
27: -14.6403618, -9.5452929, -14.6330690, -9.5384378, -4.0746555, 4.0770569
28: -10.0294352, -1.4219750, -10.0247898, -1.4258102, -6.1683807, 6.1423073
29: -45.5954819, -36.8390121, -45.5854340, -36.8302689, -5.0255337, 5.0294857
30: -32.2178268, -23.0135956, -32.1879120, -23.0182991, -5.0252438, 5.0312042
31: -32.2471466, -23.5202789, -32.2446861, -23.5317993, -6.3194733, 6.3382111
32: 7.7147593, 13.6704035, 7.7031670, 13.6754227, -4.1527271, 4.1741676
33: 4.6360044, 16.3132401, 4.5967021, 16.3121681, -6.6643791, 6.7189713
34: 20.5491028, 30.9900665, 20.5517845, 30.9898033, -5.7446098, 5.7599850
35: 16.5213032, 26.8666344, 16.5094433, 26.8653927, -5.4417801, 5.4700375
36: 28.8232079, 35.1260757, 28.8085480, 35.1256065, -3.4299202, 3.4519367
37: 11.0453224, 20.1179237, 11.0077763, 20.1130028, -5.9347038, 5.9895325
38: 34.8859596, 43.7014732, 34.8606682, 43.6899071, -6.0430984, 6.0807304
39: 9.0161476, 18.5187016, 8.9717169, 18.5080185, -6.5075874, 6.5655327
40: 15.7945385, 25.1284676, 15.7793922, 25.1272831, -5.7959461, 5.8196831
41: 6.7316031, 13.2274380, 6.7159047, 13.2237406, -5.0071411, 5.0382843
42: -12.3774128, -3.4515114, -12.3994980, -3.4668739, -7.0268021, 7.0736504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=79, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 692

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6238066, upper bound: 3.6387523
time: 4.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6238066, upper bound: 3.6387526
time: 5.29 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 12.09 seconds
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.09
Output dim: 38, lower bound: -3.5981451, upper bound: 3.6390442
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.09
Output dim: 38, lower bound: -3.5981451, upper bound: 3.6390445
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.09
Output dim: 38, lower bound: -3.6093243, upper bound: 3.6390442
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.09
Output dim: 38, lower bound: -3.6093243, upper bound: 3.6390445
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 12.09
Output dim: 38, lower bound: -3.6238066, upper bound: 3.6275713
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 12.09
Output dim: 38, lower bound: -3.6238066, upper bound: 3.6275717
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.09
Output dim: 38, lower bound: -3.6010589, upper bound: 3.6387003
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.09
Output dim: 38, lower bound: -3.6010589, upper bound: 3.6387007
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.09
Output dim: 38, lower bound: -3.6130691, upper bound: 3.6387523
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.09
Output dim: 38, lower bound: -3.6130691, upper bound: 3.6387526
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 12.09
Output dim: 38, lower bound: -3.6117935, upper bound: 3.6387003
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 12.09
Output dim: 38, lower bound: -3.6117935, upper bound: 3.6387007
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 12.09
Output dim: 38, lower bound: -3.6238066, upper bound: 3.6387523
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 12.09
Output dim: 38, lower bound: -3.6238066, upper bound: 3.6387526

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -21.5774193, -8.4755783, -21.5141220, -8.4840651, -10.4138107, 10.3598862
1: -21.4311619, -12.2314386, -21.4195080, -12.2499065, -5.2563515, 5.2668495
2: -12.3975639, -5.7779851, -12.3845434, -5.7862048, -4.2547226, 4.2591114
3: -12.0139265, -4.1668234, -11.9924097, -4.1952343, -5.3546562, 5.3591881
4: -10.2964830, 0.0092409, -10.2633467, -0.0383683, -6.0309448, 6.0514793
5: -13.5628777, -4.0424929, -13.5460224, -4.0675769, -6.1357193, 6.1442680
6: -8.3340788, 0.5444812, -8.2913179, 0.5198810, -6.4503822, 6.4282875
7: -32.1573143, -22.0503159, -32.1412392, -22.0937271, -5.8108711, 5.8427296
8: -18.8175259, -9.1136036, -18.7882042, -9.1581917, -5.1964836, 5.2135582
9: -5.3459353, 1.3943951, -5.3051481, 1.3797369, -4.0548229, 4.0221405
10: -36.1371956, -27.7532139, -36.1191063, -27.7885342, -5.2426529, 5.2527084
11: -55.1228943, -44.7702217, -55.0803757, -44.8616257, -4.9228516, 4.9759235
12: -11.5791569, -4.5818920, -11.5637331, -4.6123877, -6.2104950, 6.2120552
13: 0.8779631, 8.0111399, 0.9078093, 7.9732375, -5.2760391, 5.2877922
14: -71.0854568, -57.9445953, -71.0727081, -57.9754677, -8.2224426, 8.2505035
15: -8.9201355, 0.9094138, -8.8935623, 0.8702497, -4.8698597, 4.8907738
16: -33.5607605, -23.9660263, -33.5022507, -24.0114250, -6.4525261, 6.4199791
17: -88.6780167, -72.3847580, -88.6550751, -72.4885406, -8.1549377, 8.2383575
18: -4.1698179, 1.0747108, -4.1386042, 1.0407076, -3.3794651, 3.3826122
19: -30.5231094, -23.1997147, -30.4955006, -23.2289085, -4.6421032, 4.6464348
20: -11.1732283, -5.1517363, -11.1623306, -5.1709800, -4.9183617, 4.9226875
21: -43.5417519, -35.0482483, -43.5081940, -35.0927582, -4.2552834, 4.2614536
22: -27.0052662, -19.5268726, -26.9868965, -19.5681381, -4.3241177, 4.3520927
23: -20.8503113, -12.5024872, -20.8032837, -12.5393429, -4.7754707, 4.7623596
24: -16.8559036, -7.6329646, -16.8159485, -7.6655865, -7.1499023, 7.1412697
25: -14.6388073, -6.9532137, -14.6053009, -6.9831839, -4.1924667, 4.1889324
26: -14.6209946, -7.7981339, -14.6103344, -7.8288383, -6.5303574, 6.5356865
27: -14.6294298, -9.5233297, -14.6036873, -9.5696602, -4.0455799, 4.0645313
28: -10.0215321, -1.4237309, -10.0021486, -1.4376150, -6.1379204, 6.1323051
29: -45.5783005, -36.8071060, -45.5453110, -36.8727379, -4.9827900, 5.0166664
30: -32.1771812, -22.9984207, -32.1434097, -23.0583382, -4.9700260, 5.0007839
31: -32.2355270, -23.5098953, -32.1856918, -23.5523949, -6.2982941, 6.2876663
32: 7.7017322, 13.6753922, 7.7296891, 13.6654768, -4.1602306, 4.1389771
33: 4.5898843, 16.3122139, 4.6627741, 16.2968102, -6.7106171, 6.6408844
34: 20.5305176, 30.9900703, 20.6052589, 30.9541740, -5.7454453, 5.7030945
35: 16.4941978, 26.8651066, 16.5755424, 26.8365479, -5.4583931, 5.4003506
36: 28.7962513, 35.1250916, 28.8415623, 35.1106453, -3.4489422, 3.4171162
37: 11.0095043, 20.1167870, 11.0777826, 20.0996113, -5.9696121, 5.9119148
38: 34.8413696, 43.6861763, 34.9233360, 43.6487198, -6.0531311, 6.0074348
39: 8.9640541, 18.5066299, 9.0358610, 18.4938107, -6.5513000, 6.4888992
40: 15.7734003, 25.1245918, 15.8290291, 25.1118526, -5.8082199, 5.7584705
41: 6.7195225, 13.2266045, 6.7578902, 13.2131519, -5.0168648, 4.9886360
42: -12.3990030, -3.4547563, -12.3679123, -3.4730372, -7.0459633, 7.0312996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=78, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 759

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5858404, upper bound: 3.6387045
time: 5.76 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5978590, upper bound: 3.6387564
time: 5.13 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -21.5774193, -8.4755783, -21.5185966, -8.4833250, -10.4139938, 10.3636322
1: -21.4311619, -12.2314386, -21.4228134, -12.2464447, -5.2578621, 5.2682819
2: -12.3975639, -5.7779851, -12.3899460, -5.7842269, -4.2505665, 4.2584267
3: -12.0139265, -4.1668234, -12.0049181, -4.1858306, -5.3516045, 5.3599892
4: -10.2964830, 0.0092409, -10.2723017, -0.0352237, -6.0282669, 6.0532837
5: -13.5628777, -4.0424929, -13.5482960, -4.0643587, -6.1382942, 6.1460991
6: -8.3340788, 0.5444812, -8.2967215, 0.5292557, -6.4530487, 6.4273720
7: -32.1573143, -22.0503159, -32.1474991, -22.0854797, -5.8150902, 5.8461781
8: -18.8175259, -9.1136036, -18.7961216, -9.1526260, -5.1964035, 5.2148304
9: -5.3459353, 1.3943951, -5.3129764, 1.3821331, -4.0593262, 4.0333290
10: -36.1371956, -27.7532139, -36.1280899, -27.7747650, -5.2426910, 5.2492428
11: -55.1228943, -44.7702217, -55.0997314, -44.8261032, -4.9206772, 4.9568672
12: -11.5791569, -4.5818920, -11.5709639, -4.6031332, -6.2132187, 6.2144203
13: 0.8779631, 8.0111399, 0.8980991, 7.9778433, -5.2756538, 5.2912827
14: -71.0854568, -57.9445953, -71.0754623, -57.9727249, -8.2274017, 8.2551193
15: -8.9201355, 0.9094138, -8.9007168, 0.8726435, -4.8636227, 4.8866444
16: -33.5607605, -23.9660263, -33.5222969, -23.9887657, -6.4553452, 6.4193268
17: -88.6780167, -72.3847580, -88.6660385, -72.4699707, -8.1575203, 8.2356834
18: -4.1698179, 1.0747108, -4.1415401, 1.0458076, -3.3831825, 3.3845806
19: -30.5231094, -23.1997147, -30.5092697, -23.2101421, -4.6397686, 4.6393108
20: -11.1732283, -5.1517363, -11.1684799, -5.1612892, -4.9196930, 4.9209538
21: -43.5417519, -35.0482483, -43.5255051, -35.0659447, -4.2472439, 4.2448158
22: -27.0052662, -19.5268726, -26.9934692, -19.5578995, -4.3264027, 4.3505383
23: -20.8503113, -12.5024872, -20.8102932, -12.5253496, -4.7766323, 4.7575207
24: -16.8559036, -7.6329646, -16.8197479, -7.6556597, -7.1545334, 7.1411095
25: -14.6388073, -6.9532137, -14.6145802, -6.9667606, -4.1928768, 4.1821270
26: -14.6209946, -7.7981339, -14.6125088, -7.8251934, -6.5329437, 6.5365906
27: -14.6294298, -9.5233297, -14.6136303, -9.5541058, -4.0459042, 4.0606537
28: -10.0215321, -1.4237309, -10.0051069, -1.4346867, -6.1414871, 6.1353874
29: -45.5783005, -36.8071060, -45.5581741, -36.8495483, -4.9839287, 5.0070934
30: -32.1771812, -22.9984207, -32.1543808, -23.0367851, -4.9727287, 4.9917564
31: -32.2355270, -23.5098953, -32.2009125, -23.5305595, -6.2944679, 6.2790871
32: 7.7017322, 13.6753922, 7.7261219, 13.6660538, -4.1604538, 4.1423225
33: 4.5898843, 16.3122139, 4.6510792, 16.3000641, -6.7102737, 6.6525726
34: 20.5305176, 30.9900703, 20.5778465, 30.9727364, -5.7369099, 5.7027893
35: 16.4941978, 26.8651066, 16.5531082, 26.8491993, -5.4486542, 5.4013557
36: 28.7962513, 35.1250916, 28.8278618, 35.1192322, -3.4463024, 3.4174242
37: 11.0095043, 20.1167870, 11.0712624, 20.1035004, -5.9707260, 5.9169159
38: 34.8413696, 43.6861763, 34.9021416, 43.6639786, -6.0534477, 6.0125427
39: 8.9640541, 18.5066299, 9.0266342, 18.4946709, -6.5540352, 6.4996300
40: 15.7734003, 25.1245918, 15.8179274, 25.1136093, -5.8077736, 5.7684574
41: 6.7195225, 13.2266045, 6.7548709, 13.2160826, -5.0184402, 4.9906082
42: -12.3990030, -3.4547563, -12.3761110, -3.4601922, -7.0476761, 7.0289879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=78, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 759

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5858404, upper bound: 3.6387048
time: 6.08 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5978590, upper bound: 3.6387568
time: 5.76 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5823517, -8.4754782, -21.5309086, -8.4789009, -10.4660110, 10.3794746
1: -21.4316921, -12.2303839, -21.4229698, -12.2471075, -5.2777138, 5.2697582
2: -12.3978977, -5.7771926, -12.3877449, -5.7841201, -4.2748299, 4.2593002
3: -12.0140562, -4.1626692, -11.9988270, -4.1822567, -5.3613281, 5.3701591
4: -10.2971478, 0.0175240, -10.2785749, -0.0153463, -6.0323257, 6.0774078
5: -13.5634670, -4.0399451, -13.5528688, -4.0594797, -6.1467247, 6.1455231
6: -8.3403673, 0.5445777, -8.3104439, 0.5288848, -6.4663658, 6.4432259
7: -32.1574860, -22.0487480, -32.1428566, -22.0890312, -5.8213043, 5.8428268
8: -18.8177528, -9.0988894, -18.8069477, -9.1165867, -5.2097092, 5.2483387
9: -5.3463721, 1.3981118, -5.3132539, 1.3908178, -4.0570946, 4.0347977
10: -36.1382904, -27.7517624, -36.1255150, -27.7834167, -5.2496681, 5.2576828
11: -55.1336555, -44.7699356, -55.1098251, -44.8443832, -4.9519043, 4.9794159
12: -11.5814342, -4.5812030, -11.5705433, -4.6043682, -6.2202797, 6.2127686
13: 0.8778406, 8.0190048, 0.8973036, 7.9972568, -5.2930984, 5.3063469
14: -71.0859375, -57.9431458, -71.0761261, -57.9721909, -8.2486343, 8.2540398
15: -8.9214668, 0.9165831, -8.9089212, 0.8905277, -4.8740788, 4.9156361
16: -33.5695686, -23.9660187, -33.5289764, -23.9979286, -6.4738541, 6.4355927
17: -88.6787109, -72.3836136, -88.6568909, -72.4794922, -8.1633301, 8.2418823
18: -4.1792688, 1.0747476, -4.1649508, 1.0490954, -3.3971882, 3.3957882
19: -30.5269508, -23.1997128, -30.5072517, -23.2236290, -4.6510506, 4.6493530
20: -11.1734610, -5.1512146, -11.1633940, -5.1677070, -4.9200554, 4.9374237
21: -43.5477142, -35.0479736, -43.5256119, -35.0825577, -4.2728901, 4.2637711
22: -27.0075207, -19.5268173, -26.9940567, -19.5652103, -4.3301659, 4.3542233
23: -20.8583431, -12.5016270, -20.8263187, -12.5245972, -4.7996731, 4.7703133
24: -16.8653946, -7.6328974, -16.8441906, -7.6502953, -7.1747589, 7.1589661
25: -14.6424198, -6.9527712, -14.6165771, -6.9757771, -4.2046471, 4.1936646
26: -14.6214590, -7.7974482, -14.6124401, -7.8237658, -6.5340195, 6.5696335
27: -14.6334267, -9.5229588, -14.6169930, -9.5615988, -4.0576286, 4.0723133
28: -10.0250711, -1.4228759, -10.0126400, -1.4270592, -6.1381798, 6.1503410
29: -45.5851059, -36.8070641, -45.5655518, -36.8620758, -4.9998798, 5.0266552
30: -32.1878395, -22.9968910, -32.1732025, -23.0381737, -5.0043945, 5.0058517
31: -32.2444916, -23.5098534, -32.2130470, -23.5421181, -6.3174171, 6.3098412
32: 7.6999569, 13.6754742, 7.7237568, 13.6683464, -4.1683407, 4.1449051
33: 4.5858126, 16.3123379, 4.6496568, 16.3024101, -6.7187366, 6.6466141
34: 20.5249023, 30.9901314, 20.5883026, 30.9628754, -5.7598572, 5.7079926
35: 16.4879417, 26.8651485, 16.5562401, 26.8464375, -5.4745541, 5.4057007
36: 28.7954788, 35.1252975, 28.8381767, 35.1131401, -3.4513798, 3.4193096
37: 11.0041695, 20.1168900, 11.0605106, 20.1065941, -5.9826126, 5.9220428
38: 34.8403511, 43.6910400, 34.9100647, 43.6631775, -6.0576630, 6.0265274
39: 8.9634571, 18.5084496, 9.0259733, 18.4988518, -6.5529785, 6.4983749
40: 15.7700653, 25.1274223, 15.8143692, 25.1204262, -5.8185692, 5.7776356
41: 6.7150149, 13.2267151, 6.7435651, 13.2207575, -5.0290680, 4.9985199
42: -12.3993721, -3.4542549, -12.3689604, -3.4700041, -7.0484390, 7.0331078

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=78, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 759

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5970165, upper bound: 3.6387045
time: 5.19 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6090363, upper bound: 3.6387564
time: 5.38 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5823517, -8.4754782, -21.5353966, -8.4781456, -10.4661789, 10.3832245
1: -21.4316921, -12.2303839, -21.4262791, -12.2436371, -5.2792301, 5.2711849
2: -12.3978977, -5.7771926, -12.3931437, -5.7821670, -4.2706757, 4.2586193
3: -12.0140562, -4.1626692, -12.0113354, -4.1728377, -5.3582840, 5.3709450
4: -10.2971478, 0.0175240, -10.2875271, -0.0122356, -6.0296631, 6.0792217
5: -13.5634670, -4.0399451, -13.5551462, -4.0563035, -6.1492882, 6.1473503
6: -8.3403673, 0.5445777, -8.3158693, 0.5382565, -6.4690475, 6.4423103
7: -32.1574860, -22.0487480, -32.1491089, -22.0808010, -5.8255272, 5.8462772
8: -18.8177528, -9.0988894, -18.8148651, -9.1110573, -5.2096367, 5.2496147
9: -5.3463721, 1.3981118, -5.3210716, 1.3932147, -4.0615940, 4.0459900
10: -36.1382904, -27.7517624, -36.1345139, -27.7696571, -5.2497044, 5.2542057
11: -55.1336555, -44.7699356, -55.1291924, -44.8089447, -4.9497433, 4.9603634
12: -11.5814342, -4.5812030, -11.5777750, -4.5950975, -6.2229767, 6.2151375
13: 0.8778406, 8.0190048, 0.8875685, 8.0018444, -5.2927170, 5.3098373
14: -71.0859375, -57.9431458, -71.0788345, -57.9694519, -8.2535934, 8.2586708
15: -8.9214668, 0.9165831, -8.9161015, 0.8929358, -4.8678532, 4.9115162
16: -33.5695686, -23.9660187, -33.5489883, -23.9752636, -6.4766617, 6.4349442
17: -88.6787109, -72.3836136, -88.6678467, -72.4609604, -8.1659164, 8.2392120
18: -4.1792688, 1.0747476, -4.1678891, 1.0541949, -3.4009190, 3.3977795
19: -30.5269508, -23.1997128, -30.5210495, -23.2048645, -4.6487083, 4.6422310
20: -11.1734610, -5.1512146, -11.1695375, -5.1580400, -4.9213829, 4.9356918
21: -43.5477142, -35.0479736, -43.5429268, -35.0557289, -4.2648582, 4.2471180
22: -27.0075207, -19.5268173, -27.0006485, -19.5549908, -4.3324432, 4.3526573
23: -20.8583431, -12.5016270, -20.8333015, -12.5105829, -4.8008385, 4.7654743
24: -16.8653946, -7.6328974, -16.8480225, -7.6403999, -7.1793823, 7.1588135
25: -14.6424198, -6.9527712, -14.6258793, -6.9593611, -4.2050610, 4.1868572
26: -14.6214590, -7.7974482, -14.6146202, -7.8201065, -6.5366020, 6.5705452
27: -14.6334267, -9.5229588, -14.6269321, -9.5460358, -4.0579529, 4.0684433
28: -10.0250711, -1.4228759, -10.0155792, -1.4241405, -6.1417389, 6.1534157
29: -45.5851059, -36.8070641, -45.5784187, -36.8388863, -5.0010166, 5.0170784
30: -32.1878395, -22.9968910, -32.1842194, -23.0166359, -5.0070858, 4.9968224
31: -32.2444916, -23.5098534, -32.2283134, -23.5203133, -6.3135719, 6.3012733
32: 7.6999569, 13.6754742, 7.7201662, 13.6689196, -4.1685638, 4.1482544
33: 4.5858126, 16.3123379, 4.6379156, 16.3056660, -6.7183952, 6.6583023
34: 20.5249023, 30.9901314, 20.5609283, 30.9814644, -5.7513161, 5.7076969
35: 16.4879417, 26.8651485, 16.5338116, 26.8590622, -5.4648075, 5.4067097
36: 28.7954788, 35.1252975, 28.8244667, 35.1217422, -3.4487419, 3.4196138
37: 11.0041695, 20.1168900, 11.0539684, 20.1104965, -5.9837303, 5.9270363
38: 34.8403511, 43.6910400, 34.8888893, 43.6784172, -6.0579796, 6.0316429
39: 8.9634571, 18.5084496, 9.0167208, 18.4997120, -6.5556908, 6.5091095
40: 15.7700653, 25.1274223, 15.8032837, 25.1222115, -5.8181362, 5.7876205
41: 6.7150149, 13.2267151, 6.7405124, 13.2236900, -5.0306282, 5.0004807
42: -12.3993721, -3.4542549, -12.3771563, -3.4571404, -7.0501556, 7.0307999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=78, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 759

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5970165, upper bound: 3.6387048
time: 6.01 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6090363, upper bound: 3.6387568
time: 4.99 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -21.5345383, -8.4768724, -21.5575619, -8.4788246, -10.3752289, 10.4258881
1: -21.4266586, -12.2439270, -21.4198837, -12.2393770, -5.2611103, 5.2815990
2: -12.3921747, -5.7815905, -12.3882227, -5.7794342, -4.2609711, 4.2644310
3: -12.0095558, -4.1736736, -11.9980583, -4.1766605, -5.3676834, 5.3540154
4: -10.2818174, -0.0115393, -10.2768841, 0.0020967, -6.0615845, 6.0180283
5: -13.5533333, -4.0567513, -13.5548630, -4.0479512, -6.1375008, 6.1356964
6: -8.3164396, 0.5315919, -8.3125887, 0.5209942, -6.4255638, 6.4610786
7: -32.1554337, -22.0813980, -32.1448746, -22.0627384, -5.8418503, 5.8146362
8: -18.8031425, -9.1112747, -18.8043251, -9.1110420, -5.2300606, 5.2025528
9: -5.3177662, 1.3936148, -5.3247309, 1.3920209, -4.0340710, 4.0337296
10: -36.1309204, -27.7705936, -36.1200638, -27.7779655, -5.2406769, 5.2550659
11: -55.1326218, -44.8232040, -55.1072006, -44.8254166, -4.9447002, 4.9521904
12: -11.5779095, -4.5980229, -11.5677814, -4.5959678, -6.2064934, 6.2329788
13: 0.8929712, 8.0001097, 0.8978887, 8.0090866, -5.2995605, 5.2770157
14: -71.0783997, -57.9727173, -71.0666275, -57.9601440, -8.2404480, 8.2348099
15: -8.9068413, 0.8929195, -8.9013691, 0.9001780, -4.8901939, 4.8626575
16: -33.5505142, -23.9842873, -33.5346909, -23.9897346, -6.4275894, 6.4538307
17: -88.6759262, -72.4635162, -88.6462250, -72.4369049, -8.2075233, 8.1450272
18: -4.1697173, 1.0489702, -4.1720448, 1.0583980, -3.3831978, 3.3915253
19: -30.5213833, -23.2079086, -30.5116272, -23.2194901, -4.6446037, 4.6501579
20: -11.1697483, -5.1601815, -11.1661921, -5.1661739, -4.9325829, 4.9198494
21: -43.5446892, -35.0626183, -43.5285683, -35.0783005, -4.2497883, 4.2791176
22: -27.0020714, -19.5547619, -26.9935112, -19.5495949, -4.3361607, 4.3277111
23: -20.8327370, -12.5149822, -20.8451347, -12.5220356, -4.7576828, 4.7912540
24: -16.8466263, -7.6457438, -16.8563118, -7.6503592, -7.1474304, 7.1685867
25: -14.6245155, -6.9594183, -14.6287804, -6.9747448, -4.1797028, 4.2057972
26: -14.6104231, -7.8216486, -14.6107121, -7.8224502, -6.5388336, 6.5267067
27: -14.6286659, -9.5506878, -14.6203070, -9.5464516, -4.0594864, 4.0582542
28: -10.0147409, -1.4306835, -10.0201902, -1.4283487, -6.1607437, 6.1286697
29: -45.5812645, -36.8433838, -45.5637207, -36.8449249, -4.9996853, 5.0071697
30: -32.1851959, -23.0294609, -32.1731453, -23.0297089, -4.9834900, 5.0024834
31: -32.2275772, -23.5253029, -32.2211456, -23.5387020, -6.2961998, 6.3065224
32: 7.7199526, 13.6664362, 7.7223701, 13.6656990, -4.1404648, 4.1499138
33: 4.6390786, 16.3105469, 4.6212530, 16.2999992, -6.6489983, 6.6824875
34: 20.5616398, 30.9840603, 20.5625153, 30.9636688, -5.7085094, 5.7423744
35: 16.5353966, 26.8602962, 16.5262661, 26.8448353, -5.4079514, 5.4460354
36: 28.8254147, 35.1228485, 28.8228645, 35.1084938, -3.4132385, 3.4348373
37: 11.0551834, 20.1129417, 11.0316124, 20.1050797, -5.9181633, 5.9577789
38: 34.8968620, 43.6855736, 34.8766403, 43.6627350, -6.0109406, 6.0450516
39: 9.0257816, 18.5051003, 8.9998436, 18.4966908, -6.4875565, 6.5222740
40: 15.8056822, 25.1223125, 15.8019209, 25.1171761, -5.7790146, 5.7859077
41: 6.7408323, 13.2213011, 6.7361627, 13.2140303, -4.9891739, 5.0115623
42: -12.3772259, -3.4559534, -12.3785992, -3.4765055, -7.0170555, 7.0486565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=78, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1563

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5862844, upper bound: 3.6239230
time: 4.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5862844, upper bound: 3.6387003
time: 4.50 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -21.5345383, -8.4768724, -21.5620499, -8.4780226, -10.3754044, 10.4296417
1: -21.4266586, -12.2439270, -21.4231720, -12.2359066, -5.2626152, 5.2830276
2: -12.3921747, -5.7815905, -12.3936234, -5.7774405, -4.2568111, 4.2637520
3: -12.0095558, -4.1736736, -12.0105782, -4.1672211, -5.3646317, 5.3547707
4: -10.2818174, -0.0115393, -10.2858582, 0.0052235, -6.0589180, 6.0198383
5: -13.5533333, -4.0567513, -13.5571175, -4.0447292, -6.1400948, 6.1375275
6: -8.3164396, 0.5315919, -8.3179932, 0.5304065, -6.4282265, 6.4601669
7: -32.1554337, -22.0813980, -32.1511803, -22.0545273, -5.8460808, 5.8180866
8: -18.8031425, -9.1112747, -18.8122616, -9.1055126, -5.2299900, 5.2038517
9: -5.3177662, 1.3936148, -5.3325729, 1.3944352, -4.0385265, 4.0449848
10: -36.1309204, -27.7705936, -36.1290817, -27.7641716, -5.2407131, 5.2516136
11: -55.1326218, -44.8232040, -55.1266022, -44.7898941, -4.9425621, 4.9331512
12: -11.5779095, -4.5980229, -11.5749922, -4.5867014, -6.2091942, 6.2353325
13: 0.8929712, 8.0001097, 0.8881400, 8.0136786, -5.2991982, 5.2805290
14: -71.0783997, -57.9727173, -71.0693665, -57.9574127, -8.2454185, 8.2394447
15: -8.9068413, 0.8929195, -8.9085131, 0.9025865, -4.8839302, 4.8585453
16: -33.5505142, -23.9842873, -33.5547066, -23.9671135, -6.4301987, 6.4532051
17: -88.6759262, -72.4635162, -88.6572113, -72.4183044, -8.2101173, 8.1423569
18: -4.1697173, 1.0489702, -4.1749964, 1.0635304, -3.3869343, 3.3935356
19: -30.5213833, -23.2079086, -30.5254154, -23.2007561, -4.6422691, 4.6430359
20: -11.1697483, -5.1601815, -11.1723289, -5.1565018, -4.9339218, 4.9181480
21: -43.5446892, -35.0626183, -43.5458755, -35.0514870, -4.2417641, 4.2624817
22: -27.0020714, -19.5547619, -27.0000610, -19.5393333, -4.3384495, 4.3261585
23: -20.8327370, -12.5149822, -20.8521595, -12.5080109, -4.7588806, 4.7864399
24: -16.8466263, -7.6457438, -16.8601456, -7.6404505, -7.1520958, 7.1684532
25: -14.6245155, -6.9594183, -14.6381073, -6.9583263, -4.1801300, 4.1990089
26: -14.6104231, -7.8216486, -14.6129026, -7.8188000, -6.5414162, 6.5276260
27: -14.6286659, -9.5506878, -14.6302443, -9.5308876, -4.0598335, 4.0544090
28: -10.0147409, -1.4306835, -10.0231390, -1.4254522, -6.1642990, 6.1317482
29: -45.5812645, -36.8433838, -45.5765762, -36.8217545, -5.0008736, 4.9976082
30: -32.1851959, -23.0294609, -32.1841087, -23.0081635, -4.9861679, 4.9934692
31: -32.2275772, -23.5253029, -32.2363815, -23.5168648, -6.2924042, 6.2979698
32: 7.7199526, 13.6664362, 7.7188039, 13.6662626, -4.1406937, 4.1532631
33: 4.6390786, 16.3105469, 4.6095629, 16.3032532, -6.6487617, 6.6941795
34: 20.5616398, 30.9840603, 20.5350742, 30.9822235, -5.6999626, 5.7420444
35: 16.5353966, 26.8602962, 16.5038223, 26.8574753, -5.3982677, 5.4470272
36: 28.8254147, 35.1228485, 28.8091602, 35.1170807, -3.4106083, 3.4351492
37: 11.0551834, 20.1129417, 11.0251083, 20.1089668, -5.9192238, 5.9627762
38: 34.8968620, 43.6855736, 34.8554726, 43.6779671, -6.0112267, 6.0500641
39: 9.0257816, 18.5051003, 8.9905748, 18.4975662, -6.4903297, 6.5330048
40: 15.8056822, 25.1223125, 15.7908535, 25.1189575, -5.7785797, 5.7959003
41: 6.7408323, 13.2213011, 6.7331223, 13.2169800, -4.9907112, 5.0135078
42: -12.3772259, -3.4559534, -12.3867807, -3.4636152, -7.0187798, 7.0463448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=78, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1563

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5862844, upper bound: 3.6239234
time: 4.89 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5862844, upper bound: 3.6387007
time: 4.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -21.5438328, -8.4740868, -21.5574455, -8.4787569, -10.4070969, 10.4234695
1: -21.4284210, -12.2416077, -21.4199219, -12.2396631, -5.2673683, 5.2827435
2: -12.3951683, -5.7787285, -12.3882999, -5.7794461, -4.2702980, 4.2632370
3: -12.0135393, -4.1648407, -11.9981909, -4.1765790, -5.3716469, 5.3628731
4: -10.2925167, 0.0093410, -10.2773056, 0.0020864, -6.0718346, 6.0393772
5: -13.5598183, -4.0494342, -13.5550385, -4.0479383, -6.1471481, 6.1339073
6: -8.3307953, 0.5369194, -8.3125591, 0.5212150, -6.4402237, 6.4665756
7: -32.1564026, -22.0827904, -32.1448746, -22.0644951, -5.8448639, 5.8191853
8: -18.8147297, -9.0755329, -18.8048515, -9.1110201, -5.2414017, 5.2384453
9: -5.3223023, 1.4032469, -5.3248873, 1.3920624, -4.0384045, 4.0433941
10: -36.1363983, -27.7641029, -36.1202011, -27.7779121, -5.2464008, 5.2570648
11: -55.1678085, -44.8096695, -55.1072197, -44.8248558, -4.9806404, 4.9654331
12: -11.5799837, -4.5905080, -11.5677700, -4.5957985, -6.2073975, 6.2398033
13: 0.8869425, 8.0194578, 0.8976672, 8.0091696, -5.3056908, 5.2968750
14: -71.0824738, -57.9658546, -71.0666351, -57.9601059, -8.2540398, 8.2400284
15: -8.9197760, 0.9144998, -8.9018002, 0.9001479, -4.9031849, 4.8846359
16: -33.5739479, -23.9758511, -33.5347481, -23.9893761, -6.4513016, 6.4616051
17: -88.6762390, -72.4605179, -88.6462097, -72.4382172, -8.2076797, 8.1501465
18: -4.1933627, 1.0538881, -4.1720724, 1.0586183, -3.4070435, 3.3963242
19: -30.5304012, -23.2049828, -30.5116634, -23.2194138, -4.6541386, 4.6526890
20: -11.1718483, -5.1589832, -11.1661978, -5.1669502, -4.9322586, 4.9241219
21: -43.5615158, -35.0558472, -43.5286026, -35.0780182, -4.2669086, 4.2858410
22: -27.0055256, -19.5541363, -26.9933090, -19.5495834, -4.3400650, 4.3277054
23: -20.8522472, -12.5059528, -20.8451767, -12.5217075, -4.7775116, 4.8008537
24: -16.8665333, -7.6382618, -16.8563976, -7.6500311, -7.1675682, 7.1759987
25: -14.6285715, -6.9567747, -14.6288433, -6.9746819, -4.1842670, 4.2086182
26: -14.6148262, -7.8182378, -14.6108170, -7.8228068, -6.5390053, 6.5451012
27: -14.6400490, -9.5454121, -14.6203861, -9.5462494, -4.0704803, 4.0635891
28: -10.0295343, -1.4221704, -10.0202236, -1.4280457, -6.1609039, 6.1362305
29: -45.5942459, -36.8391380, -45.5637741, -36.8447266, -5.0127296, 5.0105000
30: -32.2174835, -23.0137482, -32.1731262, -23.0291328, -5.0163517, 5.0184231
31: -32.2469139, -23.5209351, -32.2212143, -23.5385017, -6.3169250, 6.3103943
32: 7.7150021, 13.6686230, 7.7223802, 13.6657848, -4.1456337, 4.1520615
33: 4.6363187, 16.3117867, 4.6219773, 16.3000183, -6.6546459, 6.6837196
34: 20.5493851, 30.9890480, 20.5624828, 30.9638462, -5.7209473, 5.7468987
35: 16.5215511, 26.8654633, 16.5262032, 26.8450470, -5.4219704, 5.4510612
36: 28.8233185, 35.1245041, 28.8228283, 35.1085358, -3.4152670, 3.4363346
37: 11.0457001, 20.1164455, 11.0315590, 20.1052036, -5.9280891, 5.9609909
38: 34.8861008, 43.6995010, 34.8763046, 43.6627350, -6.0219460, 6.0592651
39: 9.0164528, 18.5168571, 8.9996672, 18.4966774, -6.4966888, 6.5316963
40: 15.7948036, 25.1271667, 15.8018379, 25.1171951, -5.7890205, 5.7939816
41: 6.7319651, 13.2256451, 6.7361135, 13.2142000, -4.9981308, 5.0158615
42: -12.3772430, -3.4532442, -12.3783693, -3.4764814, -7.0172462, 7.0544968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=78, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1563

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5983001, upper bound: 3.6239753
time: 5.11 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5983001, upper bound: 3.6387523
time: 5.21 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -21.5438328, -8.4740868, -21.5619335, -8.4779844, -10.4072495, 10.4272156
1: -21.4284210, -12.2416077, -21.4232349, -12.2361889, -5.2688732, 5.2841759
2: -12.3951683, -5.7787285, -12.3937016, -5.7774534, -4.2661400, 4.2625561
3: -12.0135393, -4.1648407, -12.0107193, -4.1671848, -5.3685989, 5.3636208
4: -10.2925167, 0.0093410, -10.2862310, 0.0051928, -6.0691757, 6.0411911
5: -13.5598183, -4.0494342, -13.5573072, -4.0446939, -6.1497078, 6.1357346
6: -8.3307953, 0.5369194, -8.3180046, 0.5306039, -6.4428711, 6.4656639
7: -32.1564026, -22.0827904, -32.1511536, -22.0562706, -5.8490982, 5.8226414
8: -18.8147297, -9.0755329, -18.8127670, -9.1054840, -5.2413292, 5.2397079
9: -5.3223023, 1.4032469, -5.3327417, 1.3944460, -4.0428505, 4.0546398
10: -36.1363983, -27.7641029, -36.1292000, -27.7641220, -5.2464428, 5.2536011
11: -55.1678085, -44.8096695, -55.1266327, -44.7892914, -4.9784985, 4.9463940
12: -11.5799837, -4.5905080, -11.5749817, -4.5865474, -6.2101059, 6.2421799
13: 0.8869425, 8.0194578, 0.8879042, 8.0137701, -5.3053284, 5.3003693
14: -71.0824738, -57.9658546, -71.0694275, -57.9573975, -8.2590027, 8.2446594
15: -8.9197760, 0.9144998, -8.9089241, 0.9025879, -4.8969460, 4.8805275
16: -33.5739479, -23.9758511, -33.5547066, -23.9667244, -6.4538994, 6.4609795
17: -88.6762390, -72.4605179, -88.6571655, -72.4196320, -8.2102623, 8.1474800
18: -4.1933627, 1.0538881, -4.1750283, 1.0637357, -3.4107838, 3.3983345
19: -30.5304012, -23.2049828, -30.5254612, -23.2006531, -4.6518040, 4.6455669
20: -11.1718483, -5.1589832, -11.1723509, -5.1572657, -4.9336128, 4.9224281
21: -43.5615158, -35.0558472, -43.5459023, -35.0512009, -4.2588749, 4.2692146
22: -27.0055256, -19.5541363, -26.9999065, -19.5393543, -4.3423481, 4.3261585
23: -20.8522472, -12.5059528, -20.8521633, -12.5076981, -4.7787094, 4.7960529
24: -16.8665333, -7.6382618, -16.8602238, -7.6401300, -7.1722221, 7.1758957
25: -14.6285715, -6.9567747, -14.6381702, -6.9582491, -4.1846962, 4.2018318
26: -14.6148262, -7.8182378, -14.6129913, -7.8191376, -6.5415802, 6.5460167
27: -14.6400490, -9.5454121, -14.6303368, -9.5306873, -4.0708199, 4.0597458
28: -10.0295343, -1.4221704, -10.0231876, -1.4251261, -6.1644707, 6.1393166
29: -45.5942459, -36.8391380, -45.5766258, -36.8215790, -5.0139046, 5.0009441
30: -32.2174835, -23.0137482, -32.1841125, -23.0076027, -5.0190372, 5.0094376
31: -32.2469139, -23.5209351, -32.2364807, -23.5166626, -6.3130951, 6.3018532
32: 7.7150021, 13.6686230, 7.7188187, 13.6663446, -4.1458511, 4.1554031
33: 4.6363187, 16.3117867, 4.6103058, 16.3032646, -6.6544189, 6.6954117
34: 20.5493851, 30.9890480, 20.5350494, 30.9824219, -5.7123928, 5.7465725
35: 16.5215511, 26.8654633, 16.5037880, 26.8577023, -5.4122810, 5.4520607
36: 28.8233185, 35.1245041, 28.8091125, 35.1171227, -3.4126329, 3.4366341
37: 11.0457001, 20.1164455, 11.0250340, 20.1091003, -5.9291420, 5.9659882
38: 34.8861008, 43.6995010, 34.8551140, 43.6779861, -6.0222588, 6.0642815
39: 9.0164528, 18.5168571, 8.9903841, 18.4975471, -6.4994431, 6.5424194
40: 15.7948036, 25.1271667, 15.7907534, 25.1189976, -5.7886162, 5.8039780
41: 6.7319651, 13.2256451, 6.7330947, 13.2171421, -4.9996796, 5.0178185
42: -12.3772430, -3.4532442, -12.3865461, -3.4635925, -7.0189705, 7.0521698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=78, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1563

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5983001, upper bound: 3.6239756
time: 4.99 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5983001, upper bound: 3.6387526
time: 5.08 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -21.5353146, -8.4767933, -21.5776443, -8.4764938, -10.3822403, 10.4576645
1: -21.4272881, -12.2439003, -21.4283104, -12.2338972, -5.2664890, 5.2906990
2: -12.3924866, -5.7818208, -12.3923111, -5.7791834, -4.2634621, 4.2686958
3: -12.0097122, -4.1733861, -12.0012197, -4.1721773, -5.3718987, 5.3573723
4: -10.2830963, -0.0112605, -10.2874403, 0.0143511, -6.0773315, 6.0268745
5: -13.5536747, -4.0568838, -13.5608139, -4.0432534, -6.1447639, 6.1451683
6: -8.3168516, 0.5342963, -8.3348541, 0.5347388, -6.4367790, 6.4855232
7: -32.1560593, -22.0813255, -32.1511803, -22.0569839, -5.8492126, 5.8204155
8: -18.8035469, -9.1110649, -18.8088703, -9.1045027, -5.2391338, 5.2072945
9: -5.3176966, 1.3937058, -5.3382778, 1.3956294, -4.0377884, 4.0508747
10: -36.1322746, -27.7705536, -36.1290245, -27.7656555, -5.2500744, 5.2631512
11: -55.1338501, -44.8230743, -55.1142120, -44.8066673, -4.9646912, 4.9575920
12: -11.5787373, -4.5980024, -11.5742264, -4.5908351, -6.2069664, 6.2388954
13: 0.8925767, 8.0003023, 0.8880766, 8.0142431, -5.3066940, 5.2871170
14: -71.0809784, -57.9727592, -71.0830994, -57.9461517, -8.2543335, 8.2498055
15: -8.9089851, 0.8930793, -8.9135122, 0.9141097, -4.9097729, 4.8723755
16: -33.5512314, -23.9844704, -33.5495071, -23.9893913, -6.4291916, 6.4822044
17: -88.6802063, -72.4632111, -88.6676941, -72.4025345, -8.2483635, 8.1640205
18: -4.1702862, 1.0490849, -4.1762199, 1.0691659, -3.3946590, 3.3958797
19: -30.5213509, -23.2077847, -30.5130215, -23.2187500, -4.6456871, 4.6523151
20: -11.1697426, -5.1600046, -11.1672173, -5.1611094, -4.9388695, 4.9212990
21: -43.5445709, -35.0625229, -43.5303192, -35.0753326, -4.2526569, 4.2835350
22: -27.0032635, -19.5546761, -27.0007744, -19.5370903, -4.3512459, 4.3345051
23: -20.8328896, -12.5150146, -20.8512287, -12.5162983, -4.7650547, 4.8006210
24: -16.8468723, -7.6457615, -16.8613892, -7.6434731, -7.1543427, 7.1741867
25: -14.6246252, -6.9592524, -14.6329813, -6.9693565, -4.1858635, 4.2113914
26: -14.6119671, -7.8211107, -14.6189814, -7.8013301, -6.5611839, 6.5352974
27: -14.6289959, -9.5505581, -14.6232958, -9.5389233, -4.0667591, 4.0617485
28: -10.0146828, -1.4305075, -10.0219812, -1.4263923, -6.1644058, 6.1321869
29: -45.5825424, -36.8432465, -45.5721741, -36.8306465, -5.0151062, 5.0156784
30: -32.1855431, -23.0292664, -32.1767731, -23.0195427, -4.9937630, 5.0059605
31: -32.2277985, -23.5246544, -32.2290230, -23.5320835, -6.3043442, 6.3181076
32: 7.7197127, 13.6681967, 7.7035871, 13.6747360, -4.1472054, 4.1709557
33: 4.6387043, 16.3119335, 4.5977287, 16.3089962, -6.6570721, 6.7130089
34: 20.5613174, 30.9850807, 20.5524311, 30.9711399, -5.7159710, 5.7556190
35: 16.5351677, 26.8614769, 16.5105457, 26.8520508, -5.4151878, 5.4660473
36: 28.8253136, 35.1244392, 28.8093109, 35.1166229, -3.4201927, 3.4508181
37: 11.0547905, 20.1143970, 11.0108385, 20.1127663, -5.9253616, 5.9826050
38: 34.8967209, 43.6875801, 34.8622627, 43.6757545, -6.0228195, 6.0617867
39: 9.0254383, 18.5069637, 8.9735651, 18.5075397, -6.4977455, 6.5514145
40: 15.8054256, 25.1236000, 15.7814045, 25.1255913, -5.7856674, 5.8088245
41: 6.7404895, 13.2231178, 6.7181282, 13.2234402, -4.9979019, 5.0323410
42: -12.3774185, -3.4542308, -12.3911343, -3.4671793, -7.0254211, 7.0621681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=78, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1563

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5970164, upper bound: 3.6239230
time: 8.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5970164, upper bound: 3.6387003
time: 5.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.5353146, -8.4767933, -21.5821133, -8.4757366, -10.3823929, 10.4614334
1: -21.4272881, -12.2439003, -21.4315929, -12.2304201, -5.2679863, 5.2921200
2: -12.3924866, -5.7818208, -12.3977394, -5.7771974, -4.2593021, 4.2680149
3: -12.0097122, -4.1733861, -12.0137501, -4.1627932, -5.3688469, 5.3581238
4: -10.2830963, -0.0112605, -10.2963963, 0.0174915, -6.0746765, 6.0286942
5: -13.5536747, -4.0568838, -13.5630693, -4.0400076, -6.1473351, 6.1469917
6: -8.3168516, 0.5342963, -8.3402948, 0.5441295, -6.4394302, 6.4846039
7: -32.1560593, -22.0813255, -32.1574669, -22.0487862, -5.8534584, 5.8238544
8: -18.8035469, -9.1110649, -18.8167725, -9.0989504, -5.2390671, 5.2085800
9: -5.3176966, 1.3937058, -5.3460560, 1.3980365, -4.0422401, 4.0621567
10: -36.1322746, -27.7705536, -36.1380348, -27.7518921, -5.2501297, 5.2596912
11: -55.1338501, -44.8230743, -55.1336250, -44.7711296, -4.9625721, 4.9385529
12: -11.5787373, -4.5980024, -11.5814171, -4.5815673, -6.2096596, 6.2412338
13: 0.8925767, 8.0003023, 0.8783626, 8.0188370, -5.3063316, 5.2906075
14: -71.0809784, -57.9727592, -71.0858307, -57.9434586, -8.2592926, 8.2544365
15: -8.9089851, 0.8930793, -8.9206371, 0.9165106, -4.9035072, 4.8682575
16: -33.5512314, -23.9844704, -33.5694847, -23.9667473, -6.4318085, 6.4815865
17: -88.6802063, -72.4632111, -88.6786652, -72.3839264, -8.2509537, 8.1613464
18: -4.1702862, 1.0490849, -4.1791940, 1.0743077, -3.3984222, 3.3978882
19: -30.5213509, -23.2077847, -30.5267982, -23.1999836, -4.6433563, 4.6451855
20: -11.1697426, -5.1600046, -11.1733694, -5.1514263, -4.9401817, 4.9195976
21: -43.5445709, -35.0625229, -43.5476151, -35.0485153, -4.2446308, 4.2669029
22: -27.0032635, -19.5546761, -27.0073509, -19.5268421, -4.3535442, 4.3329582
23: -20.8328896, -12.5150146, -20.8582039, -12.5022593, -4.7662125, 4.7958164
24: -16.8468723, -7.6457615, -16.8652229, -7.6335454, -7.1589813, 7.1740570
25: -14.6246252, -6.9592524, -14.6422768, -6.9529309, -4.1862793, 4.2045918
26: -14.6119671, -7.8211107, -14.6211748, -7.7976570, -6.5637589, 6.5361938
27: -14.6289959, -9.5505581, -14.6332512, -9.5233650, -4.0671234, 4.0578995
28: -10.0146828, -1.4305075, -10.0249214, -1.4234498, -6.1679726, 6.1352615
29: -45.5825424, -36.8432465, -45.5850067, -36.8074570, -5.0163059, 5.0061111
30: -32.1855431, -23.0292664, -32.1877632, -22.9980068, -4.9964676, 4.9969387
31: -32.2277985, -23.5246544, -32.2442703, -23.5102692, -6.3005371, 6.3095551
32: 7.7197127, 13.6681967, 7.7000237, 13.6752958, -4.1474228, 4.1743088
33: 4.6387043, 16.3119335, 4.5859632, 16.3122406, -6.6568222, 6.7246857
34: 20.5613174, 30.9850807, 20.5249901, 30.9897232, -5.7074356, 5.7552986
35: 16.5351677, 26.8614769, 16.4880714, 26.8646984, -5.4055061, 5.4670544
36: 28.8253136, 35.1244392, 28.7955914, 35.1251984, -3.4175587, 3.4511261
37: 11.0547905, 20.1143970, 11.0043030, 20.1166420, -5.9264297, 5.9876022
38: 34.8967209, 43.6875801, 34.8410873, 43.6909866, -6.0231171, 6.0667953
39: 9.0254383, 18.5069637, 8.9642258, 18.5084095, -6.5004959, 6.5621567
40: 15.8054256, 25.1236000, 15.7702913, 25.1273632, -5.7852383, 5.8188782
41: 6.7404895, 13.2231178, 6.7151079, 13.2263947, -4.9994392, 5.0343018
42: -12.3774185, -3.4542308, -12.3993301, -3.4542999, -7.0271568, 7.0598450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=78, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1563

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5970164, upper bound: 3.6239234
time: 5.51 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5970164, upper bound: 3.6387007
time: 5.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5445709, -8.4739914, -21.5775051, -8.4763956, -10.4141006, 10.4552612
1: -21.4290485, -12.2415981, -21.4283600, -12.2341862, -5.2727432, 5.2918510
2: -12.3954668, -5.7789497, -12.3923941, -5.7791948, -4.2727928, 4.2674961
3: -12.0137177, -4.1645889, -12.0013580, -4.1721306, -5.3758659, 5.3662224
4: -10.2938023, 0.0096242, -10.2878389, 0.0143209, -6.0875893, 6.0482292
5: -13.5601473, -4.0495539, -13.5609722, -4.0432091, -6.1543846, 6.1433411
6: -8.3312378, 0.5396384, -8.3348980, 0.5349832, -6.4514542, 6.4910278
7: -32.1570435, -22.0827141, -32.1511536, -22.0587234, -5.8522263, 5.8249664
8: -18.8151493, -9.0752993, -18.8093605, -9.1044712, -5.2504787, 5.2431908
9: -5.3222303, 1.4033334, -5.3384237, 1.3956654, -4.0421181, 4.0605392
10: -36.1377678, -27.7641125, -36.1291428, -27.7656364, -5.2558022, 5.2651443
11: -55.1690178, -44.8095512, -55.1142502, -44.8060684, -5.0006351, 4.9708366
12: -11.5808086, -4.5905018, -11.5742245, -4.5906606, -6.2078667, 6.2457619
13: 0.8865424, 8.0196543, 0.8878496, 8.0143156, -5.3128433, 5.3069534
14: -71.0850601, -57.9658928, -71.0831299, -57.9461403, -8.2679405, 8.2550392
15: -8.9219294, 0.9146500, -8.9139261, 0.9141350, -4.9227715, 4.8943520
16: -33.5746613, -23.9760342, -33.5495262, -23.9890022, -6.4528885, 6.4899750
17: -88.6805115, -72.4601822, -88.6676559, -72.4038086, -8.2485085, 8.1691170
18: -4.1939507, 1.0539906, -4.1762767, 1.0693786, -3.4185181, 3.4006767
19: -30.5303860, -23.2048721, -30.5130692, -23.2186508, -4.6552200, 4.6548576
20: -11.1718378, -5.1587934, -11.1672239, -5.1618662, -4.9385643, 4.9255905
21: -43.5613937, -35.0557594, -43.5303574, -35.0750427, -4.2697659, 4.2902565
22: -27.0067329, -19.5540409, -27.0005970, -19.5370960, -4.3551502, 4.3345070
23: -20.8523865, -12.5059910, -20.8512573, -12.5159664, -4.7848949, 4.8102283
24: -16.8667717, -7.6382875, -16.8614826, -7.6431389, -7.1744614, 7.1816139
25: -14.6286345, -6.9565859, -14.6330385, -6.9692950, -4.1904202, 4.2142105
26: -14.6163616, -7.8176880, -14.6190653, -7.8016763, -6.5613708, 6.5536880
27: -14.6403618, -9.5452929, -14.6233902, -9.5387077, -4.0777435, 4.0670834
28: -10.0294352, -1.4219750, -10.0220070, -1.4260745, -6.1645699, 6.1397667
29: -45.5954819, -36.8390121, -45.5721741, -36.8304520, -5.0281487, 5.0190086
30: -32.2178268, -23.0135956, -32.1768036, -23.0190010, -5.0266323, 5.0218925
31: -32.2471466, -23.5202789, -32.2290649, -23.5318871, -6.3250694, 6.3219833
32: 7.7147593, 13.6704035, 7.7035866, 13.6748142, -4.1523724, 4.1730919
33: 4.6360044, 16.3132401, 4.5984840, 16.3090057, -6.6627083, 6.7142487
34: 20.5491028, 30.9900665, 20.5523796, 30.9713306, -5.7284012, 5.7601662
35: 16.5213032, 26.8666344, 16.5104942, 26.8522854, -5.4291935, 5.4710846
36: 28.8232079, 35.1260757, 28.8092461, 35.1166611, -3.4222317, 3.4523087
37: 11.0453224, 20.1179237, 11.0107803, 20.1128883, -5.9352722, 5.9858055
38: 34.8859596, 43.7014732, 34.8619080, 43.6757622, -6.0338326, 6.0759926
39: 9.0161476, 18.5187016, 8.9733906, 18.5075512, -6.5068474, 6.5608521
40: 15.7945385, 25.1284676, 15.7813187, 25.1255894, -5.7956619, 5.8168964
41: 6.7316031, 13.2274380, 6.7181048, 13.2236233, -5.0068817, 5.0366325
42: -12.3774128, -3.4515114, -12.3908901, -3.4671774, -7.0256119, 7.0680008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=78, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1563

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6090362, upper bound: 3.6239753
time: 4.84 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6090362, upper bound: 3.6387523
time: 4.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5445709, -8.4739914, -21.5820084, -8.4756832, -10.4142838, 10.4590225
1: -21.4290485, -12.2415981, -21.4316216, -12.2307034, -5.2742500, 5.2932644
2: -12.3954668, -5.7789497, -12.3978252, -5.7772017, -4.2686348, 4.2668114
3: -12.0137177, -4.1645889, -12.0138836, -4.1627636, -5.3728104, 5.3669662
4: -10.2938023, 0.0096242, -10.2967758, 0.0174781, -6.0849266, 6.0500488
5: -13.5601473, -4.0495539, -13.5632210, -4.0400100, -6.1569633, 6.1451607
6: -8.3312378, 0.5396384, -8.3402958, 0.5443430, -6.4541130, 6.4901161
7: -32.1570435, -22.0827141, -32.1574554, -22.0505447, -5.8564758, 5.8284111
8: -18.8151493, -9.0752993, -18.8172703, -9.0989342, -5.2503986, 5.2444477
9: -5.3222303, 1.4033334, -5.3462067, 1.3980637, -4.0465622, 4.0718174
10: -36.1377678, -27.7641125, -36.1381531, -27.7518501, -5.2558498, 5.2616863
11: -55.1690178, -44.8095512, -55.1336060, -44.7705307, -4.9985161, 4.9518070
12: -11.5808086, -4.5905018, -11.5814075, -4.5814009, -6.2105789, 6.2480888
13: 0.8865424, 8.0196543, 0.8780900, 8.0189161, -5.3124771, 5.3104324
14: -71.0850601, -57.9658928, -71.0858765, -57.9434128, -8.2729034, 8.2596626
15: -8.9219294, 0.9146500, -8.9210901, 0.9165287, -4.9165306, 4.8902397
16: -33.5746613, -23.9760342, -33.5694962, -23.9663963, -6.4554977, 6.4893570
17: -88.6805115, -72.4601822, -88.6786270, -72.3852158, -8.2510948, 8.1664505
18: -4.1939507, 1.0539906, -4.1792235, 1.0745084, -3.4222794, 3.4026871
19: -30.5303860, -23.2048721, -30.5268669, -23.1998978, -4.6528816, 4.6477165
20: -11.1718378, -5.1587934, -11.1733875, -5.1521931, -4.9398918, 4.9238834
21: -43.5613937, -35.0557594, -43.5476608, -35.0482140, -4.2617340, 4.2736282
22: -27.0067329, -19.5540409, -27.0071735, -19.5268364, -4.3574467, 4.3329563
23: -20.8523865, -12.5059910, -20.8582535, -12.5019550, -4.7860565, 4.8054142
24: -16.8667717, -7.6382875, -16.8652916, -7.6332178, -7.1791039, 7.1814880
25: -14.6286345, -6.9565859, -14.6423340, -6.9528637, -4.1908417, 4.2074127
26: -14.6163616, -7.8176880, -14.6212635, -7.7979898, -6.5639458, 6.5545998
27: -14.6403618, -9.5452929, -14.6333294, -9.5231705, -4.0781136, 4.0632401
28: -10.0294352, -1.4219750, -10.0249710, -1.4231704, -6.1681213, 6.1428490
29: -45.5954819, -36.8390121, -45.5850449, -36.8072548, -5.0293312, 5.0094452
30: -32.2178268, -23.0135956, -32.1877861, -22.9974174, -5.0293350, 5.0128994
31: -32.2471466, -23.5202789, -32.2443542, -23.5100632, -6.3212357, 6.3134422
32: 7.7147593, 13.6704035, 7.6999989, 13.6753855, -4.1525936, 4.1764374
33: 4.6360044, 16.3132401, 4.5866923, 16.3122807, -6.6624908, 6.7259178
34: 20.5491028, 30.9900665, 20.5249634, 30.9899139, -5.7198582, 5.7598343
35: 16.5213032, 26.8666344, 16.4880219, 26.8649235, -5.4195099, 5.4720955
36: 28.8232079, 35.1260757, 28.7955322, 35.1252289, -3.4195862, 3.4526129
37: 11.0453224, 20.1179237, 11.0042782, 20.1167660, -5.9363480, 5.9908180
38: 34.8859596, 43.7014732, 34.8407173, 43.6909943, -6.0341568, 6.0810013
39: 9.0161476, 18.5187016, 8.9640465, 18.5083904, -6.5096016, 6.5715942
40: 15.7945385, 25.1284676, 15.7702188, 25.1273880, -5.7952499, 5.8269501
41: 6.7316031, 13.2274380, 6.7150764, 13.2265558, -5.0084114, 5.0385933
42: -12.3774128, -3.4515114, -12.3990746, -3.4542990, -7.0273209, 7.0657043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=78, inp2_unstable=78, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1563

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6090362, upper bound: 3.6239756
time: 5.40 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6090362, upper bound: 3.6387526
time: 4.86 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 12.30 seconds
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5858404, upper bound: 3.6387045
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5978590, upper bound: 3.6387564
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5858404, upper bound: 3.6387048
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5978590, upper bound: 3.6387568
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5970165, upper bound: 3.6387045
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.6090363, upper bound: 3.6387564
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5970165, upper bound: 3.6387048
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.6090363, upper bound: 3.6387568
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5862844, upper bound: 3.6239230
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5862844, upper bound: 3.6387003
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5862844, upper bound: 3.6239234
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5862844, upper bound: 3.6387007
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5983001, upper bound: 3.6239753
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5983001, upper bound: 3.6387523
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5983001, upper bound: 3.6239756
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5983001, upper bound: 3.6387526
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5970164, upper bound: 3.6239230
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5970164, upper bound: 3.6387003
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5970164, upper bound: 3.6239234
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.5970164, upper bound: 3.6387007
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.6090362, upper bound: 3.6239753
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.6090362, upper bound: 3.6387523
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.6090362, upper bound: 3.6239756
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 12.30
Output dim: 38, lower bound: -3.6090362, upper bound: 3.6387526

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.5750237, -8.4785156, -21.5139465, -8.4843254, -10.4118118, 10.3498917
1: -21.4301071, -12.2317991, -21.4194469, -12.2499704, -5.2557430, 5.2642860
2: -12.3954039, -5.7779965, -12.3843708, -5.7862096, -4.2538300, 4.2560730
3: -12.0102463, -4.1680274, -11.9921074, -4.1953425, -5.3508186, 5.3578682
4: -10.2868824, 0.0088307, -10.2625751, -0.0384138, -6.0209885, 6.0502853
5: -13.5582581, -4.0435452, -13.5456324, -4.0676637, -6.1330032, 6.1399078
6: -8.3330269, 0.5389938, -8.2912369, 0.5194404, -6.4490204, 6.4222488
7: -32.1567650, -22.0510006, -32.1411819, -22.0937767, -5.8103714, 5.8401909
8: -18.8054447, -9.1142845, -18.7872314, -9.1582146, -5.1843338, 5.2122021
9: -5.3418665, 1.3935041, -5.3048229, 1.3796690, -4.0508537, 4.0209274
10: -36.1340103, -27.7547913, -36.1188431, -27.7886448, -5.2401428, 5.2508926
11: -55.1223869, -44.7847443, -55.0803070, -44.8627892, -4.9213314, 4.9611721
12: -11.5789146, -4.5864382, -11.5637093, -4.6127682, -6.2081871, 6.2074661
13: 0.8842941, 8.0089922, 0.9083222, 7.9730778, -5.2693939, 5.2853165
14: -71.0839081, -57.9483109, -71.0726013, -57.9757767, -8.2209206, 8.2437592
15: -8.9100380, 0.9088998, -8.8927593, 0.8702087, -4.8593159, 4.8895798
16: -33.5593987, -23.9753838, -33.5021553, -24.0121746, -6.4508896, 6.4101868
17: -88.6775513, -72.3884888, -88.6550446, -72.4888458, -8.1526108, 8.2332115
18: -4.1688123, 1.0692143, -4.1385260, 1.0402696, -3.3778877, 3.3770638
19: -30.5213985, -23.2031479, -30.4953613, -23.2291870, -4.6400871, 4.6428261
20: -11.1719656, -5.1542397, -11.1622238, -5.1711960, -4.9156227, 4.9216022
21: -43.5405388, -35.0551071, -43.5080872, -35.0933189, -4.2535095, 4.2545376
22: -27.0031242, -19.5270481, -26.9867134, -19.5681610, -4.3216419, 4.3507881
23: -20.8490009, -12.5103016, -20.8031960, -12.5399752, -4.7738991, 4.7544422
24: -16.8539429, -7.6412559, -16.8157845, -7.6662526, -7.1476059, 7.1328011
25: -14.6370754, -6.9550867, -14.6051483, -6.9833307, -4.1906261, 4.1867981
26: -14.6173697, -7.8006954, -14.6100454, -7.8290720, -6.5216331, 6.5339394
27: -14.6271973, -9.5282946, -14.6035137, -9.5700550, -4.0427647, 4.0591640
28: -10.0198822, -1.4310944, -10.0020218, -1.4382144, -6.1354408, 6.1275787
29: -45.5773468, -36.8119354, -45.5452271, -36.8731232, -4.9819260, 5.0117989
30: -32.1765366, -23.0117531, -32.1433716, -23.0594006, -4.9684601, 4.9875641
31: -32.2327423, -23.5148983, -32.1854706, -23.5527992, -6.2955666, 6.2817764
32: 7.7022777, 13.6732845, 7.7297406, 13.6653099, -4.1595860, 4.1367817
33: 4.5916033, 16.3114929, 4.6629286, 16.2967491, -6.7077637, 6.6400795
34: 20.5318241, 30.9848976, 20.6053562, 30.9537277, -5.7437515, 5.6981487
35: 16.4960938, 26.8593903, 16.5756969, 26.8361034, -5.4560528, 5.3944893
36: 28.7976780, 35.1238785, 28.8416672, 35.1105499, -3.4461842, 3.4157677
37: 11.0113440, 20.1137028, 11.0779667, 20.0993805, -5.9674950, 5.9087753
38: 34.8502731, 43.6855011, 34.9240265, 43.6486588, -6.0440903, 6.0062141
39: 8.9736195, 18.5062790, 9.0366926, 18.4937897, -6.5418663, 6.4877930
40: 15.7763748, 25.1237221, 15.8292780, 25.1117477, -5.8043404, 5.7575893
41: 6.7206755, 13.2224884, 6.7579956, 13.2128239, -5.0156097, 4.9844360
42: -12.3983116, -3.4556236, -12.3678608, -3.4730990, -7.0431786, 7.0290642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5856645, upper bound: 3.6255129
time: 5.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6385325
time: 4.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.5843506, -8.4756966, -21.5138416, -8.4842653, -10.4436798, 10.3474884
1: -21.4318581, -12.2294598, -21.4194660, -12.2502279, -5.2620010, 5.2654419
2: -12.3984156, -5.7751417, -12.3844404, -5.7862206, -4.2631607, 4.2548790
3: -12.0142393, -4.1592116, -11.9922361, -4.1953115, -5.3547859, 5.3667107
4: -10.2975636, 0.0297084, -10.2629757, -0.0383930, -6.0312386, 6.0716400
5: -13.5647259, -4.0362530, -13.5458107, -4.0676360, -6.1426163, 6.1380882
6: -8.3474178, 0.5442929, -8.2912359, 0.5196671, -6.4636917, 6.4277725
7: -32.1577263, -22.0524101, -32.1411705, -22.0955391, -5.8134117, 5.8447361
8: -18.8170795, -9.0785360, -18.7877026, -9.1582117, -5.1956730, 5.2480946
9: -5.3464231, 1.4031341, -5.3049736, 1.3796936, -4.0551949, 4.0305767
10: -36.1394882, -27.7483501, -36.1189690, -27.7886410, -5.2459011, 5.2528706
11: -55.1576004, -44.7712173, -55.0803108, -44.8622208, -4.9572716, 4.9744129
12: -11.5809937, -4.5789261, -11.5636978, -4.6125932, -6.2090950, 6.2143211
13: 0.8782739, 8.0283098, 0.9080648, 7.9731522, -5.2755241, 5.3051605
14: -71.0880127, -57.9414444, -71.0726166, -57.9757462, -8.2345085, 8.2489967
15: -8.9229612, 0.9305167, -8.8931780, 0.8701630, -4.8723011, 4.9115620
16: -33.5828133, -23.9669914, -33.5021973, -24.0118027, -6.4745979, 6.4179497
17: -88.6778717, -72.3855209, -88.6549988, -72.4901581, -8.1527557, 8.2383156
18: -4.1924562, 1.0741274, -4.1385632, 1.0404747, -3.4016933, 3.3818626
19: -30.5304298, -23.2002068, -30.4954071, -23.2290936, -4.6496410, 4.6453571
20: -11.1740665, -5.1530704, -11.1622477, -5.1719360, -4.9153099, 4.9258862
21: -43.5573654, -35.0483246, -43.5081253, -35.0930138, -4.2706051, 4.2612572
22: -27.0065804, -19.5264034, -26.9865513, -19.5681667, -4.3255863, 4.3507843
23: -20.8684807, -12.5012913, -20.8031979, -12.5396385, -4.7937317, 4.7640305
24: -16.8738327, -7.6338019, -16.8158512, -7.6659331, -7.1677246, 7.1402473
25: -14.6410999, -6.9524579, -14.6052036, -6.9832568, -4.1951847, 4.1896019
26: -14.6217575, -7.7972884, -14.6101437, -7.8294048, -6.5218086, 6.5523338
27: -14.6385727, -9.5230255, -14.6035900, -9.5698643, -4.0537415, 4.0645065
28: -10.0346327, -1.4226017, -10.0020599, -1.4378992, -6.1355858, 6.1351471
29: -45.5903015, -36.8076935, -45.5452766, -36.8729324, -4.9949780, 5.0151100
30: -32.2088470, -22.9960213, -32.1433716, -23.0588531, -5.0013485, 5.0035305
31: -32.2520714, -23.5105171, -32.1855545, -23.5526009, -6.3162956, 6.2856636
32: 7.6973286, 13.6754866, 7.7297392, 13.6653776, -4.1647491, 4.1389236
33: 4.5888758, 16.3127708, 4.6636262, 16.2967510, -6.7134476, 6.6413269
34: 20.5195999, 30.9898338, 20.6053276, 30.9539185, -5.7561817, 5.7026939
35: 16.4822540, 26.8645096, 16.5756073, 26.8363247, -5.4700565, 5.3995190
36: 28.7955742, 35.1255150, 28.8416367, 35.1105804, -3.4482117, 3.4172649
37: 11.0018606, 20.1172562, 11.0779076, 20.0994930, -5.9774055, 5.9119759
38: 34.8394852, 43.6993713, 34.9236870, 43.6486626, -6.0551109, 6.0204201
39: 8.9643173, 18.5180340, 9.0364780, 18.4937935, -6.5509796, 6.4972000
40: 15.7654848, 25.1285820, 15.8291588, 25.1117649, -5.8143826, 5.7656593
41: 6.7117872, 13.2268267, 6.7579741, 13.2129784, -5.0245705, 4.9887314
42: -12.3983173, -3.4529111, -12.3676147, -3.4731064, -7.0433350, 7.0348854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5976835, upper bound: 3.6255653
time: 4.57 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5976869, upper bound: 3.6385844
time: 4.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5750237, -8.4785156, -21.5183945, -8.4835539, -10.4119568, 10.3536377
1: -21.4301071, -12.2317991, -21.4227467, -12.2464933, -5.2572594, 5.2656994
2: -12.3954039, -5.7779965, -12.3897514, -5.7842312, -4.2496796, 4.2553902
3: -12.0102463, -4.1680274, -12.0046101, -4.1859598, -5.3477554, 5.3586655
4: -10.2868824, 0.0088307, -10.2715454, -0.0352497, -6.0183220, 6.0520973
5: -13.5582581, -4.0435452, -13.5479088, -4.0644531, -6.1355515, 6.1417351
6: -8.3330269, 0.5389938, -8.2966614, 0.5287998, -6.4516754, 6.4213219
7: -32.1567650, -22.0510006, -32.1474609, -22.0855141, -5.8146019, 5.8436394
8: -18.8054447, -9.1142845, -18.7951508, -9.1527100, -5.1842499, 5.2134781
9: -5.3418665, 1.3935041, -5.3126678, 1.3820503, -4.0553513, 4.0321121
10: -36.1340103, -27.7547913, -36.1278534, -27.7748756, -5.2401772, 5.2474136
11: -55.1223869, -44.7847443, -55.0996780, -44.8272858, -4.9191628, 4.9421177
12: -11.5789146, -4.5864382, -11.5709515, -4.6035142, -6.2109070, 6.2098389
13: 0.8842941, 8.0089922, 0.8986046, 7.9776468, -5.2689934, 5.2887993
14: -71.0839081, -57.9483109, -71.0753632, -57.9730148, -8.2258797, 8.2483788
15: -8.9100380, 0.9088998, -8.8998899, 0.8726192, -4.8530712, 4.8854504
16: -33.5593987, -23.9753838, -33.5221748, -23.9895325, -6.4537086, 6.4095421
17: -88.6775513, -72.3884888, -88.6660309, -72.4702911, -8.1551857, 8.2305145
18: -4.1688123, 1.0692143, -4.1414704, 1.0453594, -3.3815994, 3.3790226
19: -30.5213985, -23.2031479, -30.5091228, -23.2104111, -4.6377659, 4.6356945
20: -11.1719656, -5.1542397, -11.1683931, -5.1615143, -4.9169655, 4.9198723
21: -43.5405388, -35.0551071, -43.5253944, -35.0665054, -4.2454700, 4.2378922
22: -27.0031242, -19.5270481, -26.9932880, -19.5579205, -4.3239346, 4.3492298
23: -20.8490009, -12.5103016, -20.8101654, -12.5259743, -4.7750683, 4.7495975
24: -16.8539429, -7.6412559, -16.8196125, -7.6563630, -7.1522522, 7.1326447
25: -14.6370754, -6.9550867, -14.6144533, -6.9669008, -4.1910362, 4.1799889
26: -14.6173697, -7.8006954, -14.6122284, -7.8253980, -6.5242157, 6.5348473
27: -14.6271973, -9.5282946, -14.6134605, -9.5545006, -4.0430832, 4.0552902
28: -10.0198822, -1.4310944, -10.0049791, -1.4352902, -6.1390038, 6.1306572
29: -45.5773468, -36.8119354, -45.5580978, -36.8499527, -4.9830780, 5.0022087
30: -32.1765366, -23.0117531, -32.1543236, -23.0378761, -4.9711590, 4.9785156
31: -32.2327423, -23.5148983, -32.2007141, -23.5309620, -6.2917442, 6.2731934
32: 7.7022777, 13.6732845, 7.7261553, 13.6658726, -4.1598015, 4.1401367
33: 4.5916033, 16.3114929, 4.6512790, 16.2999973, -6.7074051, 6.6517906
34: 20.5318241, 30.9848976, 20.5779095, 30.9723129, -5.7352142, 5.6978493
35: 16.4960938, 26.8593903, 16.5532532, 26.8487415, -5.4463024, 5.3954811
36: 28.7976780, 35.1238785, 28.8279781, 35.1191254, -3.4435387, 3.4160728
37: 11.0113440, 20.1137028, 11.0714064, 20.1032448, -5.9686203, 5.9137650
38: 34.8502731, 43.6855011, 34.9028625, 43.6638947, -6.0443954, 6.0113258
39: 8.9736195, 18.5062790, 9.0273943, 18.4946556, -6.5445900, 6.4985123
40: 15.7763748, 25.1237221, 15.8181696, 25.1135674, -5.8038979, 5.7675819
41: 6.7206755, 13.2224884, 6.7549715, 13.2157459, -5.0171661, 4.9864120
42: -12.3983116, -3.4556236, -12.3760672, -3.4602304, -7.0448761, 7.0267563

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5856645, upper bound: 3.6255132
time: 4.95 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6385328
time: 5.45 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5843506, -8.4756966, -21.5182686, -8.4834642, -10.4438782, 10.3512192
1: -21.4318581, -12.2294598, -21.4227753, -12.2467318, -5.2635212, 5.2668648
2: -12.3984156, -5.7751417, -12.3898506, -5.7842422, -4.2590046, 4.2541904
3: -12.0142393, -4.1592116, -12.0047579, -4.1858840, -5.3517227, 5.3675079
4: -10.2975636, 0.0297084, -10.2719355, -0.0352802, -6.0285721, 6.0734367
5: -13.5647259, -4.0362530, -13.5480862, -4.0644026, -6.1451797, 6.1399345
6: -8.3474178, 0.5442929, -8.2966499, 0.5290428, -6.4663544, 6.4268456
7: -32.1577263, -22.0524101, -32.1474419, -22.0872765, -5.8176270, 5.8482018
8: -18.8170795, -9.0785360, -18.7956543, -9.1527157, -5.1955948, 5.2493458
9: -5.3464231, 1.4031341, -5.3128247, 1.3820906, -4.0596886, 4.0417690
10: -36.1394882, -27.7483501, -36.1279678, -27.7748547, -5.2459335, 5.2493916
11: -55.1576004, -44.7712173, -55.0997009, -44.8266602, -4.9551067, 4.9553642
12: -11.5809937, -4.5789261, -11.5709343, -4.6033316, -6.2118149, 6.2166901
13: 0.8782739, 8.0283098, 0.8983476, 7.9777355, -5.2751350, 5.3086548
14: -71.0880127, -57.9414444, -71.0753708, -57.9730225, -8.2394905, 8.2536278
15: -8.9229612, 0.9305167, -8.9003181, 0.8726268, -4.8660870, 4.9074364
16: -33.5828133, -23.9669914, -33.5221863, -23.9891586, -6.4774055, 6.4173050
17: -88.6778717, -72.3855209, -88.6659698, -72.4715576, -8.1553421, 8.2356377
18: -4.1924562, 1.0741274, -4.1415000, 1.0455842, -3.4054203, 3.3838234
19: -30.5304298, -23.2002068, -30.5091743, -23.2103233, -4.6473007, 4.6382275
20: -11.1740665, -5.1530704, -11.1683931, -5.1622663, -4.9166412, 4.9241562
21: -43.5573654, -35.0483246, -43.5254097, -35.0662003, -4.2625713, 4.2446194
22: -27.0065804, -19.5264034, -26.9931221, -19.5579147, -4.3278561, 4.3492241
23: -20.8684807, -12.5012913, -20.8101730, -12.5256500, -4.7948952, 4.7591953
24: -16.8738327, -7.6338019, -16.8196754, -7.6560326, -7.1723709, 7.1400948
25: -14.6410999, -6.9524579, -14.6145163, -6.9668369, -4.1955929, 4.1828041
26: -14.6217575, -7.7972884, -14.6123266, -7.8257399, -6.5243874, 6.5532341
27: -14.6385727, -9.5230255, -14.6135359, -9.5543079, -4.0540543, 4.0606422
28: -10.0346327, -1.4226017, -10.0049953, -1.4349835, -6.1391411, 6.1382446
29: -45.5903015, -36.8076935, -45.5581169, -36.8497543, -4.9961205, 5.0055294
30: -32.2088470, -22.9960213, -32.1543579, -23.0372963, -5.0040321, 4.9945126
31: -32.2520714, -23.5105171, -32.2007904, -23.5307617, -6.3124390, 6.2770767
32: 7.6973286, 13.6754866, 7.7261524, 13.6659508, -4.1649666, 4.1422672
33: 4.5888758, 16.3127708, 4.6519794, 16.2999897, -6.7131157, 6.6530266
34: 20.5195999, 30.9898338, 20.5778809, 30.9725113, -5.7476463, 5.7023888
35: 16.4822540, 26.8645096, 16.5531807, 26.8489552, -5.4603100, 5.4005165
36: 28.7955742, 35.1255150, 28.8279266, 35.1191635, -3.4455643, 3.4175644
37: 11.0018606, 20.1172562, 11.0713634, 20.1033554, -5.9785194, 5.9169769
38: 34.8394852, 43.6993713, 34.9024963, 43.6639214, -6.0554352, 6.0255432
39: 8.9643173, 18.5180340, 9.0272503, 18.4946404, -6.5536995, 6.5079193
40: 15.7654848, 25.1285820, 15.8180761, 25.1135597, -5.8139343, 5.7756405
41: 6.7117872, 13.2268267, 6.7549329, 13.2159081, -5.0261230, 4.9907036
42: -12.3983173, -3.4529111, -12.3758059, -3.4602318, -7.0450668, 7.0325890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5976835, upper bound: 3.6255656
time: 4.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5976869, upper bound: 3.6385848
time: 4.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -21.5799389, -8.4783983, -21.5306873, -8.4791422, -10.4639893, 10.3695145
1: -21.4305916, -12.2307425, -21.4229050, -12.2471285, -5.2771072, 5.2671814
2: -12.3957453, -5.7772293, -12.3875608, -5.7841315, -4.2739429, 4.2562599
3: -12.0103512, -4.1638799, -11.9985218, -4.1823683, -5.3574905, 5.3688126
4: -10.2875633, 0.0170419, -10.2778301, -0.0153747, -6.0223885, 6.0762196
5: -13.5588303, -4.0410147, -13.5524960, -4.0596027, -6.1439972, 6.1411743
6: -8.3393526, 0.5390913, -8.3103504, 0.5284593, -6.4650269, 6.4371986
7: -32.1569710, -22.0493965, -32.1428223, -22.0890942, -5.8208160, 5.8402786
8: -18.8056793, -9.0996323, -18.8059902, -9.1166620, -5.1975479, 5.2469807
9: -5.3422995, 1.3972287, -5.3129148, 1.3907449, -4.0531235, 4.0335712
10: -36.1350937, -27.7533417, -36.1252594, -27.7835426, -5.2471657, 5.2558689
11: -55.1331444, -44.7844772, -55.1097832, -44.8455658, -4.9503956, 4.9646626
12: -11.5812206, -4.5857434, -11.5705299, -4.6047306, -6.2179756, 6.2081528
13: 0.8841795, 8.0168390, 0.8978253, 7.9970837, -5.2864494, 5.3038635
14: -71.0844421, -57.9469032, -71.0759506, -57.9724655, -8.2471275, 8.2472992
15: -8.9113598, 0.9160380, -8.9081068, 0.8904734, -4.8635387, 4.9144440
16: -33.5681915, -23.9753685, -33.5288506, -23.9986725, -6.4722137, 6.4258003
17: -88.6782608, -72.3873749, -88.6568756, -72.4798584, -8.1610031, 8.2367477
18: -4.1782570, 1.0692384, -4.1648669, 1.0486560, -3.3955917, 3.3902340
19: -30.5252438, -23.2031479, -30.5071449, -23.2239037, -4.6490383, 4.6457329
20: -11.1722183, -5.1537428, -11.1632996, -5.1679201, -4.9173126, 4.9363384
21: -43.5464897, -35.0548210, -43.5255089, -35.0831032, -4.2711258, 4.2568378
22: -27.0053978, -19.5270081, -26.9939041, -19.5652351, -4.3277016, 4.3528919
23: -20.8570042, -12.5094566, -20.8262215, -12.5252151, -4.7980919, 4.7623920
24: -16.8634377, -7.6411905, -16.8440380, -7.6509571, -7.1724396, 7.1505013
25: -14.6406746, -6.9546757, -14.6164360, -6.9759293, -4.2028084, 4.1915245
26: -14.6178160, -7.8000331, -14.6121483, -7.8239937, -6.5252876, 6.5678864
27: -14.6311989, -9.5279312, -14.6168165, -9.5619974, -4.0548077, 4.0669460
28: -10.0234060, -1.4302382, -10.0124903, -1.4276531, -6.1357193, 6.1456184
29: -45.5841751, -36.8119049, -45.5654602, -36.8624573, -4.9990292, 5.0217800
30: -32.1872025, -23.0102692, -32.1731873, -23.0392570, -5.0028496, 4.9926167
31: -32.2416992, -23.5148640, -32.2128258, -23.5425301, -6.3147125, 6.3039627
32: 7.7005358, 13.6733665, 7.7237883, 13.6681805, -4.1676979, 4.1427059
33: 4.5875444, 16.3115501, 4.6497450, 16.3023682, -6.7158737, 6.6458092
34: 20.5262260, 30.9849586, 20.5884342, 30.9624596, -5.7581539, 5.7030468
35: 16.4898014, 26.8594093, 16.5563927, 26.8459892, -5.4722137, 5.3998451
36: 28.7968864, 35.1240959, 28.8382969, 35.1130562, -3.4486275, 3.4179611
37: 11.0059967, 20.1138363, 11.0606594, 20.1063614, -5.9805069, 5.9188843
38: 34.8492470, 43.6903763, 34.9107666, 43.6631203, -6.0486450, 6.0253105
39: 8.9730263, 18.5080757, 9.0267715, 18.4988136, -6.5435295, 6.4972687
40: 15.7730398, 25.1265812, 15.8146114, 25.1203632, -5.8147030, 5.7767620
41: 6.7161646, 13.2225990, 6.7436519, 13.2204132, -5.0277977, 4.9943123
42: -12.3986959, -3.4551196, -12.3688993, -3.4700868, -7.0456390, 7.0308762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5968402, upper bound: 3.6255129
time: 5.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5968442, upper bound: 3.6385325
time: 5.04 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.5892391, -8.4756012, -21.5306015, -8.4790621, -10.4958954, 10.3670769
1: -21.4323177, -12.2284698, -21.4229240, -12.2474022, -5.2833652, 5.2683487
2: -12.3987455, -5.7743678, -12.3876495, -5.7841420, -4.2832794, 4.2550545
3: -12.0143633, -4.1550665, -11.9986534, -4.1823239, -5.3614540, 5.3776550
4: -10.2982531, 0.0379872, -10.2781878, -0.0153816, -6.0326424, 6.0975666
5: -13.5653143, -4.0336819, -13.5526800, -4.0595493, -6.1536713, 6.1393623
6: -8.3537025, 0.5444114, -8.3103790, 0.5286711, -6.4796944, 6.4426994
7: -32.1579285, -22.0508003, -32.1428070, -22.0908718, -5.8238373, 5.8448219
8: -18.8172894, -9.0638571, -18.8065090, -9.1166296, -5.2089062, 5.2829094
9: -5.3468494, 1.4068569, -5.3130741, 1.3907789, -4.0574646, 4.0432301
10: -36.1405792, -27.7468758, -36.1253891, -27.7834969, -5.2529984, 5.2578640
11: -55.1683350, -44.7709351, -55.1097794, -44.8449821, -4.9863358, 4.9778976
12: -11.5832825, -4.5782328, -11.5705204, -4.6045423, -6.2188835, 6.2150154
13: 0.8781695, 8.0361900, 0.8975673, 7.9971619, -5.2925987, 5.3237267
14: -71.0885315, -57.9400520, -71.0760040, -57.9724159, -8.2607193, 8.2525177
15: -8.9242659, 0.9376454, -8.9085121, 0.8904819, -4.8765278, 4.9364300
16: -33.5916405, -23.9669399, -33.5288849, -23.9983063, -6.4959259, 6.4335785
17: -88.6785278, -72.3843613, -88.6568069, -72.4810867, -8.1611595, 8.2418633
18: -4.2019310, 1.0741618, -4.1649017, 1.0488620, -3.4194622, 3.3950329
19: -30.5342789, -23.2002144, -30.5071793, -23.2238159, -4.6586075, 4.6482868
20: -11.1743069, -5.1525445, -11.1633034, -5.1686859, -4.9170036, 4.9406300
21: -43.5633163, -35.0480652, -43.5255280, -35.0828056, -4.2882328, 4.2635670
22: -27.0088654, -19.5263672, -26.9937248, -19.5652370, -4.3316345, 4.3529091
23: -20.8764839, -12.5004606, -20.8262329, -12.5249138, -4.8179398, 4.7719860
24: -16.8833370, -7.6337323, -16.8440990, -7.6506424, -7.1925888, 7.1579475
25: -14.6447258, -6.9520330, -14.6165009, -6.9758596, -4.2073746, 4.1943321
26: -14.6222143, -7.7966223, -14.6122484, -7.8243265, -6.5254745, 6.5862846
27: -14.6425695, -9.5226555, -14.6169062, -9.5618000, -4.0657921, 4.0723000
28: -10.0381498, -1.4217329, -10.0125265, -1.4273679, -6.1358490, 6.1532288
29: -45.5971413, -36.8076477, -45.5655060, -36.8622589, -5.0120716, 5.0251141
30: -32.2195282, -22.9945526, -32.1731796, -23.0386753, -5.0357170, 5.0085831
31: -32.2610741, -23.5105000, -32.2129440, -23.5423546, -6.3354416, 6.3078537
32: 7.6955786, 13.6755705, 7.7237864, 13.6682577, -4.1728649, 4.1448517
33: 4.5848188, 16.3128719, 4.6504955, 16.3023834, -6.7215843, 6.6470528
34: 20.5139809, 30.9898949, 20.5883694, 30.9626427, -5.7705917, 5.7075787
35: 16.4759769, 26.8645325, 16.5563354, 26.8462143, -5.4862213, 5.4048615
36: 28.7947769, 35.1257401, 28.8382301, 35.1130829, -3.4506636, 3.4194689
37: 10.9964991, 20.1173649, 11.0606022, 20.1064835, -5.9904366, 5.9221230
38: 34.8384972, 43.7042809, 34.9104118, 43.6631165, -6.0596695, 6.0395279
39: 8.9637241, 18.5198250, 9.0265656, 18.4988098, -6.5526428, 6.5066986
40: 15.7621412, 25.1314354, 15.8145075, 25.1203804, -5.8247395, 5.7848358
41: 6.7072802, 13.2269325, 6.7436390, 13.2205744, -5.0367699, 4.9986153
42: -12.3986940, -3.4524040, -12.3686752, -3.4700646, -7.0458298, 7.0367317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6088602, upper bound: 3.6255653
time: 5.08 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6088644, upper bound: 3.6385844
time: 5.04 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5799389, -8.4783983, -21.5352020, -8.4783583, -10.4641571, 10.3732605
1: -21.4305916, -12.2307425, -21.4261818, -12.2436457, -5.2786331, 5.2686100
2: -12.3957453, -5.7772293, -12.3929663, -5.7821712, -4.2697868, 4.2555752
3: -12.0103512, -4.1638799, -12.0110407, -4.1729355, -5.3544312, 5.3696098
4: -10.2875633, 0.0170419, -10.2867584, -0.0122633, -6.0197182, 6.0780258
5: -13.5588303, -4.0410147, -13.5547733, -4.0563893, -6.1465645, 6.1429977
6: -8.3393526, 0.5390913, -8.3157835, 0.5378244, -6.4676971, 6.4362831
7: -32.1569710, -22.0493965, -32.1490974, -22.0808487, -5.8250427, 5.8437328
8: -18.8056793, -9.0996323, -18.8139229, -9.1111374, -5.1974716, 5.2482471
9: -5.3422995, 1.3972287, -5.3207483, 1.3931412, -4.0576286, 4.0447674
10: -36.1350937, -27.7533417, -36.1342468, -27.7697964, -5.2472038, 5.2523804
11: -55.1331444, -44.7844772, -55.1291962, -44.8100815, -4.9482269, 4.9456100
12: -11.5812206, -4.5857434, -11.5777626, -4.5954776, -6.2207031, 6.2105179
13: 0.8841795, 8.0168390, 0.8880821, 8.0016584, -5.2860718, 5.3073616
14: -71.0844421, -57.9469032, -71.0787506, -57.9697418, -8.2520790, 8.2519341
15: -8.9113598, 0.9160380, -8.9152441, 0.8929033, -4.8573055, 4.9103260
16: -33.5681915, -23.9753685, -33.5488739, -23.9760399, -6.4750328, 6.4251518
17: -88.6782608, -72.3873749, -88.6678619, -72.4612274, -8.1635933, 8.2340508
18: -4.1782570, 1.0692384, -4.1678076, 1.0537565, -3.3993244, 3.3922291
19: -30.5252438, -23.2031479, -30.5209160, -23.2051563, -4.6467018, 4.6386108
20: -11.1722183, -5.1537428, -11.1694355, -5.1582460, -4.9186478, 4.9346085
21: -43.5464897, -35.0548210, -43.5428162, -35.0562820, -4.2630920, 4.2401962
22: -27.0053978, -19.5270081, -27.0004635, -19.5549927, -4.3299751, 4.3513374
23: -20.8570042, -12.5094566, -20.8331871, -12.5112104, -4.7992630, 4.7575550
24: -16.8634377, -7.6411905, -16.8478355, -7.6410594, -7.1770973, 7.1503372
25: -14.6406746, -6.9546757, -14.6257372, -6.9595141, -4.2032223, 4.1847172
26: -14.6178160, -7.8000331, -14.6143274, -7.8203239, -6.5278702, 6.5687904
27: -14.6311989, -9.5279312, -14.6267481, -9.5464306, -4.0551243, 4.0630741
28: -10.0234060, -1.4302382, -10.0154438, -1.4247278, -6.1392746, 6.1487083
29: -45.5841751, -36.8119049, -45.5783234, -36.8393021, -5.0001812, 5.0122013
30: -32.1872025, -23.0102692, -32.1841278, -23.0176907, -5.0055161, 4.9835873
31: -32.2416992, -23.5148640, -32.2280884, -23.5207176, -6.3108711, 6.2953873
32: 7.7005358, 13.6733665, 7.7202163, 13.6687460, -4.1679115, 4.1460552
33: 4.5875444, 16.3115501, 4.6380968, 16.3056278, -6.7155228, 6.6575127
34: 20.5262260, 30.9849586, 20.5609970, 30.9810333, -5.7496147, 5.7027359
35: 16.4898014, 26.8594093, 16.5339642, 26.8585968, -5.4624577, 5.4008427
36: 28.7968864, 35.1240959, 28.8245888, 35.1216278, -3.4459734, 3.4182663
37: 11.0059967, 20.1138363, 11.0541277, 20.1102486, -5.9816284, 5.9238815
38: 34.8492470, 43.6903763, 34.8896103, 43.6783791, -6.0489502, 6.0304260
39: 8.9730263, 18.5080757, 9.0175037, 18.4996643, -6.5462570, 6.5080109
40: 15.7730398, 25.1265812, 15.8035383, 25.1221333, -5.8142509, 5.7867393
41: 6.7161646, 13.2225990, 6.7406139, 13.2233620, -5.0293579, 4.9962845
42: -12.3986959, -3.4551196, -12.3770895, -3.4572082, -7.0473747, 7.0285797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5968402, upper bound: 3.6255132
time: 5.04 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5968442, upper bound: 3.6385328
time: 5.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5892391, -8.4756012, -21.5350914, -8.4783058, -10.4960938, 10.3708305
1: -21.4323177, -12.2284698, -21.4262123, -12.2439299, -5.2848911, 5.2697678
2: -12.3987455, -5.7743678, -12.3930435, -5.7821741, -4.2791195, 4.2543736
3: -12.0143633, -4.1550665, -12.0111952, -4.1729054, -5.3583870, 5.3784599
4: -10.2982531, 0.0379872, -10.2871513, -0.0122386, -6.0299644, 6.0993786
5: -13.5653143, -4.0336819, -13.5549335, -4.0563517, -6.1562462, 6.1411705
6: -8.3537025, 0.5444114, -8.3158073, 0.5380584, -6.4823608, 6.4417839
7: -32.1579285, -22.0508003, -32.1490631, -22.0826302, -5.8280525, 5.8482933
8: -18.8172894, -9.0638571, -18.8144035, -9.1111193, -5.2088165, 5.2841415
9: -5.3468494, 1.4068569, -5.3209047, 1.3931761, -4.0619640, 4.0544319
10: -36.1405792, -27.7468758, -36.1343956, -27.7697506, -5.2530327, 5.2543755
11: -55.1683350, -44.7709351, -55.1291542, -44.8094521, -4.9841728, 4.9588547
12: -11.5832825, -4.5782328, -11.5777531, -4.5953045, -6.2215881, 6.2173767
13: 0.8781695, 8.0361900, 0.8878321, 8.0017242, -5.2922134, 5.3272095
14: -71.0885315, -57.9400520, -71.0787735, -57.9696808, -8.2656937, 8.2571602
15: -8.9242659, 0.9376454, -8.9156818, 0.8929214, -4.8703251, 4.9323120
16: -33.5916405, -23.9669399, -33.5489082, -23.9756508, -6.4987221, 6.4329185
17: -88.6785278, -72.3843613, -88.6677933, -72.4624786, -8.1637459, 8.2391930
18: -4.2019310, 1.0741618, -4.1678553, 1.0539773, -3.4231949, 3.3970280
19: -30.5342789, -23.2002144, -30.5209446, -23.2050629, -4.6562653, 4.6411533
20: -11.1743069, -5.1525445, -11.1694584, -5.1590137, -4.9183464, 4.9388981
21: -43.5633163, -35.0480652, -43.5428505, -35.0559807, -4.2801991, 4.2469234
22: -27.0088654, -19.5263672, -27.0002975, -19.5549927, -4.3339081, 4.3513527
23: -20.8764839, -12.5004606, -20.8332310, -12.5108843, -4.8190994, 4.7671452
24: -16.8833370, -7.6337323, -16.8479404, -7.6407299, -7.1972275, 7.1577873
25: -14.6447258, -6.9520330, -14.6258020, -6.9594550, -4.2077827, 4.1875229
26: -14.6222143, -7.7966223, -14.6144505, -7.8206463, -6.5280533, 6.5871811
27: -14.6425695, -9.5226555, -14.6268387, -9.5462475, -4.0661144, 4.0684280
28: -10.0381498, -1.4217329, -10.0154724, -1.4244425, -6.1394081, 6.1563339
29: -45.5971413, -36.8076477, -45.5783577, -36.8390808, -5.0132065, 5.0155296
30: -32.2195282, -22.9945526, -32.1841507, -23.0171490, -5.0383949, 4.9995708
31: -32.2610741, -23.5105000, -32.2281723, -23.5204945, -6.3315811, 6.2992783
32: 7.6955786, 13.6755705, 7.7202106, 13.6688185, -4.1730804, 4.1481876
33: 4.5848188, 16.3128719, 4.6388054, 16.3056221, -6.7212429, 6.6587524
34: 20.5139809, 30.9898949, 20.5609474, 30.9812317, -5.7620430, 5.7072811
35: 16.4759769, 26.8645325, 16.5338821, 26.8588295, -5.4764595, 5.4058628
36: 28.7947769, 35.1257401, 28.8245468, 35.1216812, -3.4480028, 3.4197617
37: 10.9964991, 20.1173649, 11.0540934, 20.1103630, -5.9915619, 5.9271011
38: 34.8384972, 43.7042809, 34.8892403, 43.6783600, -6.0599823, 6.0446243
39: 8.9637241, 18.5198250, 9.0172987, 18.4996719, -6.5553741, 6.5174408
40: 15.7621412, 25.1314354, 15.8034391, 25.1221676, -5.8242970, 5.7948189
41: 6.7072802, 13.2269325, 6.7405834, 13.2235298, -5.0383263, 5.0005722
42: -12.3986940, -3.4524040, -12.3768816, -3.4572093, -7.0475616, 7.0344124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6088602, upper bound: 3.6255656
time: 6.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6088644, upper bound: 3.6385848
time: 6.13 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -21.5620289, -8.4769249, -21.5575619, -8.4788246, -10.4021072, 10.4231491
1: -21.4274750, -12.2362194, -21.4198837, -12.2393770, -5.2764091, 5.2823792
2: -12.3931580, -5.7765284, -12.3882227, -5.7794342, -4.2609558, 4.2684669
3: -12.0098305, -4.1673656, -11.9980583, -4.1766605, -5.3668022, 5.3582001
4: -10.2834845, 0.0054761, -10.2768841, 0.0020967, -6.0560112, 6.0295563
5: -13.5547352, -4.0438614, -13.5548630, -4.0479512, -6.1338882, 6.1422501
6: -8.3184872, 0.5364139, -8.3125887, 0.5209942, -6.4475479, 6.4613190
7: -32.1555824, -22.0550804, -32.1448746, -22.0627384, -5.8267784, 5.8259926
8: -18.8051453, -9.1060276, -18.8043251, -9.1110420, -5.2313957, 5.2078857
9: -5.3312602, 1.3938025, -5.3247309, 1.3920209, -4.0470676, 4.0317841
10: -36.1338425, -27.7651405, -36.1200638, -27.7779655, -5.2568645, 5.2566128
11: -55.1317139, -44.8025818, -55.1072006, -44.8254166, -4.9332790, 4.9584503
12: -11.5782747, -4.5903549, -11.5677814, -4.5959678, -6.2306099, 6.2344170
13: 0.8903516, 8.0122757, 0.8978887, 8.0090866, -5.3020439, 5.2912292
14: -71.0809784, -57.9607620, -71.0666275, -57.9601440, -8.2420959, 8.2388000
15: -8.9083509, 0.9021177, -8.9013691, 0.9001780, -4.8861675, 4.8653107
16: -33.5565033, -23.9749908, -33.5346909, -23.9897346, -6.4441986, 6.4538193
17: -88.6748047, -72.4196930, -88.6462250, -72.4369049, -8.1787720, 8.1640053
18: -4.1768856, 1.0590160, -4.1720448, 1.0583980, -3.3874588, 3.3959045
19: -30.5247707, -23.2034302, -30.5116272, -23.2194901, -4.6471519, 4.6567917
20: -11.1715593, -5.1565609, -11.1661921, -5.1661739, -4.9334297, 4.9304085
21: -43.5467682, -35.0574150, -43.5285683, -35.0783005, -4.2609138, 4.2796574
22: -27.0042152, -19.5390244, -26.9935112, -19.5495949, -4.3293171, 4.3329735
23: -20.8513756, -12.5125999, -20.8451347, -12.5220356, -4.7671814, 4.7863064
24: -16.8589420, -7.6444864, -16.8563118, -7.6503592, -7.1537552, 7.1660690
25: -14.6370058, -6.9580369, -14.6287804, -6.9747448, -4.1873016, 4.2036667
26: -14.6170759, -7.8186388, -14.6107121, -7.8224502, -6.5440178, 6.5285912
27: -14.6302700, -9.5348797, -14.6203070, -9.5464516, -4.0562229, 4.0625153
28: -10.0224943, -1.4302797, -10.0201902, -1.4283487, -6.1620255, 6.1454277
29: -45.5825920, -36.8256226, -45.5637207, -36.8449249, -4.9983482, 5.0090046
30: -32.1858406, -23.0196342, -32.1731453, -23.0297089, -4.9825020, 5.0087128
31: -32.2349548, -23.5171471, -32.2211456, -23.5387020, -6.3001366, 6.3106422
32: 7.7182646, 13.6719894, 7.7223701, 13.6656990, -4.1464443, 4.1510162
33: 4.6095705, 16.3101063, 4.6212530, 16.2999992, -6.6689186, 6.6710014
34: 20.5353050, 30.9835892, 20.5625153, 30.9636688, -5.7206593, 5.7275867
35: 16.5047264, 26.8590469, 16.5262661, 26.8448353, -5.4206562, 5.4271698
36: 28.8100739, 35.1230011, 28.8228645, 35.1084938, -3.4196835, 3.4252453
37: 11.0252714, 20.1122551, 11.0316124, 20.1050797, -5.9408264, 5.9491196
38: 34.8627357, 43.6859665, 34.8766403, 43.6627350, -6.0266075, 6.0262794
39: 8.9976406, 18.5050392, 8.9998436, 18.4966908, -6.5035934, 6.5094070
40: 15.7924814, 25.1242046, 15.8019209, 25.1171761, -5.7916756, 5.7871647
41: 6.7330575, 13.2213545, 6.7361627, 13.2140303, -5.0028076, 5.0112915
42: -12.3868208, -3.4567003, -12.3785992, -3.4765055, -7.0371780, 7.0490074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5861083, upper bound: 3.6255087
time: 5.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5861122, upper bound: 3.6385283
time: 5.00 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -21.5620289, -8.4769249, -21.5620499, -8.4780226, -10.4022827, 10.4269180
1: -21.4274750, -12.2362194, -21.4231720, -12.2359066, -5.2779160, 5.2838097
2: -12.3931580, -5.7765284, -12.3936234, -5.7774405, -4.2567978, 4.2677860
3: -12.0098305, -4.1673656, -12.0105782, -4.1672211, -5.3637466, 5.3589630
4: -10.2834845, 0.0054761, -10.2858582, 0.0052235, -6.0533524, 6.0313606
5: -13.5547352, -4.0438614, -13.5571175, -4.0447292, -6.1364784, 6.1440773
6: -8.3184872, 0.5364139, -8.3179932, 0.5304065, -6.4502068, 6.4604034
7: -32.1555824, -22.0550804, -32.1511803, -22.0545273, -5.8309898, 5.8294506
8: -18.8051453, -9.1060276, -18.8122616, -9.1055126, -5.2313213, 5.2091675
9: -5.3312602, 1.3938025, -5.3325729, 1.3944352, -4.0515366, 4.0430336
10: -36.1338425, -27.7651405, -36.1290817, -27.7641716, -5.2569008, 5.2531567
11: -55.1317139, -44.8025818, -55.1266022, -44.7898941, -4.9311275, 4.9394035
12: -11.5782747, -4.5903549, -11.5749922, -4.5867014, -6.2333107, 6.2367744
13: 0.8903516, 8.0122757, 0.8881400, 8.0136786, -5.3016739, 5.2947350
14: -71.0809784, -57.9607620, -71.0693665, -57.9574127, -8.2470627, 8.2434387
15: -8.9083509, 0.9021177, -8.9085131, 0.9025865, -4.8799286, 4.8611946
16: -33.5565033, -23.9749908, -33.5547066, -23.9671135, -6.4468575, 6.4531898
17: -88.6748047, -72.4196930, -88.6572113, -72.4183044, -8.1813545, 8.1613541
18: -4.1768856, 1.0590160, -4.1749964, 1.0635304, -3.3911934, 3.3979130
19: -30.5247707, -23.2034302, -30.5254154, -23.2007561, -4.6448097, 4.6496696
20: -11.1715593, -5.1565609, -11.1723289, -5.1565018, -4.9347610, 4.9287033
21: -43.5467682, -35.0574150, -43.5458755, -35.0514870, -4.2528839, 4.2630196
22: -27.0042152, -19.5390244, -27.0000610, -19.5393333, -4.3316021, 4.3314209
23: -20.8513756, -12.5125999, -20.8521595, -12.5080109, -4.7683716, 4.7814884
24: -16.8589420, -7.6444864, -16.8601456, -7.6404505, -7.1584129, 7.1659279
25: -14.6370058, -6.9580369, -14.6381073, -6.9583263, -4.1877270, 4.1968708
26: -14.6170759, -7.8186388, -14.6129026, -7.8188000, -6.5466003, 6.5295105
27: -14.6302700, -9.5348797, -14.6302443, -9.5308876, -4.0565548, 4.0586643
28: -10.0224943, -1.4302797, -10.0231390, -1.4254522, -6.1655807, 6.1485023
29: -45.5825920, -36.8256226, -45.5765762, -36.8217545, -4.9995060, 4.9994411
30: -32.1858406, -23.0196342, -32.1841087, -23.0081635, -4.9851837, 4.9997005
31: -32.2349548, -23.5171471, -32.2363815, -23.5168648, -6.2963142, 6.3020821
32: 7.7182646, 13.6719894, 7.7188039, 13.6662626, -4.1466675, 4.1543674
33: 4.6095705, 16.3101063, 4.6095629, 16.3032532, -6.6686516, 6.6826820
34: 20.5353050, 30.9835892, 20.5350742, 30.9822235, -5.7121277, 5.7272644
35: 16.5047264, 26.8590469, 16.5038223, 26.8574753, -5.4109592, 5.4281597
36: 28.8100739, 35.1230011, 28.8091602, 35.1170807, -3.4170494, 3.4255466
37: 11.0252714, 20.1122551, 11.0251083, 20.1089668, -5.9419022, 5.9541054
38: 34.8627357, 43.6859665, 34.8554726, 43.6779671, -6.0269241, 6.0313263
39: 8.9976406, 18.5050392, 8.9905748, 18.4975662, -6.5063515, 6.5201187
40: 15.7924814, 25.1242046, 15.7908535, 25.1189575, -5.7912445, 5.7971611
41: 6.7330575, 13.2213545, 6.7331223, 13.2169800, -5.0043373, 5.0132446
42: -12.3868208, -3.4567003, -12.3867807, -3.4636152, -7.0389175, 7.0466995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5861083, upper bound: 3.6255088
time: 6.11 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5861122, upper bound: 3.6385287
time: 4.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.5712719, -8.4741116, -21.5574455, -8.4787569, -10.4340363, 10.4207230
1: -21.4292145, -12.2339258, -21.4199219, -12.2396631, -5.2826729, 5.2835121
2: -12.3961735, -5.7736363, -12.3882999, -5.7794461, -4.2702904, 4.2672729
3: -12.0138273, -4.1585484, -11.9981909, -4.1765790, -5.3707733, 5.3670692
4: -10.2942057, 0.0263834, -10.2773056, 0.0020864, -6.0662766, 6.0509033
5: -13.5612068, -4.0365548, -13.5550385, -4.0479383, -6.1435394, 6.1404419
6: -8.3328323, 0.5417212, -8.3125591, 0.5212150, -6.4622307, 6.4668198
7: -32.1565514, -22.0564919, -32.1448746, -22.0644951, -5.8297844, 5.8305225
8: -18.8167419, -9.0702438, -18.8048515, -9.1110201, -5.2427368, 5.2438068
9: -5.3357940, 1.4034190, -5.3248873, 1.3920624, -4.0514050, 4.0414524
10: -36.1393280, -27.7586975, -36.1202011, -27.7779121, -5.2625923, 5.2586021
11: -55.1669388, -44.7890472, -55.1072197, -44.8248558, -4.9692211, 4.9716911
12: -11.5803537, -4.5828381, -11.5677700, -4.5957985, -6.2315369, 6.2412643
13: 0.8843025, 8.0316133, 0.8976672, 8.0091696, -5.3081779, 5.3110733
14: -71.0850067, -57.9539261, -71.0666351, -57.9601059, -8.2557030, 8.2440186
15: -8.9212704, 0.9237528, -8.9018002, 0.9001479, -4.8991947, 4.8872814
16: -33.5799484, -23.9665871, -33.5347481, -23.9893761, -6.4679222, 6.4615936
17: -88.6750870, -72.4166565, -88.6462097, -72.4382172, -8.1789093, 8.1691170
18: -4.2005539, 1.0639176, -4.1720724, 1.0586183, -3.4113827, 3.4007015
19: -30.5337830, -23.2004662, -30.5116634, -23.2194138, -4.6566944, 4.6593323
20: -11.1736584, -5.1553574, -11.1661978, -5.1669502, -4.9331169, 4.9346905
21: -43.5636101, -35.0506020, -43.5286026, -35.0780182, -4.2780151, 4.2863693
22: -27.0076752, -19.5383930, -26.9933090, -19.5495834, -4.3332329, 4.3329792
23: -20.8708858, -12.5036287, -20.8451767, -12.5217075, -4.7870064, 4.7958927
24: -16.8788338, -7.6370068, -16.8563976, -7.6500311, -7.1738853, 7.1734886
25: -14.6410141, -6.9553967, -14.6288433, -6.9746819, -4.1918621, 4.2064762
26: -14.6214819, -7.8152094, -14.6108170, -7.8228068, -6.5441971, 6.5469971
27: -14.6416512, -9.5296001, -14.6203861, -9.5462494, -4.0672188, 4.0678596
28: -10.0372353, -1.4217491, -10.0202236, -1.4280457, -6.1621552, 6.1529694
29: -45.5955353, -36.8213921, -45.5637741, -36.8447266, -5.0113831, 5.0123329
30: -32.2181549, -23.0039368, -32.1731262, -23.0291328, -5.0153656, 5.0246716
31: -32.2543182, -23.5127544, -32.2212143, -23.5385017, -6.3208847, 6.3145180
32: 7.7133174, 13.6741982, 7.7223802, 13.6657848, -4.1516075, 4.1531658
33: 4.6068258, 16.3113785, 4.6219773, 16.3000183, -6.6745739, 6.6722336
34: 20.5230598, 30.9885406, 20.5624828, 30.9638462, -5.7330971, 5.7321281
35: 16.4908905, 26.8641472, 16.5262032, 26.8450470, -5.4346657, 5.4321861
36: 28.8079796, 35.1246490, 28.8228283, 35.1085358, -3.4217138, 3.4267435
37: 11.0158024, 20.1157951, 11.0315590, 20.1052036, -5.9507675, 5.9523277
38: 34.8519859, 43.6998825, 34.8763046, 43.6627350, -6.0376205, 6.0405121
39: 8.9883137, 18.5167923, 8.9996672, 18.4966774, -6.5127106, 6.5188103
40: 15.7816019, 25.1290550, 15.8018379, 25.1171951, -5.8016777, 5.7952366
41: 6.7241688, 13.2256765, 6.7361135, 13.2142000, -5.0117645, 5.0155983
42: -12.3867903, -3.4539888, -12.3783693, -3.4764814, -7.0373840, 7.0548515

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5981240, upper bound: 3.6255611
time: 5.17 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5981283, upper bound: 3.6385803
time: 5.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5712719, -8.4741116, -21.5619335, -8.4779844, -10.4341888, 10.4244766
1: -21.4292145, -12.2339258, -21.4232349, -12.2361889, -5.2841835, 5.2849426
2: -12.3961735, -5.7736363, -12.3937016, -5.7774534, -4.2661304, 4.2665882
3: -12.0138273, -4.1585484, -12.0107193, -4.1671848, -5.3677254, 5.3678207
4: -10.2942057, 0.0263834, -10.2862310, 0.0051928, -6.0636253, 6.0527191
5: -13.5612068, -4.0365548, -13.5573072, -4.0446939, -6.1461105, 6.1422768
6: -8.3328323, 0.5417212, -8.3180046, 0.5306039, -6.4648819, 6.4659119
7: -32.1565514, -22.0564919, -32.1511536, -22.0562706, -5.8340111, 5.8339939
8: -18.8167419, -9.0702438, -18.8127670, -9.1054840, -5.2426682, 5.2450562
9: -5.3357940, 1.4034190, -5.3327417, 1.3944460, -4.0558701, 4.0527039
10: -36.1393280, -27.7586975, -36.1292000, -27.7641220, -5.2626343, 5.2551384
11: -55.1669388, -44.7890472, -55.1266327, -44.7892914, -4.9670677, 4.9526482
12: -11.5803537, -4.5828381, -11.5749817, -4.5865474, -6.2342415, 6.2436409
13: 0.8843025, 8.0316133, 0.8879042, 8.0137701, -5.3078079, 5.3145676
14: -71.0850067, -57.9539261, -71.0694275, -57.9573975, -8.2606506, 8.2486610
15: -8.9212704, 0.9237528, -8.9089241, 0.9025879, -4.8929787, 4.8831749
16: -33.5799484, -23.9665871, -33.5547066, -23.9667244, -6.4705696, 6.4609642
17: -88.6750870, -72.4166565, -88.6571655, -72.4196320, -8.1814957, 8.1664696
18: -4.2005539, 1.0639176, -4.1750283, 1.0637357, -3.4151211, 3.4027119
19: -30.5337830, -23.2004662, -30.5254612, -23.2006531, -4.6543446, 4.6522102
20: -11.1736584, -5.1553574, -11.1723509, -5.1572657, -4.9344635, 4.9329853
21: -43.5636101, -35.0506020, -43.5459023, -35.0512009, -4.2699833, 4.2697411
22: -27.0076752, -19.5383930, -26.9999065, -19.5393543, -4.3355083, 4.3314304
23: -20.8708858, -12.5036287, -20.8521633, -12.5076981, -4.7882004, 4.7910862
24: -16.8788338, -7.6370068, -16.8602238, -7.6401300, -7.1785316, 7.1733818
25: -14.6410141, -6.9553967, -14.6381702, -6.9582491, -4.1922836, 4.1996861
26: -14.6214819, -7.8152094, -14.6129913, -7.8191376, -6.5467758, 6.5479088
27: -14.6416512, -9.5296001, -14.6303368, -9.5306873, -4.0675373, 4.0640068
28: -10.0372353, -1.4217491, -10.0231876, -1.4251261, -6.1657257, 6.1560516
29: -45.5955353, -36.8213921, -45.5766258, -36.8215790, -5.0125446, 5.0027695
30: -32.2181549, -23.0039368, -32.1841125, -23.0076027, -5.0180492, 5.0156879
31: -32.2543182, -23.5127544, -32.2364807, -23.5166626, -6.3170357, 6.3059692
32: 7.7133174, 13.6741982, 7.7188187, 13.6663446, -4.1518250, 4.1565113
33: 4.6068258, 16.3113785, 4.6103058, 16.3032646, -6.6743164, 6.6839142
34: 20.5230598, 30.9885406, 20.5350494, 30.9824219, -5.7245579, 5.7317944
35: 16.4908905, 26.8641472, 16.5037880, 26.8577023, -5.4249573, 5.4331875
36: 28.8079796, 35.1246490, 28.8091125, 35.1171227, -3.4190817, 3.4270382
37: 11.0158024, 20.1157951, 11.0250340, 20.1091003, -5.9518318, 5.9573288
38: 34.8519859, 43.6998825, 34.8551140, 43.6779861, -6.0379639, 6.0455475
39: 8.9883137, 18.5167923, 8.9903841, 18.4975471, -6.5154610, 6.5295334
40: 15.7816019, 25.1290550, 15.7907534, 25.1189976, -5.8012657, 5.8052330
41: 6.7241688, 13.2256765, 6.7330947, 13.2171421, -5.0133171, 5.0175552
42: -12.3867903, -3.4539888, -12.3865461, -3.4635925, -7.0390968, 7.0525322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5981240, upper bound: 3.6255614
time: 8.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5981283, upper bound: 3.6385806
time: 5.57 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -21.5627575, -8.4768620, -21.5776443, -8.4764938, -10.4091110, 10.4549408
1: -21.4280815, -12.2362041, -21.4283104, -12.2338972, -5.2818260, 5.2914734
2: -12.3934784, -5.7767491, -12.3923111, -5.7791834, -4.2634449, 4.2727261
3: -12.0099993, -4.1671033, -12.0012197, -4.1721773, -5.3710213, 5.3615608
4: -10.2848072, 0.0057602, -10.2874403, 0.0143511, -6.0717621, 6.0383968
5: -13.5550680, -4.0440159, -13.5608139, -4.0432534, -6.1411591, 6.1517143
6: -8.3189163, 0.5391184, -8.3348541, 0.5347388, -6.4587517, 6.4857674
7: -32.1562042, -22.0549984, -32.1511803, -22.0569839, -5.8341370, 5.8317528
8: -18.8055687, -9.1057968, -18.8088703, -9.1045027, -5.2404575, 5.2126331
9: -5.3312035, 1.3938938, -5.3382778, 1.3956294, -4.0507717, 4.0489330
10: -36.1352272, -27.7651405, -36.1290245, -27.7656555, -5.2661896, 5.2647038
11: -55.1329460, -44.8024673, -55.1142120, -44.8066673, -4.9532642, 4.9638462
12: -11.5791264, -4.5903344, -11.5742264, -4.5908351, -6.2310944, 6.2403412
13: 0.8898966, 8.0124722, 0.8880766, 8.0142431, -5.3091965, 5.3013153
14: -71.0835266, -57.9607925, -71.0830994, -57.9461517, -8.2560120, 8.2537918
15: -8.9105015, 0.9022608, -8.9135122, 0.9141097, -4.9057579, 4.8750248
16: -33.5572243, -23.9751740, -33.5495071, -23.9893913, -6.4458199, 6.4821777
17: -88.6790695, -72.4193726, -88.6676941, -72.4025345, -8.2195511, 8.1829834
18: -4.1774573, 1.0591235, -4.1762199, 1.0691659, -3.3989182, 3.4002571
19: -30.5247116, -23.2033157, -30.5130215, -23.2187500, -4.6482430, 4.6589470
20: -11.1715584, -5.1563902, -11.1672173, -5.1611094, -4.9397163, 4.9318466
21: -43.5466843, -35.0572815, -43.5303192, -35.0753326, -4.2637157, 4.2840652
22: -27.0054169, -19.5389404, -27.0007744, -19.5370903, -4.3444061, 4.3397694
23: -20.8515434, -12.5126591, -20.8512287, -12.5162983, -4.7745647, 4.7956715
24: -16.8592014, -7.6444712, -16.8613892, -7.6434731, -7.1606369, 7.1716690
25: -14.6370773, -6.9578748, -14.6329813, -6.9693565, -4.1934624, 4.2092419
26: -14.6186218, -7.8180952, -14.6189814, -7.8013301, -6.5663872, 6.5371819
27: -14.6305943, -9.5347366, -14.6232958, -9.5389233, -4.0634766, 4.0660191
28: -10.0223894, -1.4300778, -10.0219812, -1.4263923, -6.1656990, 6.1489372
29: -45.5838280, -36.8255005, -45.5721741, -36.8306465, -5.0137806, 5.0175095
30: -32.1861801, -23.0194359, -32.1767731, -23.0195427, -4.9927731, 5.0122051
31: -32.2351913, -23.5165253, -32.2290230, -23.5320835, -6.3082733, 6.3222351
32: 7.7180147, 13.6737537, 7.7035871, 13.6747360, -4.1531811, 4.1720486
33: 4.6091814, 16.3114891, 4.5977287, 16.3089962, -6.6770000, 6.7015038
34: 20.5349922, 30.9846210, 20.5524311, 30.9711399, -5.7281322, 5.7408581
35: 16.5045109, 26.8602104, 16.5105457, 26.8520508, -5.4278946, 5.4471798
36: 28.8099823, 35.1245728, 28.8093109, 35.1166229, -3.4266376, 3.4412155
37: 11.0248947, 20.1137085, 11.0108385, 20.1127663, -5.9480286, 5.9739380
38: 34.8626251, 43.6879921, 34.8622627, 43.6757545, -6.0384827, 6.0430222
39: 8.9973164, 18.5068855, 8.9735651, 18.5075397, -6.5137711, 6.5385361
40: 15.7922144, 25.1255875, 15.7814045, 25.1255913, -5.7983322, 5.8101044
41: 6.7327147, 13.2231512, 6.7181282, 13.2234402, -5.0115395, 5.0320778
42: -12.3869829, -3.4549737, -12.3911343, -3.4671793, -7.0455475, 7.0625343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5968402, upper bound: 3.6255087
time: 4.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5968441, upper bound: 3.6385283
time: 4.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -21.5627575, -8.4768620, -21.5821133, -8.4757366, -10.4092636, 10.4587173
1: -21.4280815, -12.2362041, -21.4315929, -12.2304201, -5.2833214, 5.2928905
2: -12.3934784, -5.7767491, -12.3977394, -5.7771974, -4.2592850, 4.2720432
3: -12.0099993, -4.1671033, -12.0137501, -4.1627932, -5.3679733, 5.3623238
4: -10.2848072, 0.0057602, -10.2963963, 0.0174915, -6.0691032, 6.0402184
5: -13.5550680, -4.0440159, -13.5630693, -4.0400076, -6.1437378, 6.1535339
6: -8.3189163, 0.5391184, -8.3402948, 0.5441295, -6.4614067, 6.4848518
7: -32.1562042, -22.0549984, -32.1574669, -22.0487862, -5.8383675, 5.8352070
8: -18.8055687, -9.1057968, -18.8167725, -9.0989504, -5.2403908, 5.2138977
9: -5.3312035, 1.3938938, -5.3460560, 1.3980365, -4.0552406, 4.0602150
10: -36.1352272, -27.7651405, -36.1380348, -27.7518921, -5.2662354, 5.2612457
11: -55.1329460, -44.8024673, -55.1336250, -44.7711296, -4.9511299, 4.9448109
12: -11.5791264, -4.5903344, -11.5814171, -4.5815673, -6.2337914, 6.2426834
13: 0.8898966, 8.0124722, 0.8783626, 8.0188370, -5.3088188, 5.3048019
14: -71.0835266, -57.9607925, -71.0858307, -57.9434586, -8.2609596, 8.2584267
15: -8.9105015, 0.9022608, -8.9206371, 0.9165106, -4.8995152, 4.8709087
16: -33.5572243, -23.9751740, -33.5694847, -23.9667473, -6.4484863, 6.4815636
17: -88.6790695, -72.4193726, -88.6786652, -72.3839264, -8.2221451, 8.1803207
18: -4.1774573, 1.0591235, -4.1791940, 1.0743077, -3.4026814, 3.4022655
19: -30.5247116, -23.2033157, -30.5267982, -23.1999836, -4.6459084, 4.6518173
20: -11.1715584, -5.1563902, -11.1733694, -5.1514263, -4.9410286, 4.9301472
21: -43.5466843, -35.0572815, -43.5476151, -35.0485153, -4.2556858, 4.2674351
22: -27.0054169, -19.5389404, -27.0073509, -19.5268421, -4.3466911, 4.3382206
23: -20.8515434, -12.5126591, -20.8582039, -12.5022593, -4.7757149, 4.7908592
24: -16.8592014, -7.6444712, -16.8652229, -7.6335454, -7.1652679, 7.1715431
25: -14.6370773, -6.9578748, -14.6422768, -6.9529309, -4.1938744, 4.2024460
26: -14.6186218, -7.8180952, -14.6211748, -7.7976570, -6.5689621, 6.5380821
27: -14.6305943, -9.5347366, -14.6332512, -9.5233650, -4.0638218, 4.0621643
28: -10.0223894, -1.4300778, -10.0249214, -1.4234498, -6.1692543, 6.1520309
29: -45.5838280, -36.8255005, -45.5850067, -36.8074570, -5.0149555, 5.0079384
30: -32.1861801, -23.0194359, -32.1877632, -22.9980068, -4.9954758, 5.0031891
31: -32.2351913, -23.5165253, -32.2442703, -23.5102692, -6.3044472, 6.3136787
32: 7.7180147, 13.6737537, 7.7000237, 13.6752958, -4.1534004, 4.1753998
33: 4.6091814, 16.3114891, 4.5859632, 16.3122406, -6.6767273, 6.7131805
34: 20.5349922, 30.9846210, 20.5249901, 30.9897232, -5.7196007, 5.7405396
35: 16.5045109, 26.8602104, 16.4880714, 26.8646984, -5.4181995, 5.4481850
36: 28.8099823, 35.1245728, 28.7955914, 35.1251984, -3.4240055, 3.4415207
37: 11.0248947, 20.1137085, 11.0043030, 20.1166420, -5.9491043, 5.9789314
38: 34.8626251, 43.6879921, 34.8410873, 43.6909866, -6.0388107, 6.0480614
39: 8.9973164, 18.5068855, 8.9642258, 18.5084095, -6.5165100, 6.5492783
40: 15.7922144, 25.1255875, 15.7702913, 25.1273632, -5.7979069, 5.8201561
41: 6.7327147, 13.2231512, 6.7151079, 13.2263947, -5.0130806, 5.0340424
42: -12.3869829, -3.4549737, -12.3993301, -3.4542999, -7.0472870, 7.0602074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5968402, upper bound: 3.6255090
time: 5.56 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5968441, upper bound: 3.6385287
time: 5.17 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -21.5720291, -8.4740181, -21.5775051, -8.4763956, -10.4410095, 10.4525146
1: -21.4298286, -12.2339182, -21.4283600, -12.2341862, -5.2880783, 5.2926064
2: -12.3964720, -5.7738681, -12.3923941, -5.7791948, -4.2727737, 4.2715282
3: -12.0139894, -4.1582928, -12.0013580, -4.1721306, -5.3749809, 5.3704147
4: -10.2955027, 0.0266817, -10.2878389, 0.0143209, -6.0820312, 6.0597534
5: -13.5615425, -4.0366812, -13.5609722, -4.0432091, -6.1507912, 6.1498947
6: -8.3332882, 0.5444336, -8.3348980, 0.5349832, -6.4734344, 6.4912682
7: -32.1571846, -22.0564194, -32.1511536, -22.0587234, -5.8371582, 5.8362923
8: -18.8171787, -9.0700378, -18.8093605, -9.1044712, -5.2518024, 5.2485561
9: -5.3357229, 1.4035068, -5.3384237, 1.3956654, -4.0551071, 4.0585938
10: -36.1407242, -27.7586784, -36.1291428, -27.7656364, -5.2719097, 5.2666798
11: -55.1681633, -44.7889214, -55.1142502, -44.8060684, -4.9892006, 4.9770908
12: -11.5811901, -4.5828362, -11.5742245, -4.5906606, -6.2319908, 6.2472153
13: 0.8838594, 8.0318003, 0.8878496, 8.0143156, -5.3153496, 5.3211670
14: -71.0876236, -57.9539261, -71.0831299, -57.9461403, -8.2695999, 8.2590218
15: -8.9234371, 0.9238849, -8.9139261, 0.9141350, -4.9187756, 4.8970013
16: -33.5806503, -23.9667683, -33.5495262, -23.9890022, -6.4695244, 6.4899559
17: -88.6793518, -72.4163208, -88.6676559, -72.4038086, -8.2196999, 8.1880798
18: -4.2011294, 1.0640297, -4.1762767, 1.0693786, -3.4228439, 3.4050579
19: -30.5337410, -23.2003765, -30.5130692, -23.2186508, -4.6577873, 4.6614933
20: -11.1736660, -5.1551967, -11.1672239, -5.1618662, -4.9394188, 4.9361477
21: -43.5635071, -35.0504990, -43.5303574, -35.0750427, -4.2808208, 4.2907829
22: -27.0088787, -19.5383148, -27.0005970, -19.5370960, -4.3483124, 4.3397751
23: -20.8710232, -12.5036392, -20.8512573, -12.5159664, -4.7943897, 4.8052559
24: -16.8790646, -7.6370306, -16.8614826, -7.6431389, -7.1807938, 7.1791000
25: -14.6411381, -6.9552145, -14.6330385, -6.9692950, -4.1980152, 4.2120628
26: -14.6230116, -7.8146901, -14.6190653, -7.8016763, -6.5665588, 6.5555725
27: -14.6419697, -9.5294743, -14.6233902, -9.5387077, -4.0744553, 4.0713596
28: -10.0371571, -1.4215727, -10.0220070, -1.4260745, -6.1658287, 6.1565132
29: -45.5967979, -36.8212509, -45.5721741, -36.8304520, -5.0268288, 5.0208473
30: -32.2184792, -23.0037308, -32.1768036, -23.0190010, -5.0256462, 5.0281429
31: -32.2545357, -23.5121422, -32.2290649, -23.5318871, -6.3290253, 6.3261223
32: 7.7130709, 13.6759701, 7.7035866, 13.6748142, -4.1583424, 4.1741905
33: 4.6064405, 16.3128052, 4.5984840, 16.3090057, -6.6826286, 6.7027473
34: 20.5227776, 30.9896107, 20.5523796, 30.9713306, -5.7405510, 5.7453842
35: 16.4906464, 26.8653088, 16.5104942, 26.8522854, -5.4418964, 5.4522133
36: 28.8078880, 35.1262283, 28.8092461, 35.1166611, -3.4286728, 3.4427099
37: 11.0153990, 20.1172390, 11.0107803, 20.1128883, -5.9579544, 5.9771385
38: 34.8518600, 43.7019119, 34.8619080, 43.6757622, -6.0494995, 6.0572319
39: 8.9880257, 18.5186310, 8.9733906, 18.5075512, -6.5228767, 6.5479584
40: 15.7813292, 25.1304512, 15.7813187, 25.1255894, -5.8083382, 5.8181744
41: 6.7238245, 13.2274942, 6.7181048, 13.2236233, -5.0205154, 5.0363731
42: -12.3869896, -3.4522684, -12.3908901, -3.4671774, -7.0457306, 7.0683556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5861083, upper bound: 3.6255611
time: 8.08 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6088644, upper bound: 3.6385803
time: 6.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5720291, -8.4740181, -21.5820084, -8.4756832, -10.4411850, 10.4562759
1: -21.4298286, -12.2339182, -21.4316216, -12.2307034, -5.2895775, 5.2940178
2: -12.3964720, -5.7738681, -12.3978252, -5.7772017, -4.2686138, 4.2708378
3: -12.0139894, -4.1582928, -12.0138836, -4.1627636, -5.3719330, 5.3711777
4: -10.2955027, 0.0266817, -10.2967758, 0.0174781, -6.0793610, 6.0615711
5: -13.5615425, -4.0366812, -13.5632210, -4.0400100, -6.1533661, 6.1517258
6: -8.3332882, 0.5444336, -8.3402958, 0.5443430, -6.4761009, 6.4903603
7: -32.1571846, -22.0564194, -32.1574554, -22.0505447, -5.8413925, 5.8397484
8: -18.8171787, -9.0700378, -18.8172703, -9.0989342, -5.2517223, 5.2497921
9: -5.3357229, 1.4035068, -5.3462067, 1.3980637, -4.0595608, 4.0698719
10: -36.1407242, -27.7586784, -36.1381531, -27.7518501, -5.2719631, 5.2632256
11: -55.1681633, -44.7889214, -55.1336060, -44.7705307, -4.9870720, 4.9580555
12: -11.5811901, -4.5828362, -11.5814075, -4.5814009, -6.2347183, 6.2495461
13: 0.8838594, 8.0318003, 0.8780900, 8.0189161, -5.3149834, 5.3246498
14: -71.0876236, -57.9539261, -71.0858765, -57.9434128, -8.2745514, 8.2636528
15: -8.9234371, 0.9238849, -8.9210901, 0.9165287, -4.9125595, 4.8928871
16: -33.5806503, -23.9667683, -33.5694962, -23.9663963, -6.4721832, 6.4893494
17: -88.6793518, -72.4163208, -88.6786270, -72.3852158, -8.2222900, 8.1854362
18: -4.2011294, 1.0640297, -4.1792235, 1.0745084, -3.4266033, 3.4070702
19: -30.5337410, -23.2003765, -30.5268669, -23.1998978, -4.6554413, 4.6543560
20: -11.1736660, -5.1551967, -11.1733875, -5.1521931, -4.9407349, 4.9344368
21: -43.5635071, -35.0504990, -43.5476608, -35.0482140, -4.2727852, 4.2741528
22: -27.0088787, -19.5383148, -27.0071735, -19.5268364, -4.3505974, 4.3382225
23: -20.8710232, -12.5036392, -20.8582535, -12.5019550, -4.7955437, 4.8004379
24: -16.8790646, -7.6370306, -16.8652916, -7.6332178, -7.1854286, 7.1789589
25: -14.6411381, -6.9552145, -14.6423340, -6.9528637, -4.1984291, 4.2052574
26: -14.6230116, -7.8146901, -14.6212635, -7.7979898, -6.5691338, 6.5564919
27: -14.6419697, -9.5294743, -14.6333294, -9.5231705, -4.0748138, 4.0675087
28: -10.0371571, -1.4215727, -10.0249710, -1.4231704, -6.1693802, 6.1595840
29: -45.5967979, -36.8212509, -45.5850449, -36.8072548, -5.0279942, 5.0112801
30: -32.2184792, -23.0037308, -32.1877861, -22.9974174, -5.0283432, 5.0191536
31: -32.2545357, -23.5121422, -32.2443542, -23.5100632, -6.3251572, 6.3175583
32: 7.7130709, 13.6759701, 7.6999989, 13.6753855, -4.1585617, 4.1775341
33: 4.6064405, 16.3128052, 4.5866923, 16.3122807, -6.6823807, 6.7144241
34: 20.5227776, 30.9896107, 20.5249634, 30.9899139, -5.7320194, 5.7450638
35: 16.4906464, 26.8653088, 16.4880219, 26.8649235, -5.4322014, 5.4532166
36: 28.8078880, 35.1262283, 28.7955322, 35.1252289, -3.4260311, 3.4430103
37: 11.0153990, 20.1172390, 11.0042782, 20.1167660, -5.9590378, 5.9821548
38: 34.8518600, 43.7019119, 34.8407173, 43.6909943, -6.0498505, 6.0622711
39: 8.9880257, 18.5186310, 8.9640465, 18.5083904, -6.5256157, 6.5587120
40: 15.7813292, 25.1304512, 15.7702188, 25.1273880, -5.8079262, 5.8282242
41: 6.7238245, 13.2274942, 6.7150764, 13.2265558, -5.0220604, 5.0383339
42: -12.3869896, -3.4522684, -12.3990746, -3.4542990, -7.0474510, 7.0660553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=78, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 750

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6088602, upper bound: 3.6255614
time: 9.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6088644, upper bound: 3.6385806
time: 5.72 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 17.18 seconds
IS_A2_B1_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5856645, upper bound: 3.6255129
IS_A2_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6385325
IS_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5976835, upper bound: 3.6255653
IS_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5976869, upper bound: 3.6385844
IS_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5856645, upper bound: 3.6255132
IS_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6385328
IS_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5976835, upper bound: 3.6255656
IS_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5976869, upper bound: 3.6385848
IS_A2_B1_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5968402, upper bound: 3.6255129
IS_A2_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5968442, upper bound: 3.6385325
IS_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.6088602, upper bound: 3.6255653
IS_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.6088644, upper bound: 3.6385844
IS_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5968402, upper bound: 3.6255132
IS_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5968442, upper bound: 3.6385328
IS_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.6088602, upper bound: 3.6255656
IS_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.6088644, upper bound: 3.6385848
IS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5861083, upper bound: 3.6255087
IS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5861122, upper bound: 3.6385283
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5861083, upper bound: 3.6255088
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5861122, upper bound: 3.6385287
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5981240, upper bound: 3.6255611
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5981283, upper bound: 3.6385803
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5981240, upper bound: 3.6255614
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5981283, upper bound: 3.6385806
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5968402, upper bound: 3.6255087
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5968441, upper bound: 3.6385283
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5968402, upper bound: 3.6255090
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5968441, upper bound: 3.6385287
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.5861083, upper bound: 3.6255611
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.6088644, upper bound: 3.6385803
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.6088602, upper bound: 3.6255614
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 17.18
Output dim: 38, lower bound: -3.6088644, upper bound: 3.6385806

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -21.5749893, -8.4789457, -21.5430279, -8.4853354, -10.4067230, 10.3786926
1: -21.4300957, -12.2320223, -21.4362774, -12.2506638, -5.2523880, 5.2809467
2: -12.3953953, -5.7781649, -12.3964186, -5.7866645, -4.2517967, 4.2680359
3: -12.0102329, -4.1681590, -11.9993477, -4.1954160, -5.3495598, 5.3649178
4: -10.2868633, 0.0085912, -10.2790489, -0.0386317, -6.0175362, 6.0663052
5: -13.5582209, -4.0437937, -13.5603657, -4.0680585, -6.1302872, 6.1544724
6: -8.3327951, 0.5389727, -8.2909718, 0.5339514, -6.4625931, 6.4194031
7: -32.1567421, -22.0512161, -32.1557007, -22.0942917, -5.8082314, 5.8546486
8: -18.8054333, -9.1144857, -18.7980213, -9.1586075, -5.1827164, 5.2230225
9: -5.3418612, 1.3933084, -5.3177032, 1.3793271, -4.0483456, 4.0336437
10: -36.1339951, -27.7550507, -36.1322403, -27.7889137, -5.2373657, 5.2639179
11: -55.1220551, -44.7847214, -55.0806580, -44.8624115, -4.9176083, 4.9657955
12: -11.5787487, -4.5864897, -11.5635519, -4.6021738, -6.2185326, 6.2065430
13: 0.8842968, 8.0087528, 0.9012449, 7.9730825, -5.2674599, 5.2896843
14: -71.0838623, -57.9487457, -71.0982208, -57.9765549, -8.2175713, 8.2691956
15: -8.9100342, 0.9086533, -8.9085970, 0.8695927, -4.8552189, 4.9050884
16: -33.5593491, -23.9756165, -33.5182495, -24.0127640, -6.4484253, 6.4260788
17: -88.6775818, -72.3888931, -88.6768723, -72.4889145, -8.1484451, 8.2546196
18: -4.1687598, 1.0691636, -4.1419172, 1.0403128, -3.3772831, 3.3793545
19: -30.5213547, -23.2034645, -30.4963245, -23.2298317, -4.6397209, 4.6425781
20: -11.1718168, -5.1542578, -11.1623192, -5.1650171, -4.9218063, 4.9199333
21: -43.5404816, -35.0553780, -43.5099335, -35.0940208, -4.2527809, 4.2558899
22: -27.0031052, -19.5271721, -26.9938698, -19.5682907, -4.3195076, 4.3557911
23: -20.8488655, -12.5103188, -20.8033524, -12.5395308, -4.7730103, 4.7530212
24: -16.8537178, -7.6413794, -16.8164864, -7.6661024, -7.1454086, 7.1350136
25: -14.6369839, -6.9551191, -14.6052504, -6.9829755, -4.1908360, 4.1861973
26: -14.6172180, -7.8007402, -14.6099472, -7.8242364, -6.5267563, 6.5325508
27: -14.6269512, -9.5283508, -14.6046486, -9.5699348, -4.0415001, 4.0600929
28: -10.0197020, -1.4311187, -10.0020590, -1.4318068, -6.1416588, 6.1258202
29: -45.5773315, -36.8120880, -45.5550117, -36.8733139, -4.9799728, 5.0214462
30: -32.1763878, -23.0117645, -32.1431503, -23.0555210, -4.9651680, 4.9892406
31: -32.2326698, -23.5151691, -32.1863785, -23.5533218, -6.2949715, 6.2830238
32: 7.7025013, 13.6732769, 7.7300453, 13.6792126, -4.1731987, 4.1346054
33: 4.5918884, 16.3114948, 4.6630297, 16.3125782, -6.7233582, 6.6379051
34: 20.5321045, 30.9848785, 20.6054783, 30.9686718, -5.7585449, 5.6955967
35: 16.4963074, 26.8593750, 16.5757256, 26.8462601, -5.4661007, 5.3927193
36: 28.7978344, 35.1238785, 28.8415184, 35.1190453, -3.4545069, 3.4144697
37: 11.0115433, 20.1137142, 11.0777664, 20.1084938, -5.9765663, 5.9078293
38: 34.8504829, 43.6854553, 34.9236908, 43.6582260, -6.0534973, 6.0050430
39: 8.9738245, 18.5062523, 9.0363989, 18.5012989, -6.5493546, 6.4870491
40: 15.7767057, 25.1237106, 15.8296080, 25.1313972, -5.8236122, 5.7539749
41: 6.7208385, 13.2224617, 6.7578225, 13.2231483, -5.0253716, 4.9834251
42: -12.3980970, -3.4556828, -12.3675518, -3.4586287, -7.0572701, 7.0266953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6307889
time: 4.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6381818
time: 5.05 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -21.5842972, -8.4761457, -21.5429306, -8.4853115, -10.4385986, 10.3762512
1: -21.4317970, -12.2297382, -21.4363251, -12.2509165, -5.2586460, 5.2821064
2: -12.3984051, -5.7753081, -12.3965073, -5.7866688, -4.2611313, 4.2668476
3: -12.0142279, -4.1593504, -11.9994850, -4.1953793, -5.3535271, 5.3737869
4: -10.2975435, 0.0294781, -10.2794580, -0.0386250, -6.0277824, 6.0876656
5: -13.5646896, -4.0364881, -13.5605431, -4.0680094, -6.1399193, 6.1526604
6: -8.3471527, 0.5442717, -8.2909651, 0.5341752, -6.4772530, 6.4248962
7: -32.1576920, -22.0526657, -32.1556892, -22.0960560, -5.8112717, 5.8591938
8: -18.8170834, -9.0787287, -18.7985134, -9.1586075, -5.1940689, 5.2589245
9: -5.3464036, 1.4029369, -5.3178458, 1.3793664, -4.0526829, 4.0432968
10: -36.1394730, -27.7486115, -36.1323471, -27.7888641, -5.2431297, 5.2659016
11: -55.1572800, -44.7712479, -55.0806541, -44.8618546, -4.9535561, 4.9790401
12: -11.5808210, -4.5789709, -11.5635338, -4.6019845, -6.2194328, 6.2134094
13: 0.8782769, 8.0280972, 0.9009946, 7.9731522, -5.2735977, 5.3095322
14: -71.0879593, -57.9418869, -71.0982666, -57.9765396, -8.2311554, 8.2744446
15: -8.9229488, 0.9302769, -8.9090004, 0.8696079, -4.8682079, 4.9270611
16: -33.5827560, -23.9672184, -33.5182800, -24.0123863, -6.4721298, 6.4338417
17: -88.6778717, -72.3859253, -88.6768188, -72.4902115, -8.1486015, 8.2597198
18: -4.1923909, 1.0740628, -4.1419601, 1.0405185, -3.4010983, 3.3841419
19: -30.5303574, -23.2005081, -30.4963627, -23.2297516, -4.6492672, 4.6451130
20: -11.1739063, -5.1530633, -11.1623278, -5.1657634, -4.9214897, 4.9242134
21: -43.5572891, -35.0485840, -43.5099831, -35.0937386, -4.2698765, 4.2626171
22: -27.0065746, -19.5265503, -26.9936943, -19.5683079, -4.3234386, 4.3557873
23: -20.8683910, -12.5013008, -20.8033867, -12.5391951, -4.7928429, 4.7626019
24: -16.8736115, -7.6339154, -16.8165779, -7.6657887, -7.1655579, 7.1424408
25: -14.6410141, -6.9524918, -14.6053038, -6.9829164, -4.1953926, 4.1890030
26: -14.6216125, -7.7973356, -14.6100445, -7.8245831, -6.5269279, 6.5509415
27: -14.6383152, -9.5230532, -14.6047344, -9.5697327, -4.0524845, 4.0654411
28: -10.0344667, -1.4226111, -10.0021057, -1.4315159, -6.1417885, 6.1333885
29: -45.5902901, -36.8078537, -45.5550232, -36.8731003, -4.9930077, 5.0247650
30: -32.2086906, -22.9960785, -32.1431313, -23.0549984, -4.9980354, 5.0052280
31: -32.2519836, -23.5107880, -32.1864433, -23.5531197, -6.3156853, 6.2868843
32: 7.6975498, 13.6754856, 7.7300396, 13.6792774, -4.1783524, 4.1367378
33: 4.5891733, 16.3127556, 4.6637659, 16.3126011, -6.7290554, 6.6391602
34: 20.5198498, 30.9898300, 20.6054230, 30.9688454, -5.7709618, 5.7001324
35: 16.4824409, 26.8645172, 16.5756531, 26.8464737, -5.4801006, 5.3977509
36: 28.7957458, 35.1255150, 28.8414612, 35.1190758, -3.4565420, 3.4159594
37: 11.0020714, 20.1172600, 11.0777245, 20.1086044, -5.9864807, 5.9110298
38: 34.8397217, 43.6993561, 34.9233475, 43.6582069, -6.0645180, 6.0192604
39: 8.9645119, 18.5180397, 9.0362358, 18.5013008, -6.5584831, 6.4964600
40: 15.7658262, 25.1285839, 15.8295317, 25.1314240, -5.8336582, 5.7620449
41: 6.7119360, 13.2268076, 6.7578015, 13.2233019, -5.0343513, 4.9877319
42: -12.3980989, -3.4529331, -12.3673086, -3.4586010, -7.0574532, 7.0325165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5973363, upper bound: 3.6308410
time: 6.50 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5973363, upper bound: 3.6382336
time: 5.60 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.5749893, -8.4789457, -21.5475121, -8.4845819, -10.4068832, 10.3824310
1: -21.4300957, -12.2320223, -21.4395790, -12.2471952, -5.2539024, 5.2823753
2: -12.3953953, -5.7781649, -12.4018269, -5.7847013, -4.2476349, 4.2673550
3: -12.0102329, -4.1681590, -12.0118961, -4.1860056, -5.3465157, 5.3657112
4: -10.2868633, 0.0085912, -10.2879848, -0.0355055, -6.0148544, 6.0681019
5: -13.5582209, -4.0437937, -13.5626469, -4.0648460, -6.1328621, 6.1562996
6: -8.3327951, 0.5389727, -8.2963915, 0.5433391, -6.4652519, 6.4184837
7: -32.1567421, -22.0512161, -32.1619644, -22.0860863, -5.8124619, 5.8581104
8: -18.8054333, -9.1144857, -18.8059082, -9.1530790, -5.1826439, 5.2243061
9: -5.3418612, 1.3933084, -5.3255539, 1.3817370, -4.0528545, 4.0448341
10: -36.1339951, -27.7550507, -36.1412048, -27.7751522, -5.2374001, 5.2604389
11: -55.1220551, -44.7847214, -55.1000328, -44.8269119, -4.9154472, 4.9467411
12: -11.5787487, -4.5864897, -11.5707779, -4.5928993, -6.2212563, 6.2089081
13: 0.8842968, 8.0087528, 0.8915225, 7.9776697, -5.2670822, 5.2931786
14: -71.0838623, -57.9487457, -71.1009827, -57.9738426, -8.2225342, 8.2738152
15: -8.9100342, 0.9086533, -8.9157419, 0.8720331, -4.8489723, 4.9009533
16: -33.5593491, -23.9756165, -33.5382729, -23.9901161, -6.4512482, 6.4254341
17: -88.6775818, -72.3888931, -88.6878510, -72.4703140, -8.1510468, 8.2519302
18: -4.1687598, 1.0691636, -4.1448421, 1.0454016, -3.3810062, 3.3813152
19: -30.5213547, -23.2034645, -30.5100822, -23.2110901, -4.6373901, 4.6354504
20: -11.1718168, -5.1542578, -11.1684742, -5.1553507, -4.9231415, 4.9182110
21: -43.5404816, -35.0553780, -43.5272369, -35.0672150, -4.2447414, 4.2392445
22: -27.0031052, -19.5271721, -27.0004349, -19.5580559, -4.3217926, 4.3542309
23: -20.8488655, -12.5103188, -20.8103256, -12.5254984, -4.7741756, 4.7481899
24: -16.8537178, -7.6413794, -16.8203163, -7.6561904, -7.1500435, 7.1348419
25: -14.6369839, -6.9551191, -14.6145649, -6.9665833, -4.1912556, 4.1793880
26: -14.6172180, -7.8007402, -14.6121397, -7.8205767, -6.5293465, 6.5334625
27: -14.6269512, -9.5283508, -14.6146078, -9.5543842, -4.0418224, 4.0562248
28: -10.0197020, -1.4311187, -10.0050163, -1.4289172, -6.1452217, 6.1289024
29: -45.5773315, -36.8120880, -45.5678558, -36.8501282, -4.9811211, 5.0118637
30: -32.1763878, -23.0117645, -32.1541214, -23.0339508, -4.9678555, 4.9802074
31: -32.2326698, -23.5151691, -32.2015915, -23.5314922, -6.2911530, 6.2744255
32: 7.7025013, 13.6732769, 7.7264605, 13.6797523, -4.1734123, 4.1379585
33: 4.5918884, 16.3114948, 4.6513271, 16.3158150, -6.7230110, 6.6496124
34: 20.5321045, 30.9848785, 20.5780258, 30.9872704, -5.7500076, 5.6952953
35: 16.4963074, 26.8593750, 16.5532875, 26.8588963, -5.4563446, 5.3937359
36: 28.7978344, 35.1238785, 28.8278008, 35.1276169, -3.4518547, 3.4147739
37: 11.0115433, 20.1137142, 11.0712366, 20.1123695, -5.9776878, 5.9128304
38: 34.8504829, 43.6854553, 34.9025154, 43.6734924, -6.0537949, 6.0101547
39: 8.9738245, 18.5062523, 9.0271358, 18.5021858, -6.5520897, 6.4977760
40: 15.7767057, 25.1237106, 15.8185282, 25.1331787, -5.8231773, 5.7639694
41: 6.7208385, 13.2224617, 6.7547994, 13.2260847, -5.0269279, 4.9853859
42: -12.3980970, -3.4556828, -12.3757458, -3.4457474, -7.0590057, 7.0243835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6307891
time: 6.18 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6381821
time: 4.78 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5842972, -8.4761457, -21.5473824, -8.4845114, -10.4387436, 10.3800011
1: -21.4317970, -12.2297382, -21.4396210, -12.2474394, -5.2601681, 5.2835369
2: -12.3984051, -5.7753081, -12.4019032, -5.7847047, -4.2569752, 4.2661572
3: -12.0142279, -4.1593504, -12.0120153, -4.1859913, -5.3504677, 5.3745728
4: -10.2975435, 0.0294781, -10.2884007, -0.0355235, -6.0251160, 6.0894680
5: -13.5646896, -4.0364881, -13.5628042, -4.0648575, -6.1424828, 6.1544800
6: -8.3471527, 0.5442717, -8.2964125, 0.5435485, -6.4799194, 6.4239807
7: -32.1576920, -22.0526657, -32.1619530, -22.0878448, -5.8154945, 5.8626537
8: -18.8170834, -9.0787287, -18.8064289, -9.1530771, -5.1939888, 5.2601795
9: -5.3464036, 1.4029369, -5.3257170, 1.3817542, -4.0571823, 4.0544968
10: -36.1394730, -27.7486115, -36.1413498, -27.7751007, -5.2431545, 5.2624264
11: -55.1572800, -44.7712479, -55.1000328, -44.8263016, -4.9513874, 4.9599915
12: -11.5808210, -4.5789709, -11.5707655, -4.5927272, -6.2221527, 6.2157784
13: 0.8782769, 8.0280972, 0.8912590, 7.9777417, -5.2732201, 5.3130150
14: -71.0879593, -57.9418869, -71.1010284, -57.9738045, -8.2361183, 8.2790794
15: -8.9229488, 0.9302769, -8.9161482, 0.8720331, -4.8619976, 4.9229355
16: -33.5827560, -23.9672184, -33.5382996, -23.9897385, -6.4749451, 6.4332047
17: -88.6778717, -72.3859253, -88.6878052, -72.4715424, -8.1511955, 8.2570457
18: -4.1923909, 1.0740628, -4.1448789, 1.0456171, -3.4048214, 3.3861046
19: -30.5303574, -23.2005081, -30.5101719, -23.2109985, -4.6469231, 4.6379814
20: -11.1739063, -5.1530633, -11.1684771, -5.1560946, -4.9228325, 4.9224758
21: -43.5572891, -35.0485840, -43.5272598, -35.0669365, -4.2618408, 4.2459698
22: -27.0065746, -19.5265503, -27.0002670, -19.5580540, -4.3257198, 4.3542328
23: -20.8683910, -12.5013008, -20.8103638, -12.5251579, -4.7940140, 4.7577705
24: -16.8736115, -7.6339154, -16.8204021, -7.6558728, -7.1701775, 7.1422691
25: -14.6410141, -6.9524918, -14.6146183, -6.9664965, -4.1958084, 4.1822014
26: -14.6216125, -7.7973356, -14.6122217, -7.8209119, -6.5295181, 6.5518608
27: -14.6383152, -9.5230532, -14.6146908, -9.5541964, -4.0528069, 4.0615711
28: -10.0344667, -1.4226111, -10.0050621, -1.4286047, -6.1453438, 6.1364746
29: -45.5902901, -36.8078537, -45.5678940, -36.8499451, -4.9941444, 5.0151920
30: -32.2086906, -22.9960785, -32.1541557, -23.0334110, -5.0007324, 4.9961948
31: -32.2519836, -23.5107880, -32.2016754, -23.5312920, -6.3118210, 6.2782936
32: 7.6975498, 13.6754856, 7.7264442, 13.6798410, -4.1785736, 4.1400948
33: 4.5891733, 16.3127556, 4.6520658, 16.3158379, -6.7287121, 6.6508636
34: 20.5198498, 30.9898300, 20.5780334, 30.9874668, -5.7624245, 5.6998329
35: 16.4824409, 26.8645172, 16.5532246, 26.8591156, -5.4703426, 5.3987541
36: 28.7957458, 35.1255150, 28.8277473, 35.1276665, -3.4538860, 3.4162607
37: 11.0020714, 20.1172600, 11.0711641, 20.1124878, -5.9875984, 5.9160461
38: 34.8397217, 43.6993561, 34.9021416, 43.6735077, -6.0648537, 6.0243607
39: 8.9645119, 18.5180397, 9.0269728, 18.5021534, -6.5612068, 6.5071945
40: 15.7658262, 25.1285839, 15.8184319, 25.1332111, -5.8332176, 5.7720432
41: 6.7119360, 13.2268076, 6.7547565, 13.2262526, -5.0359116, 4.9896889
42: -12.3980989, -3.4529331, -12.3754911, -3.4457293, -7.0591965, 7.0302086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5973363, upper bound: 3.6308413
time: 8.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5973363, upper bound: 3.6382340
time: 5.48 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -21.5798874, -8.4788465, -21.5597916, -8.4801655, -10.4588852, 10.3982925
1: -21.4305859, -12.2309866, -21.4397240, -12.2478399, -5.2737675, 5.2838497
2: -12.3957214, -5.7774043, -12.3996239, -5.7845941, -4.2719078, 4.2682304
3: -12.0103674, -4.1640348, -12.0057869, -4.1824436, -5.3562393, 5.3758774
4: -10.2875586, 0.0167704, -10.2942619, -0.0156139, -6.0189285, 6.0922337
5: -13.5587845, -4.0412655, -13.5672293, -4.0600019, -6.1412926, 6.1557388
6: -8.3391027, 0.5390772, -8.3101082, 0.5429580, -6.4786072, 6.4343491
7: -32.1569328, -22.0496349, -32.1573181, -22.0896645, -5.8186913, 5.8547363
8: -18.8056965, -9.0998297, -18.8167706, -9.1170368, -5.1959534, 5.2578068
9: -5.3423018, 1.3970394, -5.3257742, 1.3904247, -4.0506248, 4.0462894
10: -36.1350822, -27.7535973, -36.1386337, -27.7838020, -5.2443638, 5.2689075
11: -55.1328278, -44.7844772, -55.1101227, -44.8452148, -4.9466648, 4.9692783
12: -11.5810375, -4.5858097, -11.5703716, -4.5941415, -6.2283096, 6.2072525
13: 0.8842207, 8.0166311, 0.8907263, 7.9970803, -5.2845230, 5.3082199
14: -71.0843887, -57.9473114, -71.1015854, -57.9732132, -8.2437706, 8.2727432
15: -8.9113722, 0.9157834, -8.9239321, 0.8899441, -4.8594360, 4.9299450
16: -33.5681725, -23.9756012, -33.5449524, -23.9992523, -6.4697571, 6.4416885
17: -88.6782608, -72.3878021, -88.6786804, -72.4798584, -8.1568527, 8.2581596
18: -4.1781888, 1.0691724, -4.1682329, 1.0486946, -3.3949986, 3.3925133
19: -30.5251789, -23.2034645, -30.5081024, -23.2245827, -4.6486626, 4.6454830
20: -11.1720467, -5.1537352, -11.1633759, -5.1617651, -4.9234848, 4.9346619
21: -43.5464516, -35.0550995, -43.5273590, -35.0838242, -4.2703896, 4.2581921
22: -27.0053825, -19.5271358, -27.0010319, -19.5653801, -4.3255596, 4.3578930
23: -20.8568859, -12.5094566, -20.8263779, -12.5247307, -4.7972069, 4.7609692
24: -16.8632393, -7.6413078, -16.8447666, -7.6508155, -7.1702614, 7.1527023
25: -14.6406116, -6.9546862, -14.6165562, -6.9755793, -4.2030220, 4.1909180
26: -14.6176710, -7.8000569, -14.6120491, -7.8191733, -6.5304298, 6.5664978
27: -14.6309547, -9.5279789, -14.6179600, -9.5618801, -4.0535469, 4.0678768
28: -10.0232182, -1.4302486, -10.0125380, -1.4212809, -6.1419182, 6.1438637
29: -45.5841522, -36.8120422, -45.5752563, -36.8626251, -4.9970646, 5.0314312
30: -32.1870384, -23.0102749, -32.1729736, -23.0353622, -4.9995117, 4.9943066
31: -32.2416153, -23.5151443, -32.2137413, -23.5430832, -6.3140793, 6.3051758
32: 7.7007260, 13.6733627, 7.7240925, 13.6820860, -4.1812973, 4.1405392
33: 4.5878453, 16.3115501, 4.6498909, 16.3181648, -6.7314701, 6.6436234
34: 20.5264626, 30.9849567, 20.5885220, 30.9773979, -5.7729530, 5.7004929
35: 16.4900265, 26.8594112, 16.5564537, 26.8561287, -5.4822598, 5.3980732
36: 28.7970581, 35.1240997, 28.8381157, 35.1215668, -3.4569492, 3.4166555
37: 11.0062122, 20.1138172, 11.0604420, 20.1154709, -5.9895744, 5.9179497
38: 34.8494949, 43.6903458, 34.9104767, 43.6726837, -6.0580368, 6.0241356
39: 8.9732418, 18.5080528, 9.0265255, 18.5063515, -6.5510216, 6.4965019
40: 15.7733917, 25.1265602, 15.8149452, 25.1399746, -5.8339710, 5.7731457
41: 6.7163315, 13.2225733, 6.7434626, 13.2307577, -5.0375786, 4.9933090
42: -12.3984766, -3.4551392, -12.3685722, -3.4556129, -7.0597534, 7.0285225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1769

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6273534
time: 4.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6385325
time: 5.12 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -21.5891876, -8.4760523, -21.5596504, -8.4801149, -10.4907684, 10.3958359
1: -21.4323273, -12.2287083, -21.4397736, -12.2481213, -5.2800198, 5.2850094
2: -12.3987360, -5.7745323, -12.3997192, -5.7845893, -4.2812424, 4.2670269
3: -12.0143433, -4.1552062, -12.0059147, -4.1824045, -5.3601875, 5.3847427
4: -10.2982388, 0.0377033, -10.2946529, -0.0156294, -6.0291862, 6.1135788
5: -13.5652857, -4.0339422, -13.5674000, -4.0599627, -6.1510010, 6.1539192
6: -8.3534698, 0.5443816, -8.3101215, 0.5431615, -6.4932709, 6.4398384
7: -32.1578751, -22.0510426, -32.1573181, -22.0913849, -5.8217049, 5.8592739
8: -18.8173046, -9.0640450, -18.8172569, -9.1170101, -5.2072926, 5.2937260
9: -5.3468370, 1.4066513, -5.3259315, 1.3904459, -4.0549583, 4.0559406
10: -36.1405869, -27.7471561, -36.1387558, -27.7837486, -5.2502079, 5.2708893
11: -55.1680222, -44.7709236, -55.1101189, -44.8446274, -4.9826050, 4.9825382
12: -11.5831261, -4.5782962, -11.5703459, -4.5939531, -6.2292252, 6.2140999
13: 0.8781723, 8.0359831, 0.8904927, 7.9971552, -5.2906494, 5.3280716
14: -71.0884857, -57.9404831, -71.1016388, -57.9731903, -8.2573547, 8.2779884
15: -8.9242821, 0.9373827, -8.9243507, 0.8899465, -4.8724213, 4.9519253
16: -33.5916061, -23.9671898, -33.5449715, -23.9988899, -6.4934578, 6.4494629
17: -88.6785431, -72.3847809, -88.6786423, -72.4811020, -8.1570015, 8.2632790
18: -4.2018642, 1.0740857, -4.1682625, 1.0489109, -3.4188652, 3.3973141
19: -30.5341949, -23.2005215, -30.5081367, -23.2244987, -4.6582413, 4.6480312
20: -11.1741533, -5.1525631, -11.1634026, -5.1625233, -4.9231911, 4.9389515
21: -43.5632553, -35.0483246, -43.5273666, -35.0835419, -4.2874908, 4.2649174
22: -27.0088272, -19.5264912, -27.0008507, -19.5653915, -4.3294907, 4.3579102
23: -20.8763924, -12.5004616, -20.8264370, -12.5244303, -4.8170433, 4.7705612
24: -16.8831329, -7.6338539, -16.8448277, -7.6505036, -7.1903877, 7.1601562
25: -14.6446362, -6.9520350, -14.6165924, -6.9755177, -4.2075787, 4.1937370
26: -14.6220579, -7.7966609, -14.6121531, -7.8195095, -6.5305748, 6.5848923
27: -14.6423149, -9.5226860, -14.6180458, -9.5616798, -4.0645351, 4.0732365
28: -10.0379944, -1.4217486, -10.0125895, -1.4209726, -6.1420517, 6.1514626
29: -45.5971222, -36.8078003, -45.5752716, -36.8624344, -5.0100937, 5.0347748
30: -32.2193489, -22.9945431, -32.1729889, -23.0347958, -5.0323944, 5.0102615
31: -32.2609825, -23.5107632, -32.2138252, -23.5428696, -6.3348312, 6.3090630
32: 7.6957679, 13.6755619, 7.7240915, 13.6821384, -4.1864586, 4.1426735
33: 4.5851336, 16.3128567, 4.6506004, 16.3181877, -6.7371788, 6.6448631
34: 20.5142479, 30.9899006, 20.5885162, 30.9775848, -5.7853775, 5.7050228
35: 16.4761906, 26.8645267, 16.5563622, 26.8563557, -5.4962635, 5.4030933
36: 28.7949600, 35.1257401, 28.8380699, 35.1216011, -3.4589758, 3.4181614
37: 10.9967194, 20.1173611, 11.0604076, 20.1155720, -5.9995041, 5.9211655
38: 34.8386993, 43.7042656, 34.9100914, 43.6726913, -6.0690536, 6.0383492
39: 8.9638977, 18.5198002, 9.0263414, 18.5063477, -6.5601501, 6.5059280
40: 15.7625017, 25.1314182, 15.8148499, 25.1400032, -5.8440094, 5.7812271
41: 6.7074528, 13.2269201, 6.7434478, 13.2309237, -5.0465393, 4.9975967
42: -12.3984756, -3.4524648, -12.3683290, -3.4556012, -7.0599365, 7.0343781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1769

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5976869, upper bound: 3.6274033
time: 5.00 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5976869, upper bound: 3.6385844
time: 5.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.5798874, -8.4788465, -21.5642891, -8.4794130, -10.4590530, 10.4020462
1: -21.4305859, -12.2309866, -21.4430428, -12.2443390, -5.2752762, 5.2852821
2: -12.3957214, -5.7774043, -12.4050379, -5.7826242, -4.2677536, 4.2675457
3: -12.0103674, -4.1640348, -12.0183420, -4.1730127, -5.3531837, 5.3766632
4: -10.2875586, 0.0167704, -10.3031998, -0.0124934, -6.0162582, 6.0940380
5: -13.5587845, -4.0412655, -13.5694895, -4.0567741, -6.1438713, 6.1575584
6: -8.3391027, 0.5390772, -8.3155251, 0.5523276, -6.4812622, 6.4334106
7: -32.1569328, -22.0496349, -32.1635742, -22.0814075, -5.8229103, 5.8581867
8: -18.8056965, -9.0998297, -18.8246765, -9.1115360, -5.1958790, 5.2590694
9: -5.3423018, 1.3970394, -5.3336234, 1.3928273, -4.0551224, 4.0574818
10: -36.1350822, -27.7535973, -36.1476402, -27.7700405, -5.2444077, 5.2654228
11: -55.1328278, -44.7844772, -55.1294785, -44.8096886, -4.9445019, 4.9502392
12: -11.5810375, -4.5858097, -11.5775881, -4.5848761, -6.2310410, 6.2096367
13: 0.8842207, 8.0166311, 0.8810055, 8.0016718, -5.2841339, 5.3117142
14: -71.0843887, -57.9473114, -71.1043396, -57.9705086, -8.2487373, 8.2773628
15: -8.9113722, 0.9157834, -8.9311104, 0.8923228, -4.8532028, 4.9258175
16: -33.5681725, -23.9756012, -33.5649643, -23.9766350, -6.4725685, 6.4410439
17: -88.6782608, -72.3878021, -88.6896591, -72.4612732, -8.1594391, 8.2554703
18: -4.1781888, 1.0691724, -4.1711783, 1.0537989, -3.3987255, 3.3944988
19: -30.5251789, -23.2034645, -30.5218792, -23.2058372, -4.6463318, 4.6383533
20: -11.1720467, -5.1537352, -11.1695089, -5.1520786, -4.9248238, 4.9329376
21: -43.5464516, -35.0550995, -43.5446701, -35.0570068, -4.2623653, 4.2415466
22: -27.0053825, -19.5271358, -27.0076141, -19.5551376, -4.3278313, 4.3563366
23: -20.8568859, -12.5094566, -20.8333740, -12.5107498, -4.7983742, 4.7561302
24: -16.8632393, -7.6413078, -16.8485680, -7.6409149, -7.1749001, 7.1525497
25: -14.6406116, -6.9546862, -14.6258450, -6.9591579, -4.2034302, 4.1841087
26: -14.6176710, -7.8000569, -14.6142292, -7.8155012, -6.5330162, 6.5674095
27: -14.6309547, -9.5279789, -14.6278954, -9.5463266, -4.0538654, 4.0640125
28: -10.0232182, -1.4302486, -10.0154963, -1.4183484, -6.1454811, 6.1469650
29: -45.5841522, -36.8120422, -45.5880966, -36.8394699, -4.9982052, 5.0218601
30: -32.1870384, -23.0102749, -32.1839371, -23.0137844, -5.0021935, 4.9852753
31: -32.2416153, -23.5151443, -32.2289772, -23.5212555, -6.3102493, 6.2966118
32: 7.7007260, 13.6733627, 7.7205048, 13.6826229, -4.1815224, 4.1438904
33: 4.5878453, 16.3115501, 4.6381955, 16.3214016, -6.7311287, 6.6553345
34: 20.5264626, 30.9849567, 20.5611172, 30.9959602, -5.7644196, 5.7001934
35: 16.4900265, 26.8594112, 16.5339947, 26.8687592, -5.4725113, 5.3990765
36: 28.7970581, 35.1240997, 28.8244057, 35.1301422, -3.4542990, 3.4169683
37: 11.0062122, 20.1138172, 11.0539303, 20.1193447, -5.9906998, 5.9229507
38: 34.8494949, 43.6903458, 34.8892670, 43.6879387, -6.0583611, 6.0292473
39: 8.9732418, 18.5080528, 9.0172281, 18.5072060, -6.5537491, 6.5072289
40: 15.7733917, 25.1265602, 15.8038521, 25.1417732, -5.8335190, 5.7831345
41: 6.7163315, 13.2225733, 6.7404404, 13.2336807, -5.0391235, 4.9952660
42: -12.3984766, -3.4551392, -12.3767891, -3.4427600, -7.0614777, 7.0262032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1769

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6273537
time: 15.37 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6385328
time: 5.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5891876, -8.4760523, -21.5641575, -8.4793348, -10.4909439, 10.3996010
1: -21.4323273, -12.2287083, -21.4430714, -12.2446251, -5.2815323, 5.2864342
2: -12.3987360, -5.7745323, -12.4051352, -5.7826362, -4.2770767, 4.2663422
3: -12.0143433, -4.1552062, -12.0184546, -4.1729908, -5.3571358, 5.3855324
4: -10.2982388, 0.0377033, -10.3035851, -0.0125008, -6.0265045, 6.1153908
5: -13.5652857, -4.0339422, -13.5696821, -4.0567613, -6.1535606, 6.1557465
6: -8.3534698, 0.5443816, -8.3155527, 0.5525755, -6.4959297, 6.4389153
7: -32.1578751, -22.0510426, -32.1635742, -22.0831909, -5.8259277, 5.8627357
8: -18.8173046, -9.0640450, -18.8251457, -9.1114960, -5.2072067, 5.2949657
9: -5.3468370, 1.4066513, -5.3338003, 1.3928545, -4.0594559, 4.0671329
10: -36.1405869, -27.7471561, -36.1477547, -27.7699986, -5.2502346, 5.2674046
11: -55.1680222, -44.7709236, -55.1294899, -44.8091049, -4.9804401, 4.9634876
12: -11.5831261, -4.5782962, -11.5775766, -4.5846987, -6.2319374, 6.2164726
13: 0.8781723, 8.0359831, 0.8807570, 8.0017500, -5.2902832, 5.3315659
14: -71.0884857, -57.9404831, -71.1044159, -57.9704742, -8.2623215, 8.2826233
15: -8.9242821, 0.9373827, -8.9315014, 0.8923173, -4.8662243, 4.9477959
16: -33.5916061, -23.9671898, -33.5649910, -23.9762421, -6.4962730, 6.4488068
17: -88.6785431, -72.3847809, -88.6895981, -72.4625092, -8.1595917, 8.2606010
18: -4.2018642, 1.0740857, -4.1712170, 1.0540221, -3.4225960, 3.3992958
19: -30.5341949, -23.2005215, -30.5219002, -23.2057381, -4.6558990, 4.6408997
20: -11.1741533, -5.1525631, -11.1695414, -5.1528502, -4.9245224, 4.9372196
21: -43.5632553, -35.0483246, -43.5446739, -35.0567322, -4.2794628, 4.2482758
22: -27.0088272, -19.5264912, -27.0074520, -19.5551491, -4.3317661, 4.3563442
23: -20.8763924, -12.5004616, -20.8333931, -12.5104465, -4.8182125, 4.7657280
24: -16.8831329, -7.6338539, -16.8486404, -7.6405964, -7.1950417, 7.1599922
25: -14.6446362, -6.9520350, -14.6259174, -6.9590917, -4.2079887, 4.1869259
26: -14.6220579, -7.7966609, -14.6143284, -7.8158598, -6.5331650, 6.5858002
27: -14.6423149, -9.5226860, -14.6279774, -9.5461149, -4.0648537, 4.0693512
28: -10.0379944, -1.4217486, -10.0155163, -1.4180576, -6.1456032, 6.1545448
29: -45.5971222, -36.8078003, -45.5881233, -36.8392525, -5.0112419, 5.0251846
30: -32.2193489, -22.9945431, -32.1839638, -23.0132618, -5.0350704, 5.0012550
31: -32.2609825, -23.5107632, -32.2290802, -23.5210590, -6.3309631, 6.3004761
32: 7.6957679, 13.6755619, 7.7205200, 13.6827049, -4.1866913, 4.1460266
33: 4.5851336, 16.3128567, 4.6389122, 16.3214397, -6.7368450, 6.6565590
34: 20.5142479, 30.9899006, 20.5610905, 30.9961567, -5.7768345, 5.7047157
35: 16.4761906, 26.8645267, 16.5339241, 26.8689785, -5.4865074, 5.4040947
36: 28.7949600, 35.1257401, 28.8243542, 35.1301842, -3.4563179, 3.4184589
37: 10.9967194, 20.1173611, 11.0538816, 20.1194649, -6.0006180, 5.9261551
38: 34.8386993, 43.7042656, 34.8889122, 43.6879387, -6.0693855, 6.0434570
39: 8.9638977, 18.5198002, 9.0170345, 18.5071869, -6.5628662, 6.5166740
40: 15.7625017, 25.1314182, 15.8037672, 25.1417904, -5.8435726, 5.7911968
41: 6.7074528, 13.2269201, 6.7404118, 13.2338543, -5.0480995, 4.9995689
42: -12.3984756, -3.4524648, -12.3765306, -3.4427500, -7.0616875, 7.0320511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1769

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5976869, upper bound: 3.6274036
time: 5.71 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5976869, upper bound: 3.6385848
time: 6.24 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -21.5619297, -8.4773722, -21.5866375, -8.4798555, -10.3969879, 10.4519196
1: -21.4274750, -12.2364798, -21.4367161, -12.2400799, -5.2730713, 5.2990417
2: -12.3931484, -5.7767048, -12.4002781, -5.7798905, -4.2589149, 4.2804394
3: -12.0097942, -4.1675053, -12.0053282, -4.1767206, -5.3655586, 5.3652725
4: -10.2834797, 0.0052062, -10.2933521, 0.0018618, -6.0525627, 6.0455627
5: -13.5547123, -4.0441275, -13.5695934, -4.0483322, -6.1311951, 6.1568184
6: -8.3182402, 0.5364146, -8.3123360, 0.5355163, -6.4611130, 6.4584541
7: -32.1555710, -22.0553322, -32.1593971, -22.0632896, -5.8246269, 5.8404427
8: -18.8051243, -9.1061897, -18.8150978, -9.1114178, -5.2297707, 5.2187099
9: -5.3312321, 1.3935900, -5.3376207, 1.3917122, -4.0445671, 4.0445061
10: -36.1338387, -27.7654114, -36.1334381, -27.7781754, -5.2540817, 5.2696438
11: -55.1314201, -44.8026009, -55.1075478, -44.8250504, -4.9295483, 4.9630642
12: -11.5780954, -4.5903926, -11.5676098, -4.5853772, -6.2409401, 6.2334938
13: 0.8903386, 8.0120716, 0.8908266, 8.0091162, -5.3001289, 5.2955780
14: -71.0809479, -57.9611511, -71.0922394, -57.9608994, -8.2387543, 8.2642479
15: -8.9083061, 0.9018717, -8.9171963, 0.8995957, -4.8820705, 4.8807964
16: -33.5564690, -23.9752178, -33.5507965, -23.9903221, -6.4417381, 6.4696960
17: -88.6748276, -72.4201279, -88.6680527, -72.4369736, -8.1746216, 8.1854134
18: -4.1768165, 1.0589516, -4.1754084, 1.0584340, -3.3868484, 3.3981686
19: -30.5246964, -23.2037487, -30.5126076, -23.2201805, -4.6467762, 4.6565418
20: -11.1714077, -5.1565566, -11.1662693, -5.1600118, -4.9396133, 4.9287338
21: -43.5467377, -35.0576553, -43.5304108, -35.0790253, -4.2601910, 4.2810001
22: -27.0041733, -19.5391560, -27.0006180, -19.5497398, -4.3271713, 4.3379669
23: -20.8512707, -12.5126324, -20.8453388, -12.5215569, -4.7662773, 4.7848892
24: -16.8587284, -7.6446004, -16.8570442, -7.6501842, -7.1515808, 7.1682625
25: -14.6369114, -6.9580545, -14.6289177, -6.9744134, -4.1875134, 4.2030640
26: -14.6169128, -7.8186760, -14.6106033, -7.8176260, -6.5491714, 6.5272217
27: -14.6300220, -9.5349121, -14.6214523, -9.5463037, -4.0549603, 4.0634480
28: -10.0222759, -1.4303051, -10.0202475, -1.4219553, -6.1682396, 6.1436501
29: -45.5825958, -36.8257675, -45.5734901, -36.8451042, -4.9963799, 5.0186634
30: -32.1857033, -23.0196495, -32.1729431, -23.0258293, -4.9791336, 5.0104027
31: -32.2348480, -23.5174179, -32.2220230, -23.5392361, -6.2995224, 6.3118401
32: 7.7184839, 13.6719704, 7.7226677, 13.6795893, -4.1600533, 4.1488438
33: 4.6098862, 16.3101139, 4.6213493, 16.3158302, -6.6845226, 6.6688309
34: 20.5355721, 30.9836159, 20.5626183, 30.9785671, -5.7354488, 5.7250404
35: 16.5049591, 26.8590469, 16.5262909, 26.8549862, -5.4306984, 5.4253922
36: 28.8102512, 35.1229897, 28.8226986, 35.1170006, -3.4279995, 3.4239435
37: 11.0254555, 20.1122360, 11.0314102, 20.1141968, -5.9498940, 5.9481697
38: 34.8629913, 43.6859436, 34.8763123, 43.6722984, -6.0360107, 6.0251045
39: 8.9978476, 18.5050278, 8.9996185, 18.5042343, -6.5110779, 6.5086288
40: 15.7928352, 25.1241760, 15.8022671, 25.1368313, -5.8109474, 5.7835503
41: 6.7332292, 13.2213287, 6.7359819, 13.2243853, -5.0125771, 5.0102692
42: -12.3865986, -3.4566994, -12.3782864, -3.4619930, -7.0512886, 7.0466270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5861122, upper bound: 3.6338235
time: 4.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5861122, upper bound: 3.6385283
time: 5.00 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5619297, -8.4773722, -21.5911064, -8.4790535, -10.3971786, 10.4556885
1: -21.4274750, -12.2364798, -21.4400368, -12.2365980, -5.2745705, 5.3004723
2: -12.3931484, -5.7767048, -12.4056711, -5.7779036, -4.2547626, 4.2797604
3: -12.0097942, -4.1675053, -12.0178432, -4.1672831, -5.3625145, 5.3660278
4: -10.2834797, 0.0052062, -10.3023214, 0.0049579, -6.0499001, 6.0473747
5: -13.5547123, -4.0441275, -13.5718594, -4.0451136, -6.1337776, 6.1586456
6: -8.3182402, 0.5364146, -8.3177443, 0.5448874, -6.4637756, 6.4575386
7: -32.1555710, -22.0553322, -32.1656609, -22.0550652, -5.8288612, 5.8439007
8: -18.8051243, -9.1061897, -18.8230228, -9.1059046, -5.2297211, 5.2199879
9: -5.3312321, 1.3935900, -5.3454633, 1.3941084, -4.0490265, 4.0557499
10: -36.1338387, -27.7654114, -36.1424294, -27.7644253, -5.2541180, 5.2661819
11: -55.1314201, -44.8026009, -55.1269379, -44.7895279, -4.9273930, 4.9440250
12: -11.5780954, -4.5903926, -11.5748425, -4.5761237, -6.2436447, 6.2358513
13: 0.8903386, 8.0120716, 0.8810621, 8.0137148, -5.2997513, 5.2990723
14: -71.0809479, -57.9611511, -71.0950394, -57.9582062, -8.2437134, 8.2688713
15: -8.9083061, 0.9018717, -8.9243431, 0.9019961, -4.8758335, 4.8766785
16: -33.5564690, -23.9752178, -33.5707893, -23.9677048, -6.4444008, 6.4690666
17: -88.6748276, -72.4201279, -88.6790314, -72.4183960, -8.1772003, 8.1827469
18: -4.1768165, 1.0589516, -4.1783781, 1.0635581, -3.3905926, 3.4001789
19: -30.5246964, -23.2037487, -30.5263824, -23.2014275, -4.6444397, 4.6494122
20: -11.1714077, -5.1565566, -11.1724176, -5.1503448, -4.9409409, 4.9270096
21: -43.5467377, -35.0576553, -43.5476990, -35.0521812, -4.2521572, 4.2643661
22: -27.0041733, -19.5391560, -27.0072136, -19.5395069, -4.3294563, 4.3364162
23: -20.8512707, -12.5126324, -20.8523273, -12.5075626, -4.7674522, 4.7800713
24: -16.8587284, -7.6446004, -16.8608551, -7.6403050, -7.1562347, 7.1681366
25: -14.6369114, -6.9580545, -14.6382084, -6.9580107, -4.1879349, 4.1962662
26: -14.6169128, -7.8186760, -14.6128016, -7.8139744, -6.5517387, 6.5281258
27: -14.6300220, -9.5349121, -14.6313915, -9.5307617, -4.0552979, 4.0595894
28: -10.0222759, -1.4303051, -10.0231848, -1.4190676, -6.1717873, 6.1467323
29: -45.5825958, -36.8257675, -45.5863647, -36.8219452, -4.9975357, 5.0090923
30: -32.1857033, -23.0196495, -32.1839027, -23.0042839, -4.9818249, 5.0013943
31: -32.2348480, -23.5174179, -32.2372932, -23.5173759, -6.2957115, 6.3032951
32: 7.7184839, 13.6719704, 7.7191091, 13.6801577, -4.1602802, 4.1522007
33: 4.6098862, 16.3101139, 4.6096745, 16.3190517, -6.6842613, 6.6805115
34: 20.5355721, 30.9836159, 20.5351658, 30.9971848, -5.7269211, 5.7247276
35: 16.5049591, 26.8590469, 16.5038719, 26.8676186, -5.4209976, 5.4263897
36: 28.8102512, 35.1229897, 28.8089943, 35.1255951, -3.4253740, 3.4242487
37: 11.0254555, 20.1122360, 11.0248880, 20.1180763, -5.9509735, 5.9531631
38: 34.8629913, 43.6859436, 34.8551369, 43.6875420, -6.0363350, 6.0301476
39: 8.9978476, 18.5050278, 8.9903088, 18.5050831, -6.5138359, 6.5193634
40: 15.7928352, 25.1241760, 15.7911978, 25.1385899, -5.8105087, 5.7935505
41: 6.7332292, 13.2213287, 6.7329597, 13.2273169, -5.0141258, 5.0122414
42: -12.3865986, -3.4566994, -12.3864899, -3.4491448, -7.0530090, 7.0443230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5861122, upper bound: 3.6338241
time: 4.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5861122, upper bound: 3.6385284
time: 5.12 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -21.5712013, -8.4745445, -21.5865288, -8.4797583, -10.4289017, 10.4494934
1: -21.4292068, -12.2341709, -21.4367561, -12.2403631, -5.2793217, 5.3001823
2: -12.3961334, -5.7738323, -12.4003696, -5.7799087, -4.2682495, 4.2792416
3: -12.0138292, -4.1586652, -12.0054617, -4.1766753, -5.3695183, 5.3741226
4: -10.2941751, 0.0261444, -10.2937164, 0.0018325, -6.0628128, 6.0669155
5: -13.5611839, -4.0368090, -13.5697784, -4.0483270, -6.1408310, 6.1550064
6: -8.3326054, 0.5417062, -8.3123312, 0.5357430, -6.4757881, 6.4639511
7: -32.1565247, -22.0567474, -32.1593971, -22.0650387, -5.8276443, 5.8449821
8: -18.8167439, -9.0704250, -18.8156147, -9.1114292, -5.2411289, 5.2546368
9: -5.3357601, 1.4032066, -5.3377705, 1.3917158, -4.0489006, 4.0541611
10: -36.1393089, -27.7589531, -36.1335678, -27.7781715, -5.2598152, 5.2716255
11: -55.1666031, -44.7890816, -55.1075745, -44.8244553, -4.9654827, 4.9763165
12: -11.5801878, -4.5829000, -11.5675936, -4.5851936, -6.2418594, 6.2403564
13: 0.8843166, 8.0314093, 0.8905675, 8.0091867, -5.3062439, 5.3154373
14: -71.0849838, -57.9543381, -71.0922699, -57.9608688, -8.2523460, 8.2694588
15: -8.9212494, 0.9234982, -8.9176083, 0.8995981, -4.8950996, 4.9027672
16: -33.5799179, -23.9668121, -33.5508194, -23.9899750, -6.4654350, 6.4774818
17: -88.6751022, -72.4171143, -88.6679993, -72.4382172, -8.1747742, 8.1905098
18: -4.2004910, 1.0638583, -4.1754560, 1.0586553, -3.4107723, 3.4029675
19: -30.5337162, -23.2007828, -30.5126457, -23.2200851, -4.6563339, 4.6590824
20: -11.1735220, -5.1553659, -11.1662779, -5.1607828, -4.9392967, 4.9330120
21: -43.5635910, -35.0508575, -43.5304298, -35.0787354, -4.2772884, 4.2877159
22: -27.0076523, -19.5385265, -27.0004654, -19.5497437, -4.3310852, 4.3379650
23: -20.8707581, -12.5036325, -20.8453674, -12.5212345, -4.7861195, 4.7944660
24: -16.8786201, -7.6371322, -16.8571033, -7.6498699, -7.1716919, 7.1757050
25: -14.6409426, -6.9553928, -14.6289577, -6.9743328, -4.1920719, 4.2058754
26: -14.6213207, -7.8152680, -14.6107244, -7.8179893, -6.5493202, 6.5456123
27: -14.6414003, -9.5296516, -14.6215277, -9.5461197, -4.0659409, 4.0687885
28: -10.0370512, -1.4217665, -10.0202713, -1.4216632, -6.1683578, 6.1512032
29: -45.5955009, -36.8215485, -45.5735359, -36.8449135, -5.0094147, 5.0219860
30: -32.2179909, -23.0039616, -32.1729431, -23.0252876, -5.0120220, 5.0263557
31: -32.2542114, -23.5130119, -32.2221184, -23.5390129, -6.3202591, 6.3157387
32: 7.7135353, 13.6741858, 7.7226386, 13.6796684, -4.1652126, 4.1509895
33: 4.6071444, 16.3113937, 4.6220932, 16.3158340, -6.6901550, 6.6700592
34: 20.5233307, 30.9885445, 20.5625839, 30.9787750, -5.7478867, 5.7295837
35: 16.4910965, 26.8641415, 16.5262413, 26.8552113, -5.4447098, 5.4304295
36: 28.8081589, 35.1246338, 28.8226566, 35.1170387, -3.4300308, 3.4254417
37: 11.0160084, 20.1157684, 11.0313387, 20.1143208, -5.9598274, 5.9513779
38: 34.8522034, 43.6998444, 34.8759842, 43.6723061, -6.0470314, 6.0393333
39: 8.9885216, 18.5167732, 8.9994221, 18.5042229, -6.5201988, 6.5180511
40: 15.7819347, 25.1290359, 15.8021736, 25.1368484, -5.8209457, 5.7916317
41: 6.7243423, 13.2256651, 6.7359509, 13.2245541, -5.0215454, 5.0145760
42: -12.3865757, -3.4539957, -12.3780327, -3.4619799, -7.0514832, 7.0524712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5981283, upper bound: 3.6338756
time: 5.59 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5981283, upper bound: 3.6385803
time: 4.95 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5712013, -8.4745445, -21.5909901, -8.4790220, -10.4290924, 10.4532547
1: -21.4292068, -12.2341709, -21.4400692, -12.2368908, -5.2808247, 5.3016033
2: -12.3961334, -5.7738323, -12.4057627, -5.7779083, -4.2640953, 4.2785549
3: -12.0138292, -4.1586652, -12.0179863, -4.1672697, -5.3664665, 5.3748817
4: -10.2941751, 0.0261444, -10.3026886, 0.0049224, -6.0601540, 6.0687351
5: -13.5611839, -4.0368090, -13.5720196, -4.0451317, -6.1434250, 6.1568375
6: -8.3326054, 0.5417062, -8.3177471, 0.5451051, -6.4784508, 6.4630394
7: -32.1565247, -22.0567474, -32.1656647, -22.0568161, -5.8318901, 5.8484535
8: -18.8167439, -9.0704250, -18.8235111, -9.1058836, -5.2410679, 5.2558784
9: -5.3357601, 1.4032066, -5.3456097, 1.3941355, -4.0533562, 4.0654068
10: -36.1393089, -27.7589531, -36.1425705, -27.7643890, -5.2598457, 5.2681561
11: -55.1666031, -44.7890816, -55.1269302, -44.7889481, -4.9633350, 4.9572735
12: -11.5801878, -4.5829000, -11.5748186, -4.5759583, -6.2445679, 6.2427216
13: 0.8843166, 8.0314093, 0.8808190, 8.0137901, -5.3058853, 5.3189316
14: -71.0849838, -57.9543381, -71.0950623, -57.9581604, -8.2572937, 8.2740860
15: -8.9212494, 0.9234982, -8.9247675, 0.9020000, -4.8888988, 4.8986454
16: -33.5799179, -23.9668121, -33.5708160, -23.9673195, -6.4681015, 6.4768448
17: -88.6751022, -72.4171143, -88.6789856, -72.4196701, -8.1773415, 8.1878662
18: -4.2004910, 1.0638583, -4.1784062, 1.0637646, -3.4145069, 3.4049740
19: -30.5337162, -23.2007828, -30.5264053, -23.2013321, -4.6539783, 4.6519623
20: -11.1735220, -5.1553659, -11.1724339, -5.1511006, -4.9406395, 4.9313107
21: -43.5635910, -35.0508575, -43.5477371, -35.0519180, -4.2692528, 4.2710838
22: -27.0076523, -19.5385265, -27.0070457, -19.5395012, -4.3333702, 4.3364182
23: -20.8707581, -12.5036325, -20.8523483, -12.5072422, -4.7873001, 4.7896519
24: -16.8786201, -7.6371322, -16.8609352, -7.6399965, -7.1763458, 7.1755714
25: -14.6409426, -6.9553928, -14.6382627, -6.9579268, -4.1924953, 4.1990833
26: -14.6213207, -7.8152680, -14.6128912, -7.8143210, -6.5519066, 6.5465240
27: -14.6414003, -9.5296516, -14.6314735, -9.5305691, -4.0662804, 4.0649261
28: -10.0370512, -1.4217665, -10.0232248, -1.4187319, -6.1719131, 6.1542931
29: -45.5955009, -36.8215485, -45.5863686, -36.8217545, -5.0105705, 5.0124245
30: -32.2179909, -23.0039616, -32.1839066, -23.0037136, -5.0147171, 5.0173588
31: -32.2542114, -23.5130119, -32.2373619, -23.5171890, -6.3164024, 6.3071823
32: 7.7135353, 13.6741858, 7.7190766, 13.6802368, -4.1654320, 4.1543503
33: 4.6071444, 16.3113937, 4.6103926, 16.3190708, -6.6898975, 6.6817398
34: 20.5233307, 30.9885445, 20.5351562, 30.9973602, -5.7393475, 5.7292595
35: 16.4910965, 26.8641415, 16.5037994, 26.8678360, -5.4350033, 5.4314289
36: 28.8081589, 35.1246338, 28.8089371, 35.1256294, -3.4273977, 3.4257355
37: 11.0160084, 20.1157684, 11.0248508, 20.1182060, -5.9608917, 5.9563828
38: 34.8522034, 43.6998444, 34.8548050, 43.6875496, -6.0473747, 6.0443649
39: 8.9885216, 18.5167732, 8.9901352, 18.5050697, -6.5229454, 6.5287781
40: 15.7819347, 25.1290359, 15.7911005, 25.1386223, -5.8205376, 5.8016224
41: 6.7243423, 13.2256651, 6.7329211, 13.2274628, -5.0230942, 5.0165443
42: -12.3865757, -3.4539957, -12.3862467, -3.4491258, -7.0532227, 7.0501671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1597

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5981283, upper bound: 3.6338759
time: 6.07 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5981283, upper bound: 3.6385803
time: 5.45 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -21.5627232, -8.4772921, -21.6067429, -8.4775076, -10.4039993, 10.4837036
1: -21.4280739, -12.2364712, -21.4451523, -12.2345753, -5.2784615, 5.3081436
2: -12.3934536, -5.7769403, -12.4043865, -5.7796445, -4.2614059, 4.2846870
3: -12.0099773, -4.1672506, -12.0085096, -4.1722527, -5.3697701, 5.3686333
4: -10.2847805, 0.0054901, -10.3038568, 0.0140827, -6.0683250, 6.0544243
5: -13.5550461, -4.0442514, -13.5755463, -4.0436630, -6.1384583, 6.1662788
6: -8.3186693, 0.5391127, -8.3346138, 0.5492530, -6.4723320, 6.4828911
7: -32.1561928, -22.0552406, -32.1657257, -22.0575333, -5.8320045, 5.8462067
8: -18.8055248, -9.1059799, -18.8196373, -9.1048641, -5.2388592, 5.2234631
9: -5.3311853, 1.3937064, -5.3511596, 1.3953217, -4.0482731, 4.0616360
10: -36.1352158, -27.7653694, -36.1423721, -27.7659111, -5.2634010, 5.2777309
11: -55.1326370, -44.8024902, -55.1145782, -44.8063278, -4.9495335, 4.9684658
12: -11.5789528, -4.5904074, -11.5740347, -4.5802250, -6.2414246, 6.2394409
13: 0.8899189, 8.0122671, 0.8810161, 8.0142717, -5.3072624, 5.3056641
14: -71.0834961, -57.9611511, -71.1087189, -57.9469070, -8.2526665, 8.2792358
15: -8.9104776, 0.9020371, -8.9293566, 0.9135799, -4.9016590, 4.8905163
16: -33.5571976, -23.9754238, -33.5656128, -23.9899826, -6.4433441, 6.4980621
17: -88.6790848, -72.4197464, -88.6895218, -72.4025574, -8.2154121, 8.2043991
18: -4.1773944, 1.0590491, -4.1795998, 1.0692179, -3.3983212, 3.4025288
19: -30.5246582, -23.2036171, -30.5140076, -23.2194061, -4.6478729, 4.6586990
20: -11.1713924, -5.1563950, -11.1672916, -5.1549535, -4.9458923, 4.9301739
21: -43.5466080, -35.0575447, -43.5321655, -35.0760574, -4.2629871, 4.2854099
22: -27.0054016, -19.5390797, -27.0079155, -19.5372276, -4.3422623, 4.3447628
23: -20.8514099, -12.5126781, -20.8514061, -12.5158138, -4.7736588, 4.7942543
24: -16.8589611, -7.6445961, -16.8621178, -7.6433229, -7.1584549, 7.1738586
25: -14.6369801, -6.9578934, -14.6331015, -6.9690037, -4.1936741, 4.2086449
26: -14.6184864, -7.8181419, -14.6188831, -7.7965016, -6.5715179, 6.5357971
27: -14.6303577, -9.5347786, -14.6244583, -9.5388107, -4.0622158, 4.0669441
28: -10.0222034, -1.4300946, -10.0220451, -1.4199904, -6.1719017, 6.1471786
29: -45.5838242, -36.8256378, -45.5819321, -36.8308258, -5.0118065, 5.0271664
30: -32.1860428, -23.0194817, -32.1765862, -23.0156670, -4.9894295, 5.0138779
31: -32.2351036, -23.5167885, -32.2299080, -23.5326195, -6.3076630, 6.3234482
32: 7.7182083, 13.6737356, 7.7038970, 13.6886263, -4.1667862, 4.1698723
33: 4.6094999, 16.3114758, 4.5978489, 16.3248177, -6.6925926, 6.6993484
34: 20.5352726, 30.9846230, 20.5525227, 30.9860573, -5.7429237, 5.7383137
35: 16.5047264, 26.8602104, 16.5106258, 26.8622093, -5.4379272, 5.4454174
36: 28.8101501, 35.1245689, 28.8091164, 35.1251106, -3.4349575, 3.4399176
37: 11.0250721, 20.1137028, 11.0106468, 20.1218872, -5.9571037, 5.9730072
38: 34.8628540, 43.6879807, 34.8619080, 43.6853180, -6.0478783, 6.0418587
39: 8.9975023, 18.5068645, 8.9733047, 18.5150833, -6.5212593, 6.5377808
40: 15.7925520, 25.1255608, 15.7817411, 25.1452255, -5.8176041, 5.8064938
41: 6.7328939, 13.2231102, 6.7179751, 13.2337828, -5.0213203, 5.0310593
42: -12.3867826, -3.4549942, -12.3908348, -3.4527154, -7.0596771, 7.0601463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5964923, upper bound: 3.6307847
time: 5.06 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5964923, upper bound: 3.6381776
time: 5.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5627232, -8.4772921, -21.6112232, -8.4767675, -10.4041901, 10.4874802
1: -21.4280739, -12.2364712, -21.4484272, -12.2311249, -5.2799740, 5.3095608
2: -12.3934536, -5.7769403, -12.4097919, -5.7776732, -4.2572556, 4.2840061
3: -12.0099773, -4.1672506, -12.0210295, -4.1628342, -5.3667221, 5.3694000
4: -10.2847805, 0.0054901, -10.3128519, 0.0171998, -6.0656548, 6.0562325
5: -13.5550461, -4.0442514, -13.5778303, -4.0404272, -6.1410255, 6.1681137
6: -8.3186693, 0.5391127, -8.3400488, 0.5586404, -6.4749832, 6.4819908
7: -32.1561928, -22.0552406, -32.1720009, -22.0493011, -5.8362465, 5.8496609
8: -18.8055248, -9.1059799, -18.8275414, -9.0993462, -5.2387962, 5.2247295
9: -5.3311853, 1.3937064, -5.3589177, 1.3977175, -4.0527363, 4.0729256
10: -36.1352158, -27.7653694, -36.1514015, -27.7521515, -5.2634659, 5.2742691
11: -55.1326370, -44.8024902, -55.1339188, -44.7707672, -4.9474030, 4.9494286
12: -11.5789528, -4.5904074, -11.5812454, -4.5709634, -6.2441330, 6.2417831
13: 0.8899189, 8.0122671, 0.8712761, 8.0188684, -5.3069038, 5.3091507
14: -71.0834961, -57.9611511, -71.1114655, -57.9442215, -8.2576027, 8.2838707
15: -8.9104776, 0.9020371, -8.9364948, 0.9159517, -4.8954220, 4.8863945
16: -33.5571976, -23.9754238, -33.5855713, -23.9673481, -6.4460182, 6.4974518
17: -88.6790848, -72.4197464, -88.7004929, -72.3839722, -8.2179985, 8.2017403
18: -4.1773944, 1.0590491, -4.1825719, 1.0743380, -3.4020844, 3.4045334
19: -30.5246582, -23.2036171, -30.5277786, -23.2006607, -4.6455383, 4.6515751
20: -11.1713924, -5.1563950, -11.1734362, -5.1452684, -4.9472122, 4.9284706
21: -43.5466080, -35.0575447, -43.5494537, -35.0492210, -4.2549534, 4.2687759
22: -27.0054016, -19.5390797, -27.0144958, -19.5269909, -4.3445473, 4.3432178
23: -20.8514099, -12.5126781, -20.8583851, -12.5018196, -4.7748146, 4.7894344
24: -16.8589611, -7.6445961, -16.8659554, -7.6333833, -7.1631012, 7.1737213
25: -14.6369801, -6.9578934, -14.6423969, -6.9526095, -4.1940804, 4.2018547
26: -14.6184864, -7.8181419, -14.6210661, -7.7928371, -6.5740967, 6.5367012
27: -14.6303577, -9.5347786, -14.6343994, -9.5232315, -4.0625725, 4.0630798
28: -10.0222034, -1.4300946, -10.0249758, -1.4170734, -6.1754684, 6.1502762
29: -45.5838242, -36.8256378, -45.5947914, -36.8076477, -5.0129929, 5.0176010
30: -32.1860428, -23.0194817, -32.1875801, -22.9940910, -4.9921360, 5.0048656
31: -32.2351036, -23.5167885, -32.2451668, -23.5108185, -6.3038330, 6.3148994
32: 7.7182083, 13.6737356, 7.7003078, 13.6891985, -4.1670113, 4.1732235
33: 4.6094999, 16.3114758, 4.5860949, 16.3280659, -6.6923294, 6.7110291
34: 20.5352726, 30.9846230, 20.5251331, 31.0046539, -5.7343845, 5.7379837
35: 16.5047264, 26.8602104, 16.4881248, 26.8748550, -5.4282322, 5.4464340
36: 28.8101501, 35.1245689, 28.7954006, 35.1336937, -3.4323292, 3.4402237
37: 11.0250721, 20.1137028, 11.0041008, 20.1257629, -5.9581718, 5.9780197
38: 34.8628540, 43.6879807, 34.8407593, 43.7005463, -6.0482254, 6.0468941
39: 8.9975023, 18.5068645, 8.9639769, 18.5159321, -6.5240135, 6.5485382
40: 15.7925520, 25.1255608, 15.7706423, 25.1469765, -5.8171673, 5.8165588
41: 6.7328939, 13.2231102, 6.7149444, 13.2367287, -5.0228577, 5.0330200
42: -12.3867826, -3.4549942, -12.3990126, -3.4398549, -7.0613823, 7.0578461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5964923, upper bound: 3.6307851
time: 5.08 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5964923, upper bound: 3.6381779
time: 6.10 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -21.5719643, -8.4744654, -21.6065865, -8.4774885, -10.4359360, 10.4812927
1: -21.4298286, -12.2341661, -21.4451981, -12.2348690, -5.2847176, 5.3092709
2: -12.3964539, -5.7740707, -12.4044600, -5.7796640, -4.2707367, 4.2834892
3: -12.0139809, -4.1584234, -12.0086393, -4.1722159, -5.3737259, 5.3774872
4: -10.2955112, 0.0264397, -10.3042870, 0.0140675, -6.0785637, 6.0757751
5: -13.5615482, -4.0369344, -13.5757027, -4.0436287, -6.1480865, 6.1644745
6: -8.3330383, 0.5444269, -8.3346205, 0.5494707, -6.4870186, 6.4884186
7: -32.1571808, -22.0566597, -32.1657143, -22.0592823, -5.8350296, 5.8507347
8: -18.8171673, -9.0702333, -18.8201447, -9.1048098, -5.2502060, 5.2593689
9: -5.3357143, 1.4033101, -5.3512964, 1.3953402, -4.0526009, 4.0713043
10: -36.1407013, -27.7589378, -36.1425133, -27.7658863, -5.2691422, 5.2797127
11: -55.1678543, -44.7889709, -55.1145554, -44.8057327, -4.9854641, 4.9817085
12: -11.5810184, -4.5828791, -11.5740376, -4.5800767, -6.2423210, 6.2462921
13: 0.8838949, 8.0315981, 0.8807597, 8.0143404, -5.3134003, 5.3255196
14: -71.0875854, -57.9542923, -71.1087570, -57.9468689, -8.2662392, 8.2844543
15: -8.9234180, 0.9236460, -8.9297638, 0.9135690, -4.9146767, 4.9124908
16: -33.5806274, -23.9670105, -33.5656395, -23.9896183, -6.4670525, 6.5058365
17: -88.6793518, -72.4167557, -88.6894684, -72.4038391, -8.2155457, 8.2095070
18: -4.2010555, 1.0639727, -4.1796393, 1.0694335, -3.4222412, 3.4073238
19: -30.5336761, -23.2006760, -30.5140457, -23.2193298, -4.6574249, 4.6612434
20: -11.1735010, -5.1551943, -11.1673164, -5.1556959, -4.9456062, 4.9344463
21: -43.5634460, -35.0507584, -43.5321999, -35.0757675, -4.2800903, 4.2921276
22: -27.0088501, -19.5384293, -27.0077553, -19.5372467, -4.3461685, 4.3447590
23: -20.8708992, -12.5036736, -20.8514481, -12.5154963, -4.7934895, 4.8038368
24: -16.8788681, -7.6371408, -16.8621998, -7.6429901, -7.1785965, 7.1812897
25: -14.6410360, -6.9552412, -14.6331329, -6.9689341, -4.1982231, 4.2114620
26: -14.6228666, -7.8147125, -14.6189651, -7.7968502, -6.5716896, 6.5541916
27: -14.6417179, -9.5295048, -14.6245289, -9.5386028, -4.0731983, 4.0722771
28: -10.0369825, -1.4215946, -10.0220613, -1.4197037, -6.1720314, 6.1547356
29: -45.5967712, -36.8213959, -45.5819626, -36.8306274, -5.0248451, 5.0305023
30: -32.2183380, -23.0037594, -32.1766090, -23.0151176, -5.0223045, 5.0298252
31: -32.2544479, -23.5124130, -32.2299995, -23.5323944, -6.3284149, 6.3273239
32: 7.7132888, 13.6759510, 7.7038898, 13.6887054, -4.1719475, 4.1720161
33: 4.6067715, 16.3127689, 4.5985961, 16.3248405, -6.6982155, 6.7005692
34: 20.5230103, 30.9895992, 20.5524731, 30.9862709, -5.7553539, 5.7428474
35: 16.4908638, 26.8653107, 16.5105553, 26.8624344, -5.4519444, 5.4504452
36: 28.8080559, 35.1262283, 28.8090668, 35.1251488, -3.4369888, 3.4414120
37: 11.0156183, 20.1172371, 11.0105944, 20.1219940, -5.9670296, 5.9762344
38: 34.8521080, 43.7018890, 34.8615723, 43.6853104, -6.0588951, 6.0560608
39: 8.9882240, 18.5185986, 8.9731073, 18.5150700, -6.5303764, 6.5472031
40: 15.7816811, 25.1304359, 15.7816658, 25.1452293, -5.8276043, 5.8145790
41: 6.7240095, 13.2274685, 6.7179379, 13.2339430, -5.0302849, 5.0353508
42: -12.3867693, -3.4522972, -12.3905668, -3.4527099, -7.0598679, 7.0659866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6085125, upper bound: 3.6308365
time: 5.24 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6085125, upper bound: 3.6382295
time: 5.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5719643, -8.4744654, -21.6111069, -8.4767179, -10.4360809, 10.4850616
1: -21.4298286, -12.2341661, -21.4484539, -12.2313948, -5.2862339, 5.3106937
2: -12.3964539, -5.7740707, -12.4098740, -5.7776699, -4.2665787, 4.2828045
3: -12.0139809, -4.1584234, -12.0211668, -4.1628208, -5.3706703, 5.3782425
4: -10.2955112, 0.0264397, -10.3132248, 0.0171999, -6.0759201, 6.0775833
5: -13.5615482, -4.0369344, -13.5779819, -4.0404377, -6.1506615, 6.1663055
6: -8.3330383, 0.5444269, -8.3400393, 0.5588813, -6.4896622, 6.4875031
7: -32.1571808, -22.0566597, -32.1719437, -22.0510597, -5.8392601, 5.8541985
8: -18.8171673, -9.0702333, -18.8280506, -9.0993099, -5.2501316, 5.2606125
9: -5.3357143, 1.4033101, -5.3590589, 1.3977364, -4.0570564, 4.0825939
10: -36.1407013, -27.7589378, -36.1515236, -27.7521019, -5.2691879, 5.2762527
11: -55.1678543, -44.7889709, -55.1339264, -44.7701797, -4.9833393, 4.9626751
12: -11.5810184, -4.5828791, -11.5812340, -4.5708175, -6.2450218, 6.2486496
13: 0.8838949, 8.0315981, 0.8710285, 8.0189581, -5.3130302, 5.3290024
14: -71.0875854, -57.9542923, -71.1115341, -57.9441795, -8.2712173, 8.2890930
15: -8.9234180, 0.9236460, -8.9369011, 0.9159765, -4.9084702, 4.9083729
16: -33.5806274, -23.9670105, -33.5855942, -23.9669838, -6.4697151, 6.5052338
17: -88.6793518, -72.4167557, -88.7004395, -72.3852386, -8.2181435, 8.2068520
18: -4.2010555, 1.0639727, -4.1825886, 1.0745409, -3.4260025, 3.4093246
19: -30.5336761, -23.2006760, -30.5278206, -23.2005844, -4.6550674, 4.6541061
20: -11.1735010, -5.1551943, -11.1734638, -5.1460285, -4.9469261, 4.9327526
21: -43.5634460, -35.0507584, -43.5494843, -35.0489349, -4.2720604, 4.2754936
22: -27.0088501, -19.5384293, -27.0143127, -19.5270042, -4.3484535, 4.3432102
23: -20.8708992, -12.5036736, -20.8584175, -12.5015163, -4.7946358, 4.7990322
24: -16.8788681, -7.6371408, -16.8660355, -7.6330824, -7.1832314, 7.1811523
25: -14.6410360, -6.9552412, -14.6424599, -6.9525428, -4.1986427, 4.2046623
26: -14.6228666, -7.8147125, -14.6211395, -7.7931814, -6.5742760, 6.5550995
27: -14.6417179, -9.5295048, -14.6344633, -9.5230541, -4.0735531, 4.0684223
28: -10.0369825, -1.4215946, -10.0250149, -1.4167897, -6.1755905, 6.1578178
29: -45.5967712, -36.8213959, -45.5948334, -36.8074379, -5.0260315, 5.0209427
30: -32.2183380, -23.0037594, -32.1875839, -22.9935684, -5.0250130, 5.0208263
31: -32.2544479, -23.5124130, -32.2452354, -23.5106201, -6.3245544, 6.3187714
32: 7.7132888, 13.6759510, 7.7003083, 13.6892700, -4.1721706, 4.1753750
33: 4.6067715, 16.3127689, 4.5868034, 16.3280907, -6.6979733, 6.7122421
34: 20.5230103, 30.9895992, 20.5250759, 31.0048370, -5.7468147, 5.7425232
35: 16.4908638, 26.8653107, 16.4880466, 26.8750725, -5.4422493, 5.4514542
36: 28.8080559, 35.1262283, 28.7953377, 35.1337280, -3.4343548, 3.4417067
37: 11.0156183, 20.1172371, 11.0040417, 20.1258755, -5.9681015, 5.9812355
38: 34.8521080, 43.7018890, 34.8403931, 43.7005539, -6.0592499, 6.0611000
39: 8.9882240, 18.5185986, 8.9637938, 18.5159111, -6.5331306, 6.5579681
40: 15.7816811, 25.1304359, 15.7705374, 25.1470146, -5.8271942, 5.8246231
41: 6.7240095, 13.2274685, 6.7149076, 13.2368793, -5.0318336, 5.0373116
42: -12.3867693, -3.4522972, -12.3987608, -3.4398279, -7.0615921, 7.0636787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=77, inp2_unstable=77, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.6085125, upper bound: 3.6308369
time: 5.98 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.6085125, upper bound: 3.6382298
time: 4.59 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 12.66 seconds
IS_A2_B1_A2_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6307889
IS_A2_B1_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6381818
IS_A2_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5973363, upper bound: 3.6308410
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5973363, upper bound: 3.6382336
IS_A2_B1_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6307891
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6381821
IS_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5973363, upper bound: 3.6308413
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5973363, upper bound: 3.6382340
IS_A2_B1_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6273534
IS_A2_B1_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6385325
IS_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5976869, upper bound: 3.6274033
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5976869, upper bound: 3.6385844
IS_A2_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6273537
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6385328
IS_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5976869, upper bound: 3.6274036
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5976869, upper bound: 3.6385848
IS_A2_B2_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5861122, upper bound: 3.6338235
IS_A2_B2_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5861122, upper bound: 3.6385283
IS_A2_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5861122, upper bound: 3.6338241
IS_A2_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5861122, upper bound: 3.6385284
IS_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5981283, upper bound: 3.6338756
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5981283, upper bound: 3.6385803
IS_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5981283, upper bound: 3.6338759
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5981283, upper bound: 3.6385803
IS_A2_B2_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5964923, upper bound: 3.6307847
IS_A2_B2_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5964923, upper bound: 3.6381776
IS_A2_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5964923, upper bound: 3.6307851
IS_A2_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.5964923, upper bound: 3.6381779
IS_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.6085125, upper bound: 3.6308365
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.6085125, upper bound: 3.6382295
IS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.6085125, upper bound: 3.6308369
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 12.66
Output dim: 38, lower bound: -3.6085125, upper bound: 3.6382298

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -21.5742188, -8.4789572, -21.5428028, -8.4853430, -10.4193726, 10.3741264
1: -21.4290352, -12.2320938, -21.4359684, -12.2506914, -5.2381802, 5.2796555
2: -12.3949013, -5.7782068, -12.3962708, -5.7866936, -4.2480202, 4.2669334
3: -12.0095510, -4.1682668, -11.9991484, -4.1954479, -5.3407555, 5.3647232
4: -10.2857018, 0.0084291, -10.2787170, -0.0386372, -6.0028229, 6.0658932
5: -13.5576019, -4.0438442, -13.5601997, -4.0680780, -6.1219788, 6.1535835
6: -8.3326015, 0.5378714, -8.2909203, 0.5336348, -6.4621391, 6.4122925
7: -32.1557770, -22.0512791, -32.1554146, -22.0943298, -5.7956390, 5.8543053
8: -18.8046989, -9.1146021, -18.7977848, -9.1586571, -5.1725998, 5.2227364
9: -5.3412209, 1.3932781, -5.3175044, 1.3793133, -4.0388680, 4.0335331
10: -36.1330109, -27.7551689, -36.1319427, -27.7889347, -5.2217102, 5.2637444
11: -55.1211739, -44.7848816, -55.0803528, -44.8624458, -4.9152470, 4.9655857
12: -11.5786514, -4.5867538, -11.5635166, -4.6022487, -6.2192383, 6.2059822
13: 0.8850516, 8.0086393, 0.9014567, 7.9730606, -5.2609024, 5.2893333
14: -71.0823975, -57.9487801, -71.0977707, -57.9765892, -8.1977158, 8.2687340
15: -8.9086590, 0.9085894, -8.9081869, 0.8695993, -4.8368168, 4.9048119
16: -33.5579758, -23.9757271, -33.5178223, -24.0127811, -6.4534874, 6.4232864
17: -88.6756668, -72.3891602, -88.6762848, -72.4889526, -8.1282082, 8.2538681
18: -4.1684327, 1.0690997, -4.1418195, 1.0402865, -3.3755322, 3.3792019
19: -30.5212402, -23.2038021, -30.4963188, -23.2299461, -4.6388130, 4.6424541
20: -11.1717358, -5.1547604, -11.1622963, -5.1651626, -4.9201889, 4.9204025
21: -43.5403595, -35.0556870, -43.5098839, -35.0941200, -4.2525482, 4.2540245
22: -27.0025520, -19.5272312, -26.9936943, -19.5682907, -4.3142815, 4.3555870
23: -20.8487282, -12.5109901, -20.8033085, -12.5397158, -4.7727032, 4.7445221
24: -16.8535881, -7.6420646, -16.8164673, -7.6663270, -7.1450729, 7.1296959
25: -14.6369104, -6.9556646, -14.6052284, -6.9831676, -4.1906185, 4.1786518
26: -14.6165190, -7.8010273, -14.6097393, -7.8243151, -6.5254402, 6.5359612
27: -14.6268396, -9.5285769, -14.6046228, -9.5700130, -4.0416889, 4.0596046
28: -10.0195856, -1.4316912, -10.0020370, -1.4320014, -6.1396332, 6.1278954
29: -45.5766602, -36.8121262, -45.5548248, -36.8733215, -4.9729767, 5.0211296
30: -32.1762924, -23.0122414, -32.1431274, -23.0556736, -4.9649029, 4.9860668
31: -32.2324371, -23.5158768, -32.1862717, -23.5535393, -6.2945824, 6.2757645
32: 7.7026072, 13.6725559, 7.7300644, 13.6789923, -4.1728859, 4.1290436
33: 4.5920687, 16.3102798, 4.6630840, 16.3122120, -6.7228718, 6.6243973
34: 20.5322151, 30.9840355, 20.6055241, 30.9684124, -5.7581520, 5.6893234
35: 16.4964237, 26.8584499, 16.5757732, 26.8459663, -5.4656944, 5.3830318
36: 28.7978973, 35.1231842, 28.8415279, 35.1188316, -3.4542522, 3.4070435
37: 11.0117073, 20.1127853, 11.0777960, 20.1082230, -5.9761505, 5.8992310
38: 34.8505974, 43.6846619, 34.9237289, 43.6579895, -6.0531578, 5.9972610
39: 8.9739828, 18.5052071, 9.0364857, 18.5009995, -6.5489235, 6.4769135
40: 15.7768478, 25.1227722, 15.8296623, 25.1311131, -5.8232193, 5.7431831
41: 6.7209663, 13.2216759, 6.7578588, 13.2229080, -5.0250244, 4.9787560
42: -12.3980618, -3.4565544, -12.3675060, -3.4589045, -7.0569992, 7.0198708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 759

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6316468
time: 5.94 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6381818
time: 5.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5834599, -8.4761534, -21.5426540, -8.4853172, -10.4512787, 10.3717232
1: -21.4307556, -12.2297544, -21.4359703, -12.2509413, -5.2444420, 5.2808247
2: -12.3979378, -5.7753315, -12.3963680, -5.7866907, -4.2573547, 4.2657337
3: -12.0135527, -4.1594543, -11.9992752, -4.1954341, -5.3447304, 5.3735733
4: -10.2963762, 0.0293608, -10.2790880, -0.0386556, -6.0130920, 6.0872650
5: -13.5640945, -4.0365562, -13.5603886, -4.0680285, -6.1316109, 6.1517601
6: -8.3469887, 0.5431714, -8.2909336, 0.5338515, -6.4768066, 6.4178047
7: -32.1567383, -22.0526676, -32.1554184, -22.0960770, -5.7986794, 5.8588581
8: -18.8163052, -9.0788727, -18.7982540, -9.1586342, -5.1839504, 5.2586365
9: -5.3457656, 1.4028986, -5.3176699, 1.3793449, -4.0431938, 4.0431995
10: -36.1385002, -27.7487221, -36.1320724, -27.7888908, -5.2274704, 5.2657261
11: -55.1563911, -44.7713699, -55.0803719, -44.8618927, -4.9511948, 4.9788399
12: -11.5807209, -4.5792556, -11.5635042, -4.6020761, -6.2201385, 6.2128220
13: 0.8790204, 8.0279617, 0.9012302, 7.9731154, -5.2670326, 5.3091812
14: -71.0864716, -57.9418716, -71.0978241, -57.9765320, -8.2113190, 8.2739716
15: -8.9215736, 0.9301729, -8.9085817, 0.8695917, -4.8498173, 4.9267902
16: -33.5813904, -23.9673271, -33.5178833, -24.0124168, -6.4771957, 6.4310379
17: -88.6759720, -72.3861160, -88.6762543, -72.4902191, -8.1283455, 8.2589836
18: -4.1920924, 1.0740075, -4.1418672, 1.0404994, -3.3993530, 3.3840027
19: -30.5302658, -23.2008743, -30.4963493, -23.2298622, -4.6483650, 4.6449890
20: -11.1738319, -5.1535616, -11.1623154, -5.1659083, -4.9198837, 4.9246883
21: -43.5571709, -35.0488892, -43.5099297, -35.0938339, -4.2696495, 4.2607594
22: -27.0059948, -19.5266075, -26.9935265, -19.5683117, -4.3182144, 4.3555927
23: -20.8682251, -12.5019474, -20.8033562, -12.5394020, -4.7925320, 4.7541008
24: -16.8734665, -7.6346040, -16.8165379, -7.6660080, -7.1652222, 7.1371460
25: -14.6409473, -6.9530144, -14.6052828, -6.9830861, -4.1951714, 4.1814518
26: -14.6209106, -7.7976198, -14.6098289, -7.8246713, -6.5256157, 6.5543709
27: -14.6382122, -9.5233068, -14.6047039, -9.5698261, -4.0526752, 4.0649529
28: -10.0343723, -1.4231758, -10.0020828, -1.4316807, -6.1397743, 6.1354561
29: -45.5896301, -36.8078995, -45.5548401, -36.8731308, -4.9860153, 5.0244503
30: -32.2085609, -22.9965286, -32.1431122, -23.0551357, -4.9977779, 5.0020447
31: -32.2517738, -23.5114975, -32.1863937, -23.5533295, -6.3152962, 6.2796211
32: 7.6976581, 13.6747322, 7.7300634, 13.6790524, -4.1780510, 4.1311836
33: 4.5893106, 16.3115711, 4.6638346, 16.3122196, -6.7285500, 6.6256142
34: 20.5200005, 30.9889946, 20.6054535, 30.9686089, -5.7705841, 5.6938629
35: 16.4825993, 26.8635826, 16.5757103, 26.8462009, -5.4797001, 5.3880577
36: 28.7958012, 35.1248474, 28.8414841, 35.1188698, -3.4562855, 3.4085360
37: 11.0022554, 20.1163216, 11.0777550, 20.1083412, -5.9860535, 5.9024506
38: 34.8398132, 43.6985664, 34.9233932, 43.6579819, -6.0641937, 6.0114822
39: 8.9646778, 18.5169983, 9.0362835, 18.5009842, -6.5580368, 6.4863358
40: 15.7659302, 25.1276417, 15.8295717, 25.1311321, -5.8332558, 5.7512589
41: 6.7120681, 13.2260218, 6.7578306, 13.2230692, -5.0340004, 4.9830589
42: -12.3980570, -3.4538286, -12.3672895, -3.4588690, -7.0571899, 7.0256958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 535

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6310388
time: 5.41 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6374970
time: 5.47 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -21.5742188, -8.4789572, -21.5472527, -8.4845905, -10.4195404, 10.3778992
1: -21.4290352, -12.2320938, -21.4392624, -12.2472076, -5.2396965, 5.2810860
2: -12.3949013, -5.7782068, -12.4016972, -5.7847009, -4.2438583, 4.2662487
3: -12.0095510, -4.1682668, -12.0116940, -4.1860595, -5.3377113, 5.3655090
4: -10.2857018, 0.0084291, -10.2876501, -0.0355653, -6.0001564, 6.0676975
5: -13.5576019, -4.0438442, -13.5624390, -4.0648503, -6.1245155, 6.1554070
6: -8.3326015, 0.5378714, -8.2963448, 0.5430138, -6.4648018, 6.4113731
7: -32.1557770, -22.0512791, -32.1616631, -22.0861092, -5.7998734, 5.8577538
8: -18.8046989, -9.1146021, -18.8056927, -9.1531382, -5.1725121, 5.2240181
9: -5.3412209, 1.3932781, -5.3253736, 1.3817306, -4.0433636, 4.0447311
10: -36.1330109, -27.7551689, -36.1409111, -27.7751656, -5.2217484, 5.2602615
11: -55.1211739, -44.7848816, -55.0997505, -44.8269577, -4.9130859, 4.9465332
12: -11.5786514, -4.5867538, -11.5707512, -4.5929704, -6.2219620, 6.2083359
13: 0.8850516, 8.0086393, 0.8917373, 7.9776225, -5.2605057, 5.2928352
14: -71.0823975, -57.9487801, -71.1005554, -57.9738350, -8.2026711, 8.2733574
15: -8.9086590, 0.9085894, -8.9153423, 0.8720398, -4.8305759, 4.9006786
16: -33.5579758, -23.9757271, -33.5378494, -23.9901485, -6.4563255, 6.4226341
17: -88.6756668, -72.3891602, -88.6872559, -72.4703674, -8.1307907, 8.2511826
18: -4.1684327, 1.0690997, -4.1447325, 1.0453873, -3.3792610, 3.3811607
19: -30.5212402, -23.2038021, -30.5100822, -23.2112160, -4.6364765, 4.6353302
20: -11.1717358, -5.1547604, -11.1684437, -5.1554875, -4.9215240, 4.9186783
21: -43.5403595, -35.0556870, -43.5271759, -35.0672913, -4.2445087, 4.2373829
22: -27.0025520, -19.5272312, -27.0002785, -19.5580597, -4.3165684, 4.3540154
23: -20.8487282, -12.5109901, -20.8103027, -12.5256786, -4.7738743, 4.7396736
24: -16.8535881, -7.6420646, -16.8202896, -7.6563954, -7.1497116, 7.1295509
25: -14.6369104, -6.9556646, -14.6145573, -6.9667397, -4.1910305, 4.1718464
26: -14.6165190, -7.8010273, -14.6119213, -7.8206744, -6.5280228, 6.5368767
27: -14.6268396, -9.5285769, -14.6145611, -9.5544701, -4.0420055, 4.0557346
28: -10.0195856, -1.4316912, -10.0049925, -1.4290664, -6.1431808, 6.1309738
29: -45.5766602, -36.8121262, -45.5676880, -36.8501358, -4.9741154, 5.0115490
30: -32.1762924, -23.0122414, -32.1541061, -23.0340996, -4.9676037, 4.9770336
31: -32.2324371, -23.5158768, -32.2015076, -23.5316963, -6.2907600, 6.2671700
32: 7.7026072, 13.6725559, 7.7264881, 13.6795406, -4.1731110, 4.1323967
33: 4.5920687, 16.3102798, 4.6513810, 16.3154640, -6.7225189, 6.6361008
34: 20.5322151, 30.9840355, 20.5781212, 30.9870186, -5.7496166, 5.6890240
35: 16.4964237, 26.8584499, 16.5533142, 26.8586025, -5.4559383, 5.3840389
36: 28.7978973, 35.1231842, 28.8278065, 35.1274109, -3.4515991, 3.4073563
37: 11.0117073, 20.1127853, 11.0712519, 20.1120987, -5.9772568, 5.9042397
38: 34.8505974, 43.6846619, 34.9025421, 43.6732483, -6.0534668, 6.0023766
39: 8.9739828, 18.5052071, 9.0271893, 18.5018768, -6.5516396, 6.4876595
40: 15.7768478, 25.1227722, 15.8185625, 25.1328964, -5.8227768, 5.7531815
41: 6.7209663, 13.2216759, 6.7548280, 13.2258282, -5.0265846, 4.9807243
42: -12.3980618, -3.4565544, -12.3757114, -3.4460177, -7.0587273, 7.0175476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 759

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6166938
time: 20.95 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6381821
time: 5.41 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5834599, -8.4761534, -21.5471344, -8.4845190, -10.4514236, 10.3754730
1: -21.4307556, -12.2297544, -21.4393120, -12.2474546, -5.2459526, 5.2822533
2: -12.3979378, -5.7753315, -12.4017792, -5.7847204, -4.2531948, 4.2650490
3: -12.0135527, -4.1594543, -12.0118351, -4.1860075, -5.3416824, 5.3743591
4: -10.2963762, 0.0293608, -10.2880344, -0.0355392, -6.0104179, 6.0890656
5: -13.5640945, -4.0365562, -13.5626106, -4.0648704, -6.1341591, 6.1535835
6: -8.3469887, 0.5431714, -8.2963476, 0.5432234, -6.4794731, 6.4168930
7: -32.1567383, -22.0526676, -32.1616669, -22.0878582, -5.8028870, 5.8623219
8: -18.8163052, -9.0788727, -18.8061771, -9.1530991, -5.1838799, 5.2598877
9: -5.3457656, 1.4028986, -5.3255272, 1.3817320, -4.0476894, 4.0543880
10: -36.1385002, -27.7487221, -36.1410561, -27.7751312, -5.2275047, 5.2622414
11: -55.1563911, -44.7713699, -55.0997467, -44.8263397, -4.9490318, 4.9597912
12: -11.5807209, -4.5792556, -11.5707436, -4.5928135, -6.2228584, 6.2151833
13: 0.8790204, 8.0279617, 0.8914741, 7.9777012, -5.2666397, 5.3126755
14: -71.0864716, -57.9418716, -71.1005783, -57.9737930, -8.2162781, 8.2786064
15: -8.9215736, 0.9301729, -8.9157333, 0.8720322, -4.8436050, 4.9226608
16: -33.5813904, -23.9673271, -33.5378799, -23.9897766, -6.4800072, 6.4304047
17: -88.6759720, -72.3861160, -88.6872101, -72.4716644, -8.1309319, 8.2563057
18: -4.1920924, 1.0740075, -4.1447802, 1.0455945, -3.4030800, 3.3859653
19: -30.5302658, -23.2008743, -30.5101433, -23.2111168, -4.6460114, 4.6378670
20: -11.1738319, -5.1535616, -11.1684561, -5.1562519, -4.9212227, 4.9229584
21: -43.5571709, -35.0488892, -43.5272064, -35.0670242, -4.2616215, 4.2441139
22: -27.0059948, -19.5266075, -27.0000877, -19.5580730, -4.3204937, 4.3540230
23: -20.8682251, -12.5019474, -20.8103333, -12.5253696, -4.7937050, 4.7492599
24: -16.8734665, -7.6346040, -16.8203659, -7.6560869, -7.1698723, 7.1369781
25: -14.6409473, -6.9530144, -14.6145878, -6.9666686, -4.1955891, 4.1746502
26: -14.6209106, -7.7976198, -14.6120043, -7.8210144, -6.5282021, 6.5552979
27: -14.6382122, -9.5233068, -14.6146507, -9.5542669, -4.0529861, 4.0610790
28: -10.0343723, -1.4231758, -10.0050306, -1.4288015, -6.1433258, 6.1385460
29: -45.5896301, -36.8078995, -45.5677109, -36.8499374, -4.9871502, 5.0148697
30: -32.2085609, -22.9965286, -32.1540833, -23.0335693, -5.0004807, 4.9930191
31: -32.2517738, -23.5114975, -32.2015991, -23.5315132, -6.3114510, 6.2710419
32: 7.6976581, 13.6747322, 7.7264729, 13.6796198, -4.1782722, 4.1345291
33: 4.5893106, 16.3115711, 4.6521363, 16.3154621, -6.7282104, 6.6373253
34: 20.5200005, 30.9889946, 20.5780334, 30.9872284, -5.7620525, 5.6935482
35: 16.4825993, 26.8635826, 16.5532722, 26.8588238, -5.4699421, 5.3890648
36: 28.7958012, 35.1248474, 28.8277779, 35.1274605, -3.4536257, 3.4088364
37: 11.0022554, 20.1163216, 11.0712109, 20.1122017, -5.9871750, 5.9074554
38: 34.8398132, 43.6985664, 34.9021988, 43.6732407, -6.0645180, 6.0165825
39: 8.9646778, 18.5169983, 9.0270128, 18.5018616, -6.5607491, 6.4970856
40: 15.7659302, 25.1276417, 15.8184690, 25.1329060, -5.8328152, 5.7612476
41: 6.7120681, 13.2260218, 6.7548018, 13.2259884, -5.0355568, 4.9850311
42: -12.3980570, -3.4538286, -12.3754606, -3.4460208, -7.0589027, 7.0233803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 535

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6310392
time: 6.16 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6374973
time: 5.15 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -21.5781021, -8.4788551, -21.5597916, -8.4801655, -10.4577026, 10.4209747
1: -21.4303780, -12.2310791, -21.4397240, -12.2478399, -5.2734051, 5.2948170
2: -12.3956661, -5.7774386, -12.3996239, -5.7845941, -4.2716103, 4.2799911
3: -12.0103178, -4.1644087, -12.0057869, -4.1824436, -5.3561935, 5.3692169
4: -10.2874737, 0.0160040, -10.2942619, -0.0156139, -6.0188789, 6.0682106
5: -13.5587149, -4.0415139, -13.5672293, -4.0600019, -6.1408195, 6.1623726
6: -8.3385096, 0.5390713, -8.3101082, 0.5429580, -6.4738541, 6.4343338
7: -32.1568871, -22.0499954, -32.1573181, -22.0896645, -5.8184586, 5.8599396
8: -18.8056488, -9.1007957, -18.8167706, -9.1170368, -5.1959267, 5.2284050
9: -5.3422594, 1.3966579, -5.3257742, 1.3904247, -4.0506001, 4.0364113
10: -36.1350098, -27.7537746, -36.1386337, -27.7838020, -5.2442417, 5.2710209
11: -55.1323166, -44.7844963, -55.1101227, -44.8452148, -4.9193554, 4.9692593
12: -11.5809269, -4.5859241, -11.5703716, -4.5941415, -6.2223053, 6.2071190
13: 0.8842248, 8.0159569, 0.8907263, 7.9970803, -5.2845001, 5.3003578
14: -71.0843048, -57.9474068, -71.1015854, -57.9732132, -8.2432976, 8.2847862
15: -8.9112701, 0.9151077, -8.9239321, 0.8899441, -4.8593502, 4.9112091
16: -33.5677299, -23.9756050, -33.5449524, -23.9992523, -6.4591255, 6.4416847
17: -88.6775436, -72.3879013, -88.6786804, -72.4798584, -8.1563873, 8.2575188
18: -4.1777048, 1.0691819, -4.1682329, 1.0486946, -3.3811359, 3.3924999
19: -30.5249138, -23.2034569, -30.5081024, -23.2245827, -4.6409168, 4.6454773
20: -11.1720009, -5.1537991, -11.1633759, -5.1617651, -4.9310760, 4.9344444
21: -43.5461426, -35.0551033, -43.5273590, -35.0838242, -4.2539158, 4.2581692
22: -27.0050240, -19.5271416, -27.0010319, -19.5653801, -4.3204708, 4.3578873
23: -20.8561096, -12.5095081, -20.8263779, -12.5247307, -4.7801113, 4.7609158
24: -16.8623276, -7.6413236, -16.8447666, -7.6508155, -7.1589432, 7.1526642
25: -14.6402349, -6.9547219, -14.6165562, -6.9755793, -4.1950932, 4.1908855
26: -14.6176291, -7.8003669, -14.6120491, -7.8191733, -6.5473480, 6.5659180
27: -14.6305561, -9.5279999, -14.6179600, -9.5618801, -4.0473976, 4.0678482
28: -10.0230169, -1.4303138, -10.0125380, -1.4212809, -6.1543312, 6.1434517
29: -45.5838127, -36.8120422, -45.5752563, -36.8626251, -4.9867630, 5.0314274
30: -32.1860123, -23.0103569, -32.1729736, -23.0353622, -4.9708881, 4.9942017
31: -32.2408981, -23.5151539, -32.2137413, -23.5430832, -6.3092079, 6.3051605
32: 7.7008858, 13.6733494, 7.7240925, 13.6820860, -4.1810837, 4.1425266
33: 4.5882063, 16.3115215, 4.6498909, 16.3181648, -6.7265797, 6.6435928
34: 20.5269756, 30.9848976, 20.5885220, 30.9773979, -5.7601929, 5.7004700
35: 16.4906235, 26.8594131, 16.5564537, 26.8561287, -5.4677563, 5.3980579
36: 28.7973709, 35.1240768, 28.8381157, 35.1215668, -3.4561882, 3.4166403
37: 11.0066671, 20.1137829, 11.0604420, 20.1154709, -5.9812317, 5.9179115
38: 34.8495407, 43.6899261, 34.9104767, 43.6726837, -6.0579567, 6.0123940
39: 8.9733000, 18.5077343, 9.0265255, 18.5063515, -6.5509720, 6.4947433
40: 15.7735825, 25.1259422, 15.8149452, 25.1399746, -5.8337917, 5.7700043
41: 6.7167492, 13.2225609, 6.7434626, 13.2307577, -5.0327339, 4.9932671
42: -12.3983374, -3.4552188, -12.3685722, -3.4556129, -7.0600853, 7.0284691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 759

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6319993
time: 4.95 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6385325
time: 5.14 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5873756, -8.4760866, -21.5596504, -8.4801149, -10.4895859, 10.4185486
1: -21.4321213, -12.2287636, -21.4397736, -12.2481213, -5.2796612, 5.2959824
2: -12.3986912, -5.7745886, -12.3997192, -5.7845893, -4.2809448, 4.2787838
3: -12.0143270, -4.1555634, -12.0059147, -4.1824045, -5.3601570, 5.3780670
4: -10.2981749, 0.0369308, -10.2946529, -0.0156294, -6.0291405, 6.0895710
5: -13.5651894, -4.0341926, -13.5674000, -4.0599627, -6.1505165, 6.1605453
6: -8.3528681, 0.5443628, -8.3101215, 0.5431615, -6.4885330, 6.4398270
7: -32.1578331, -22.0514069, -32.1573181, -22.0913849, -5.8214874, 5.8644676
8: -18.8172703, -9.0650339, -18.8172569, -9.1170101, -5.2072697, 5.2643242
9: -5.3468108, 1.4063028, -5.3259315, 1.3904459, -4.0549297, 4.0460720
10: -36.1405067, -27.7473030, -36.1387558, -27.7837486, -5.2500629, 5.2730026
11: -55.1675415, -44.7709808, -55.1101189, -44.8446274, -4.9553070, 4.9825153
12: -11.5829935, -4.5783935, -11.5703459, -4.5939531, -6.2232132, 6.2139893
13: 0.8782057, 8.0353088, 0.8904927, 7.9971552, -5.2906265, 5.3201981
14: -71.0884247, -57.9405632, -71.1016388, -57.9731903, -8.2568665, 8.2900124
15: -8.9242020, 0.9366841, -8.9243507, 0.8899465, -4.8723392, 4.9331818
16: -33.5911331, -23.9671917, -33.5449715, -23.9988899, -6.4828339, 6.4494553
17: -88.6778107, -72.3848724, -88.6786423, -72.4811020, -8.1565361, 8.2626495
18: -4.2013907, 1.0740848, -4.1682625, 1.0489109, -3.4050064, 3.3973026
19: -30.5339222, -23.2005119, -30.5081367, -23.2244987, -4.6504669, 4.6480198
20: -11.1740999, -5.1526041, -11.1634026, -5.1625233, -4.9307709, 4.9387379
21: -43.5629501, -35.0483246, -43.5273666, -35.0835419, -4.2710285, 4.2648945
22: -27.0084991, -19.5265179, -27.0008507, -19.5653915, -4.3243904, 4.3579025
23: -20.8755817, -12.5005035, -20.8264370, -12.5244303, -4.7999573, 4.7704868
24: -16.8822098, -7.6338596, -16.8448277, -7.6505036, -7.1790466, 7.1601295
25: -14.6443043, -6.9520712, -14.6165924, -6.9755177, -4.1996536, 4.1936989
26: -14.6220322, -7.7969537, -14.6121531, -7.8195095, -6.5475121, 6.5843124
27: -14.6419373, -9.5227213, -14.6180458, -9.5616798, -4.0583858, 4.0732098
28: -10.0377865, -1.4218202, -10.0125895, -1.4209726, -6.1544609, 6.1510544
29: -45.5967712, -36.8078270, -45.5752716, -36.8624344, -4.9997997, 5.0347710
30: -32.2183113, -22.9946671, -32.1729889, -23.0347958, -5.0037556, 5.0101547
31: -32.2602692, -23.5107574, -32.2138252, -23.5428696, -6.3299561, 6.3090591
32: 7.6959295, 13.6755581, 7.7240915, 13.6821384, -4.1862411, 4.1446590
33: 4.5854526, 16.3128586, 4.6506004, 16.3181877, -6.7322235, 6.6448326
34: 20.5147591, 30.9898796, 20.5885162, 30.9775848, -5.7726288, 5.7050056
35: 16.4767723, 26.8645210, 16.5563622, 26.8563557, -5.4817657, 5.4030838
36: 28.7952785, 35.1257210, 28.8380699, 35.1216011, -3.4582214, 3.4181509
37: 10.9971600, 20.1173115, 11.0604076, 20.1155720, -5.9911652, 5.9211311
38: 34.8387947, 43.7038383, 34.9100914, 43.6726913, -6.0689659, 6.0265961
39: 8.9639616, 18.5195007, 9.0263414, 18.5063477, -6.5600929, 6.5041656
40: 15.7626715, 25.1308022, 15.8148499, 25.1400032, -5.8438148, 5.7780819
41: 6.7078686, 13.2268858, 6.7434478, 13.2309237, -5.0417061, 4.9975700
42: -12.3983316, -3.4525018, -12.3683290, -3.4556012, -7.0602379, 7.0343018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1599

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5899425, upper bound: 3.6382337
time: 5.30 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5973362, upper bound: 3.6382337
time: 6.68 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -21.5781021, -8.4788551, -21.5642891, -8.4794130, -10.4578705, 10.4247208
1: -21.4303780, -12.2310791, -21.4430428, -12.2443390, -5.2749100, 5.2962475
2: -12.3956661, -5.7774386, -12.4050379, -5.7826242, -4.2674522, 4.2793102
3: -12.0103178, -4.1644087, -12.0183420, -4.1730127, -5.3531342, 5.3700104
4: -10.2874737, 0.0160040, -10.3031998, -0.0124934, -6.0162086, 6.0700283
5: -13.5587149, -4.0415139, -13.5694895, -4.0567741, -6.1433983, 6.1642036
6: -8.3385096, 0.5390713, -8.3155251, 0.5523276, -6.4765091, 6.4334068
7: -32.1568871, -22.0499954, -32.1635742, -22.0814075, -5.8226814, 5.8633881
8: -18.8056488, -9.1007957, -18.8246765, -9.1115360, -5.1958523, 5.2296753
9: -5.3422594, 1.3966579, -5.3336234, 1.3928273, -4.0550995, 4.0476074
10: -36.1350098, -27.7537746, -36.1476402, -27.7700405, -5.2442818, 5.2675381
11: -55.1323166, -44.7844963, -55.1294785, -44.8096886, -4.9171982, 4.9502144
12: -11.5809269, -4.5859241, -11.5775881, -4.5848761, -6.2250366, 6.2094994
13: 0.8842248, 8.0159569, 0.8810055, 8.0016718, -5.2841034, 5.3038445
14: -71.0843048, -57.9474068, -71.1043396, -57.9705086, -8.2482643, 8.2894058
15: -8.9112701, 0.9151077, -8.9311104, 0.8923228, -4.8531170, 4.9070778
16: -33.5677299, -23.9756050, -33.5649643, -23.9766350, -6.4619484, 6.4410362
17: -88.6775436, -72.3879013, -88.6896591, -72.4612732, -8.1589737, 8.2548409
18: -4.1777048, 1.0691819, -4.1711783, 1.0537989, -3.3848629, 3.3944874
19: -30.5249138, -23.2034569, -30.5218792, -23.2058372, -4.6385880, 4.6383457
20: -11.1720009, -5.1537991, -11.1695089, -5.1520786, -4.9324036, 4.9327164
21: -43.5461426, -35.0551033, -43.5446701, -35.0570068, -4.2458916, 4.2415257
22: -27.0050240, -19.5271416, -27.0076141, -19.5551376, -4.3227501, 4.3563271
23: -20.8561096, -12.5095081, -20.8333740, -12.5107498, -4.7812767, 4.7560730
24: -16.8623276, -7.6413236, -16.8485680, -7.6409149, -7.1635971, 7.1525116
25: -14.6402349, -6.9547219, -14.6258450, -6.9591579, -4.1955051, 4.1840782
26: -14.6176291, -7.8003669, -14.6142292, -7.8155012, -6.5499306, 6.5668297
27: -14.6305561, -9.5279999, -14.6278954, -9.5463266, -4.0477161, 4.0639820
28: -10.0230169, -1.4303138, -10.0154963, -1.4183484, -6.1578941, 6.1465454
29: -45.5838127, -36.8120422, -45.5880966, -36.8394699, -4.9879169, 5.0218525
30: -32.1860123, -23.0103569, -32.1839371, -23.0137844, -4.9735680, 4.9851665
31: -32.2408981, -23.5151539, -32.2289772, -23.5212555, -6.3053780, 6.2965965
32: 7.7008858, 13.6733494, 7.7205048, 13.6826229, -4.1813049, 4.1458778
33: 4.5882063, 16.3115215, 4.6381955, 16.3214016, -6.7262402, 6.6553040
34: 20.5269756, 30.9848976, 20.5611172, 30.9959602, -5.7516518, 5.7001724
35: 16.4906235, 26.8594131, 16.5339947, 26.8687592, -5.4580002, 5.3990593
36: 28.7973709, 35.1240768, 28.8244057, 35.1301422, -3.4535351, 3.4169540
37: 11.0066671, 20.1137829, 11.0539303, 20.1193447, -5.9823494, 5.9229202
38: 34.8495407, 43.6899261, 34.8892670, 43.6879387, -6.0582771, 6.0175056
39: 8.9733000, 18.5077343, 9.0172281, 18.5072060, -6.5536957, 6.5054550
40: 15.7735825, 25.1259422, 15.8038521, 25.1417732, -5.8333359, 5.7799931
41: 6.7167492, 13.2225609, 6.7404404, 13.2336807, -5.0342827, 4.9952278
42: -12.3983374, -3.4552188, -12.3767891, -3.4427600, -7.0618057, 7.0261383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 759

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5856645, upper bound: 3.6189817
time: 5.56 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5856645, upper bound: 3.6255130
time: 5.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5873756, -8.4760866, -21.5641575, -8.4793348, -10.4897537, 10.4223022
1: -21.4321213, -12.2287636, -21.4430714, -12.2446251, -5.2811775, 5.2974091
2: -12.3986912, -5.7745886, -12.4051352, -5.7826362, -4.2767792, 4.2780991
3: -12.0143270, -4.1555634, -12.0184546, -4.1729908, -5.3571014, 5.3788605
4: -10.2981749, 0.0369308, -10.3035851, -0.0125008, -6.0264587, 6.0913734
5: -13.5651894, -4.0341926, -13.5696821, -4.0567613, -6.1530838, 6.1623802
6: -8.3528681, 0.5443628, -8.3155527, 0.5525755, -6.4911842, 6.4388962
7: -32.1578331, -22.0514069, -32.1635742, -22.0831909, -5.8257141, 5.8679237
8: -18.8172703, -9.0650339, -18.8251457, -9.1114960, -5.2071800, 5.2655621
9: -5.3468108, 1.4063028, -5.3338003, 1.3928545, -4.0594254, 4.0572624
10: -36.1405067, -27.7473030, -36.1477547, -27.7699986, -5.2500858, 5.2695217
11: -55.1675415, -44.7709808, -55.1294899, -44.8091049, -4.9531479, 4.9634666
12: -11.5829935, -4.5783935, -11.5775766, -4.5846987, -6.2259254, 6.2163582
13: 0.8782057, 8.0353088, 0.8807570, 8.0017500, -5.2902603, 5.3237038
14: -71.0884247, -57.9405632, -71.1044159, -57.9704742, -8.2618561, 8.2946472
15: -8.9242020, 0.9366841, -8.9315014, 0.8923173, -4.8661423, 4.9290581
16: -33.5911331, -23.9671917, -33.5649910, -23.9762421, -6.4856491, 6.4488029
17: -88.6778107, -72.3848724, -88.6895981, -72.4625092, -8.1591187, 8.2599754
18: -4.2013907, 1.0740848, -4.1712170, 1.0540221, -3.4087334, 3.3992844
19: -30.5339222, -23.2005119, -30.5219002, -23.2057381, -4.6481209, 4.6408844
20: -11.1740999, -5.1526041, -11.1695414, -5.1528502, -4.9321022, 4.9370060
21: -43.5629501, -35.0483246, -43.5446739, -35.0567322, -4.2629967, 4.2482548
22: -27.0084991, -19.5265179, -27.0074520, -19.5551491, -4.3266640, 4.3563309
23: -20.8755817, -12.5005035, -20.8333931, -12.5104465, -4.8011189, 4.7656574
24: -16.8822098, -7.6338596, -16.8486404, -7.6405964, -7.1836967, 7.1599693
25: -14.6443043, -6.9520712, -14.6259174, -6.9590917, -4.2000675, 4.1868877
26: -14.6220322, -7.7969537, -14.6143284, -7.8158598, -6.5500984, 6.5852165
27: -14.6419373, -9.5227213, -14.6279774, -9.5461149, -4.0587044, 4.0693245
28: -10.0377865, -1.4218202, -10.0155163, -1.4180576, -6.1580200, 6.1541367
29: -45.5967712, -36.8078270, -45.5881233, -36.8392525, -5.0009480, 5.0251846
30: -32.2183113, -22.9946671, -32.1839638, -23.0132618, -5.0064354, 5.0011482
31: -32.2602692, -23.5107574, -32.2290802, -23.5210590, -6.3260841, 6.3004684
32: 7.6959295, 13.6755581, 7.7205200, 13.6827049, -4.1864700, 4.1480103
33: 4.5854526, 16.3128586, 4.6389122, 16.3214397, -6.7318859, 6.6565285
34: 20.5147591, 30.9898796, 20.5610905, 30.9961567, -5.7640953, 5.7046986
35: 16.4767723, 26.8645210, 16.5339241, 26.8689785, -5.4720039, 5.4040833
36: 28.7952785, 35.1257210, 28.8243542, 35.1301842, -3.4555664, 3.4184465
37: 10.9971600, 20.1173115, 11.0538816, 20.1194649, -5.9922829, 5.9261208
38: 34.8387947, 43.7038383, 34.8889122, 43.6879387, -6.0692978, 6.0317078
39: 8.9639616, 18.5195007, 9.0170345, 18.5071869, -6.5628128, 6.5149002
40: 15.7626715, 25.1308022, 15.8037672, 25.1417904, -5.8433666, 5.7880535
41: 6.7078686, 13.2268858, 6.7404118, 13.2338543, -5.0432701, 4.9995384
42: -12.3983316, -3.4525018, -12.3765306, -3.4427500, -7.0620003, 7.0319786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1599

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5899425, upper bound: 3.6382340
time: 4.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5973362, upper bound: 3.6382340
time: 6.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5579834, -8.4811563, -21.5866375, -8.4798555, -10.3944473, 10.4406815
1: -21.4219856, -12.2365484, -21.4367161, -12.2400799, -5.2671089, 5.2965565
2: -12.3915615, -5.7777085, -12.4002781, -5.7798905, -4.2575302, 4.2779598
3: -12.0071468, -4.1688685, -12.0053282, -4.1767206, -5.3632851, 5.3636818
4: -10.2769680, 0.0037479, -10.2933521, 0.0018618, -6.0452003, 6.0449867
5: -13.5527449, -4.0461941, -13.5695934, -4.0483322, -6.1295547, 6.1530762
6: -8.3162031, 0.5253098, -8.3123360, 0.5355163, -6.4598999, 6.4470940
7: -32.1505814, -22.0557404, -32.1593971, -22.0632896, -5.8191528, 5.8398857
8: -18.8011246, -9.1073551, -18.8150978, -9.1114178, -5.2254753, 5.2179890
9: -5.3288035, 1.3930535, -5.3376207, 1.3917122, -4.0426426, 4.0439854
10: -36.1260567, -27.7660561, -36.1334381, -27.7781754, -5.2466087, 5.2689114
11: -55.1253281, -44.8033066, -55.1075478, -44.8250504, -4.9233170, 4.9625759
12: -11.5745134, -4.5910568, -11.5676098, -4.5853772, -6.2390480, 6.2317429
13: 0.8940356, 8.0107975, 0.8908266, 8.0091162, -5.2958755, 5.2946053
14: -71.0678329, -57.9613991, -71.0922394, -57.9608994, -8.2258263, 8.2622414
15: -8.8990755, 0.9011693, -8.9171963, 0.8995957, -4.8720646, 4.8802433
16: -33.5529327, -23.9759464, -33.5507965, -23.9903221, -6.4392395, 6.4633560
17: -88.6560822, -72.4222717, -88.6680527, -72.4369736, -8.1558113, 8.1835899
18: -4.1735067, 1.0583792, -4.1754084, 1.0584340, -3.3832893, 3.3977737
19: -30.5235023, -23.2042046, -30.5126076, -23.2201805, -4.6450977, 4.6561432
20: -11.1709690, -5.1588526, -11.1662693, -5.1600118, -4.9374580, 4.9264946
21: -43.5443840, -35.0580826, -43.5304108, -35.0790253, -4.2573280, 4.2793140
22: -26.9977493, -19.5396614, -27.0006180, -19.5497398, -4.3204441, 4.3376369
23: -20.8500271, -12.5152779, -20.8453388, -12.5215569, -4.7652855, 4.7814445
24: -16.8572483, -7.6482162, -16.8570442, -7.6501842, -7.1500015, 7.1643410
25: -14.6360760, -6.9600935, -14.6289177, -6.9744134, -4.1867046, 4.2007160
26: -14.6093721, -7.8215179, -14.6106033, -7.8176260, -6.5412064, 6.5258293
27: -14.6275578, -9.5355358, -14.6214523, -9.5463037, -4.0523376, 4.0622959
28: -10.0212345, -1.4322876, -10.0202475, -1.4219553, -6.1648178, 6.1421738
29: -45.5753860, -36.8263397, -45.5734901, -36.8451042, -4.9889946, 5.0175915
30: -32.1823769, -23.0205536, -32.1729431, -23.0258293, -4.9757614, 5.0093613
31: -32.2330093, -23.5217361, -32.2220230, -23.5392361, -6.2980385, 6.3062935
32: 7.7196803, 13.6643209, 7.7226677, 13.6795893, -4.1592674, 4.1410313
33: 4.6117835, 16.3025074, 4.6213493, 16.3158302, -6.6832409, 6.6604881
34: 20.5370865, 30.9774494, 20.5626183, 30.9785671, -5.7343197, 5.7182541
35: 16.5063782, 26.8521919, 16.5262909, 26.8549862, -5.4294586, 5.4179401
36: 28.8109665, 35.1159744, 28.8226986, 35.1170006, -3.4273510, 3.4169464
37: 11.0274429, 20.1061268, 11.0314102, 20.1141968, -5.9483185, 5.9416656
38: 34.8639641, 43.6769333, 34.8763123, 43.6722984, -6.0353508, 6.0161972
39: 8.9996548, 18.4968891, 8.9996185, 18.5042343, -6.5092545, 6.4996719
40: 15.7941332, 25.1175671, 15.8022671, 25.1368313, -5.8099785, 5.7767277
41: 6.7347732, 13.2131405, 6.7359819, 13.2243853, -5.0113525, 5.0020409
42: -12.3857851, -3.4645004, -12.3782864, -3.4619930, -7.0510216, 7.0387421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6338233
time: 6.22 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6338238
time: 4.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5781021, -8.4788551, -21.5866375, -8.4798555, -10.4153748, 10.4495544
1: -21.4303780, -12.2310791, -21.4367161, -12.2400799, -5.2772560, 5.3008156
2: -12.3956661, -5.7774386, -12.4002781, -5.7798905, -4.2610512, 4.2802734
3: -12.0103178, -4.1644087, -12.0053282, -4.1767206, -5.3661003, 5.3677216
4: -10.2874737, 0.0160040, -10.2933521, 0.0018618, -6.0556908, 6.0591125
5: -13.5587149, -4.0415139, -13.5695934, -4.0483322, -6.1367378, 6.1605682
6: -8.3385096, 0.5390713, -8.3123360, 0.5355163, -6.4818459, 6.4602242
7: -32.1568871, -22.0499954, -32.1593971, -22.0632896, -5.8271675, 5.8458824
8: -18.8056488, -9.1007957, -18.8150978, -9.1114178, -5.2314568, 5.2260380
9: -5.3422594, 1.3966579, -5.3376207, 1.3917122, -4.0566120, 4.0479259
10: -36.1350098, -27.7537746, -36.1334381, -27.7781754, -5.2557831, 5.2769070
11: -55.1323166, -44.7844963, -55.1075478, -44.8250504, -4.9315739, 4.9809742
12: -11.5809269, -4.5859241, -11.5676098, -4.5853772, -6.2417374, 6.2329330
13: 0.8842248, 8.0159569, 0.8908266, 8.0091162, -5.3058929, 5.3011017
14: -71.0843048, -57.9474068, -71.0922394, -57.9608994, -8.2421837, 8.2739296
15: -8.9112701, 0.9151077, -8.9171963, 0.8995957, -4.8844929, 4.8976402
16: -33.5677299, -23.9756050, -33.5507965, -23.9903221, -6.4561882, 6.4667854
17: -88.6775436, -72.3879013, -88.6680527, -72.4369736, -8.1793480, 8.2195625
18: -4.1777048, 1.0691819, -4.1754084, 1.0584340, -3.3879910, 3.4084969
19: -30.5249138, -23.2034569, -30.5126076, -23.2201805, -4.6474934, 4.6569958
20: -11.1720009, -5.1537991, -11.1662693, -5.1600118, -4.9390907, 4.9313164
21: -43.5461426, -35.0551033, -43.5304108, -35.0790253, -4.2600899, 4.2818546
22: -27.0050240, -19.5271416, -27.0006180, -19.5497398, -4.3285751, 4.3513203
23: -20.8561096, -12.5095081, -20.8453388, -12.5215569, -4.7739220, 4.7887802
24: -16.8623276, -7.6413236, -16.8570442, -7.6501842, -7.1548462, 7.1711273
25: -14.6402349, -6.9547219, -14.6289177, -6.9744134, -4.1921177, 4.2069721
26: -14.6176291, -7.8003669, -14.6106033, -7.8176260, -6.5501556, 6.5464973
27: -14.6305561, -9.5279999, -14.6214523, -9.5463037, -4.0558472, 4.0692673
28: -10.0230169, -1.4303138, -10.0202475, -1.4219553, -6.1685600, 6.1439133
29: -45.5838127, -36.8120422, -45.5734901, -36.8451042, -4.9987450, 5.0315590
30: -32.1860123, -23.0103569, -32.1729431, -23.0258293, -4.9795303, 5.0193176
31: -32.2408981, -23.5151539, -32.2220230, -23.5392361, -6.3085632, 6.3148727
32: 7.7008858, 13.6733494, 7.7226677, 13.6795893, -4.1785812, 4.1503983
33: 4.5882063, 16.3115215, 4.6213493, 16.3158302, -6.7119274, 6.6706810
34: 20.5269756, 30.9848976, 20.5626183, 30.9785671, -5.7460995, 5.7268848
35: 16.4906235, 26.8594131, 16.5262909, 26.8549862, -5.4478741, 5.4265442
36: 28.7973709, 35.1240768, 28.8226986, 35.1170006, -3.4414577, 3.4258747
37: 11.0066671, 20.1137829, 11.0314102, 20.1141968, -5.9716873, 5.9496040
38: 34.8495407, 43.6899261, 34.8763123, 43.6722984, -6.0497437, 6.0304375
39: 8.9733000, 18.5077343, 8.9996185, 18.5042343, -6.5362206, 6.5116081
40: 15.7735825, 25.1259422, 15.8022671, 25.1368313, -5.8317986, 5.7850761
41: 6.7167492, 13.2225609, 6.7359819, 13.2243853, -5.0303307, 5.0118523
42: -12.3983374, -3.4552188, -12.3782864, -3.4619930, -7.0628319, 7.0480537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6385279
time: 5.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6385281
time: 4.98 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5579834, -8.4811563, -21.5911064, -8.4790535, -10.3946228, 10.4444504
1: -21.4219856, -12.2365484, -21.4400368, -12.2365980, -5.2686119, 5.2979870
2: -12.3915615, -5.7777085, -12.4056711, -5.7779036, -4.2533779, 4.2772732
3: -12.0071468, -4.1688685, -12.0178432, -4.1672831, -5.3602409, 5.3644485
4: -10.2769680, 0.0037479, -10.3023214, 0.0049579, -6.0425415, 6.0467987
5: -13.5527449, -4.0461941, -13.5718594, -4.0451136, -6.1321411, 6.1549110
6: -8.3162031, 0.5253098, -8.3177443, 0.5448874, -6.4625702, 6.4461784
7: -32.1505814, -22.0557404, -32.1656609, -22.0550652, -5.8233948, 5.8433475
8: -18.8011246, -9.1073551, -18.8230228, -9.1059046, -5.2254257, 5.2192650
9: -5.3288035, 1.3930535, -5.3454633, 1.3941084, -4.0471039, 4.0552292
10: -36.1260567, -27.7660561, -36.1424294, -27.7644253, -5.2466450, 5.2654457
11: -55.1253281, -44.8033066, -55.1269379, -44.7895279, -4.9211597, 4.9435272
12: -11.5745134, -4.5910568, -11.5748425, -4.5761237, -6.2417603, 6.2341003
13: 0.8940356, 8.0107975, 0.8810621, 8.0137148, -5.2955017, 5.2980995
14: -71.0678329, -57.9613991, -71.0950394, -57.9582062, -8.2307892, 8.2668648
15: -8.8990755, 0.9011693, -8.9243431, 0.9019961, -4.8658276, 4.8761311
16: -33.5529327, -23.9759464, -33.5707893, -23.9677048, -6.4419212, 6.4627342
17: -88.6560822, -72.4222717, -88.6790314, -72.4183960, -8.1583900, 8.1809235
18: -4.1735067, 1.0583792, -4.1783781, 1.0635581, -3.3870316, 3.3997822
19: -30.5235023, -23.2042046, -30.5263824, -23.2014275, -4.6427574, 4.6490192
20: -11.1709690, -5.1588526, -11.1724176, -5.1503448, -4.9387894, 4.9247780
21: -43.5443840, -35.0580826, -43.5476990, -35.0521812, -4.2492981, 4.2626724
22: -26.9977493, -19.5396614, -27.0072136, -19.5395069, -4.3227291, 4.3360901
23: -20.8500271, -12.5152779, -20.8523273, -12.5075626, -4.7664680, 4.7766323
24: -16.8572483, -7.6482162, -16.8608551, -7.6403050, -7.1546631, 7.1642189
25: -14.6360760, -6.9600935, -14.6382084, -6.9580107, -4.1871300, 4.1939182
26: -14.6093721, -7.8215179, -14.6128016, -7.8139744, -6.5437737, 6.5267296
27: -14.6275578, -9.5355358, -14.6313915, -9.5307617, -4.0526752, 4.0584354
28: -10.0212345, -1.4322876, -10.0231848, -1.4190676, -6.1683693, 6.1452637
29: -45.5753860, -36.8263397, -45.5863647, -36.8219452, -4.9901543, 5.0080166
30: -32.1823769, -23.0205536, -32.1839027, -23.0042839, -4.9784565, 5.0003490
31: -32.2330093, -23.5217361, -32.2372932, -23.5173759, -6.2942314, 6.2977448
32: 7.7196803, 13.6643209, 7.7191091, 13.6801577, -4.1594906, 4.1443863
33: 4.6117835, 16.3025074, 4.6096745, 16.3190517, -6.6829758, 6.6721725
34: 20.5370865, 30.9774494, 20.5351658, 30.9971848, -5.7257881, 5.7179451
35: 16.5063782, 26.8521919, 16.5038719, 26.8676186, -5.4197578, 5.4189358
36: 28.8109665, 35.1159744, 28.8089943, 35.1255951, -3.4247217, 3.4172535
37: 11.0274429, 20.1061268, 11.0248880, 20.1180763, -5.9494057, 5.9466667
38: 34.8639641, 43.6769333, 34.8551369, 43.6875420, -6.0356750, 6.0212479
39: 8.9996548, 18.4968891, 8.9903088, 18.5050831, -6.5120125, 6.5103951
40: 15.7941332, 25.1175671, 15.7911978, 25.1385899, -5.8095398, 5.7867279
41: 6.7347732, 13.2131405, 6.7329597, 13.2273169, -5.0129051, 5.0040092
42: -12.3857851, -3.4645004, -12.3864899, -3.4491448, -7.0527573, 7.0364342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6338239
time: 4.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6338239
time: 5.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5781021, -8.4788551, -21.5911064, -8.4790535, -10.4155426, 10.4533234
1: -21.4303780, -12.2310791, -21.4400368, -12.2365980, -5.2787628, 5.3022480
2: -12.3956661, -5.7774386, -12.4056711, -5.7779036, -4.2568989, 4.2795925
3: -12.0103178, -4.1644087, -12.0178432, -4.1672831, -5.3630524, 5.3685074
4: -10.2874737, 0.0160040, -10.3023214, 0.0049579, -6.0530167, 6.0609322
5: -13.5587149, -4.0415139, -13.5718594, -4.0451136, -6.1393242, 6.1624031
6: -8.3385096, 0.5390713, -8.3177443, 0.5448874, -6.4845047, 6.4592934
7: -32.1568871, -22.0499954, -32.1656609, -22.0550652, -5.8314095, 5.8493404
8: -18.8056488, -9.1007957, -18.8230228, -9.1059046, -5.2313995, 5.2273102
9: -5.3422594, 1.3966579, -5.3454633, 1.3941084, -4.0611076, 4.0591698
10: -36.1350098, -27.7537746, -36.1424294, -27.7644253, -5.2558231, 5.2734241
11: -55.1323166, -44.7844963, -55.1269379, -44.7895279, -4.9294147, 4.9619236
12: -11.5809269, -4.5859241, -11.5748425, -4.5761237, -6.2444534, 6.2352943
13: 0.8842248, 8.0159569, 0.8810621, 8.0137148, -5.3055077, 5.3045883
14: -71.0843048, -57.9474068, -71.0950394, -57.9582062, -8.2471542, 8.2785416
15: -8.9112701, 0.9151077, -8.9243431, 0.9019961, -4.8782558, 4.8935242
16: -33.5677299, -23.9756050, -33.5707893, -23.9677048, -6.4589996, 6.4661636
17: -88.6775436, -72.3879013, -88.6790314, -72.4183960, -8.1819267, 8.2168732
18: -4.1777048, 1.0691819, -4.1783781, 1.0635581, -3.3917332, 3.4105053
19: -30.5249138, -23.2034569, -30.5263824, -23.2014275, -4.6451569, 4.6498642
20: -11.1720009, -5.1537991, -11.1724176, -5.1503448, -4.9404182, 4.9295692
21: -43.5461426, -35.0551033, -43.5476990, -35.0521812, -4.2520599, 4.2652092
22: -27.0050240, -19.5271416, -27.0072136, -19.5395069, -4.3308620, 4.3497620
23: -20.8561096, -12.5095081, -20.8523273, -12.5075626, -4.7751045, 4.7839451
24: -16.8623276, -7.6413236, -16.8608551, -7.6403050, -7.1595001, 7.1709709
25: -14.6402349, -6.9547219, -14.6382084, -6.9580107, -4.1925430, 4.2001667
26: -14.6176291, -7.8003669, -14.6128016, -7.8139744, -6.5527306, 6.5473976
27: -14.6305561, -9.5279999, -14.6313915, -9.5307617, -4.0561810, 4.0653915
28: -10.0230169, -1.4303138, -10.0231848, -1.4190676, -6.1721153, 6.1469955
29: -45.5838127, -36.8120422, -45.5863647, -36.8219452, -4.9999008, 5.0219727
30: -32.1860123, -23.0103569, -32.1839027, -23.0042839, -4.9822216, 5.0102863
31: -32.2408981, -23.5151539, -32.2372932, -23.5173759, -6.3047523, 6.3063049
32: 7.7008858, 13.6733494, 7.7191091, 13.6801577, -4.1788044, 4.1537514
33: 4.5882063, 16.3115215, 4.6096745, 16.3190517, -6.7115822, 6.6823654
34: 20.5269756, 30.9848976, 20.5351658, 30.9971848, -5.7375717, 5.7265892
35: 16.4906235, 26.8594131, 16.5038719, 26.8676186, -5.4381161, 5.4275417
36: 28.7973709, 35.1240768, 28.8089943, 35.1255951, -3.4388094, 3.4261799
37: 11.0066671, 20.1137829, 11.0248880, 20.1180763, -5.9728088, 5.9545937
38: 34.8495407, 43.6899261, 34.8551369, 43.6875420, -6.0500717, 6.0355530
39: 8.9733000, 18.5077343, 8.9903088, 18.5050831, -6.5389481, 6.5223389
40: 15.7735825, 25.1259422, 15.7911978, 25.1385899, -5.8313446, 5.7950783
41: 6.7167492, 13.2225609, 6.7329597, 13.2273169, -5.0318947, 5.0138168
42: -12.3983374, -3.4552188, -12.3864899, -3.4491448, -7.0645599, 7.0457497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6385283
time: 4.81 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6385287
time: 5.18 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5672455, -8.4783888, -21.5865288, -8.4797583, -10.4263382, 10.4382553
1: -21.4237137, -12.2342529, -21.4367561, -12.2403631, -5.2733631, 5.2976971
2: -12.3945704, -5.7748389, -12.4003696, -5.7799087, -4.2668610, 4.2767563
3: -12.0111580, -4.1600070, -12.0054617, -4.1766753, -5.3672409, 5.3725357
4: -10.2876625, 0.0246730, -10.2937164, 0.0018325, -6.0554657, 6.0663376
5: -13.5592213, -4.0388803, -13.5697784, -4.0483270, -6.1391907, 6.1512718
6: -8.3306007, 0.5306103, -8.3123312, 0.5357430, -6.4745789, 6.4526100
7: -32.1515274, -22.0571251, -32.1593971, -22.0650387, -5.8221664, 5.8444252
8: -18.8127232, -9.0715923, -18.8156147, -9.1114292, -5.2368336, 5.2539215
9: -5.3333635, 1.4026828, -5.3377705, 1.3917158, -4.0469780, 4.0536385
10: -36.1315498, -27.7596016, -36.1335678, -27.7781715, -5.2523422, 5.2708988
11: -55.1605225, -44.7897644, -55.1075745, -44.8244553, -4.9592495, 4.9758244
12: -11.5765877, -4.5835438, -11.5675936, -4.5851936, -6.2399597, 6.2386131
13: 0.8880055, 8.0301390, 0.8905675, 8.0091867, -5.3020020, 5.3144646
14: -71.0719147, -57.9545135, -71.0922699, -57.9608688, -8.2394295, 8.2674751
15: -8.9120235, 0.9227800, -8.9176083, 0.8995981, -4.8850994, 4.9022255
16: -33.5763931, -23.9675331, -33.5508194, -23.9899750, -6.4629402, 6.4711494
17: -88.6563797, -72.4192581, -88.6679993, -72.4382172, -8.1559601, 8.1886940
18: -4.1972079, 1.0632889, -4.1754560, 1.0586553, -3.4072227, 3.4025764
19: -30.5325451, -23.2012672, -30.5126457, -23.2200851, -4.6546555, 4.6586781
20: -11.1730614, -5.1576586, -11.1662779, -5.1607828, -4.9371414, 4.9307766
21: -43.5612335, -35.0513191, -43.5304298, -35.0787354, -4.2744312, 4.2860317
22: -27.0012093, -19.5390244, -27.0004654, -19.5497437, -4.3243599, 4.3376369
23: -20.8694973, -12.5062580, -20.8453674, -12.5212345, -4.7851238, 4.7910423
24: -16.8771172, -7.6407804, -16.8571033, -7.6498699, -7.1701508, 7.1717911
25: -14.6401129, -6.9574556, -14.6289577, -6.9743328, -4.1912689, 4.2035217
26: -14.6137590, -7.8181028, -14.6107244, -7.8179893, -6.5413780, 6.5442123
27: -14.6389408, -9.5302467, -14.6215277, -9.5461197, -4.0633259, 4.0676346
28: -10.0359812, -1.4237872, -10.0202713, -1.4216632, -6.1649437, 6.1497269
29: -45.5883331, -36.8221054, -45.5735359, -36.8449135, -5.0020294, 5.0209122
30: -32.2146606, -23.0048294, -32.1729431, -23.0252876, -5.0086498, 5.0253162
31: -32.2524033, -23.5173759, -32.2221184, -23.5390129, -6.3187866, 6.3101807
32: 7.7147226, 13.6665144, 7.7226386, 13.6796684, -4.1644192, 4.1431808
33: 4.6090717, 16.3037891, 4.6220932, 16.3158340, -6.6888847, 6.6617050
34: 20.5248680, 30.9823914, 20.5625839, 30.9787750, -5.7467575, 5.7227974
35: 16.4925289, 26.8572845, 16.5262413, 26.8552113, -5.4434814, 5.4229794
36: 28.8088570, 35.1176224, 28.8226566, 35.1170387, -3.4293900, 3.4184418
37: 11.0179739, 20.1096535, 11.0313387, 20.1143208, -5.9582748, 5.9448700
38: 34.8531952, 43.6908379, 34.8759842, 43.6723061, -6.0463715, 6.0304184
39: 8.9903316, 18.5086327, 8.9994221, 18.5042229, -6.5183983, 6.5090790
40: 15.7832565, 25.1224384, 15.8021736, 25.1368484, -5.8199921, 5.7848015
41: 6.7259016, 13.2174816, 6.7359509, 13.2245541, -5.0203285, 5.0063324
42: -12.3857994, -3.4617722, -12.3780327, -3.4619799, -7.0512009, 7.0445976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6338755
time: 4.89 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6338754
time: 5.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5873756, -8.4760866, -21.5865288, -8.4797583, -10.4472427, 10.4471436
1: -21.4321213, -12.2287636, -21.4367561, -12.2403631, -5.2835178, 5.3019638
2: -12.3986912, -5.7745886, -12.4003696, -5.7799087, -4.2703857, 4.2790718
3: -12.0143270, -4.1555634, -12.0054617, -4.1766753, -5.3700523, 5.3765640
4: -10.2981749, 0.0369308, -10.2937164, 0.0018325, -6.0659523, 6.0804710
5: -13.5651894, -4.0341926, -13.5697784, -4.0483270, -6.1463737, 6.1587563
6: -8.3528681, 0.5443628, -8.3123312, 0.5357430, -6.4965286, 6.4657021
7: -32.1578331, -22.0514069, -32.1593971, -22.0650387, -5.8301849, 5.8504143
8: -18.8172703, -9.0650339, -18.8156147, -9.1114292, -5.2428074, 5.2619724
9: -5.3468108, 1.4063028, -5.3377705, 1.3917158, -4.0609398, 4.0575790
10: -36.1405067, -27.7473030, -36.1335678, -27.7781715, -5.2615089, 5.2788906
11: -55.1675415, -44.7709808, -55.1075745, -44.8244553, -4.9675140, 4.9942303
12: -11.5829935, -4.5783935, -11.5675936, -4.5851936, -6.2426376, 6.2398148
13: 0.8782057, 8.0353088, 0.8905675, 8.0091867, -5.3120193, 5.3209457
14: -71.0884247, -57.9405632, -71.0922699, -57.9608688, -8.2557831, 8.2791405
15: -8.9242020, 0.9366841, -8.9176083, 0.8995981, -4.8975220, 4.9196148
16: -33.5911331, -23.9671917, -33.5508194, -23.9899750, -6.4798889, 6.4745636
17: -88.6778107, -72.3848724, -88.6679993, -72.4382172, -8.1794930, 8.2246628
18: -4.2013907, 1.0740848, -4.1754560, 1.0586553, -3.4119225, 3.4132996
19: -30.5339222, -23.2005119, -30.5126457, -23.2200851, -4.6570473, 4.6595287
20: -11.1740999, -5.1526041, -11.1662779, -5.1607828, -4.9387589, 4.9355984
21: -43.5629501, -35.0483246, -43.5304298, -35.0787354, -4.2771988, 4.2885704
22: -27.0084991, -19.5265179, -27.0004654, -19.5497437, -4.3325043, 4.3513241
23: -20.8755817, -12.5005035, -20.8453674, -12.5212345, -4.7937679, 4.7983589
24: -16.8822098, -7.6338596, -16.8571033, -7.6498699, -7.1749573, 7.1785851
25: -14.6443043, -6.9520712, -14.6289577, -6.9743328, -4.1966801, 4.2097759
26: -14.6220322, -7.7969537, -14.6107244, -7.8179893, -6.5503235, 6.5648842
27: -14.6419373, -9.5227213, -14.6215277, -9.5461197, -4.0668297, 4.0746098
28: -10.0377865, -1.4218202, -10.0202713, -1.4216632, -6.1686745, 6.1514664
29: -45.5967712, -36.8078270, -45.5735359, -36.8449135, -5.0117912, 5.0348835
30: -32.2183113, -22.9946671, -32.1729431, -23.0252876, -5.0124187, 5.0352688
31: -32.2602692, -23.5107574, -32.2221184, -23.5390129, -6.3293228, 6.3187675
32: 7.6959295, 13.6755581, 7.7226386, 13.6796684, -4.1837406, 4.1525364
33: 4.5854526, 16.3128586, 4.6220932, 16.3158340, -6.7175674, 6.6719055
34: 20.5147591, 30.9898796, 20.5625839, 30.9787750, -5.7585373, 5.7314320
35: 16.4767723, 26.8645210, 16.5262413, 26.8552113, -5.4618816, 5.4315796
36: 28.7952785, 35.1257210, 28.8226566, 35.1170387, -3.4434967, 3.4273720
37: 10.9971600, 20.1173115, 11.0313387, 20.1143208, -5.9816284, 5.9528122
38: 34.8387947, 43.7038383, 34.8759842, 43.6723061, -6.0607567, 6.0446510
39: 8.9639616, 18.5195007, 8.9994221, 18.5042229, -6.5453415, 6.5210342
40: 15.7626715, 25.1308022, 15.8021736, 25.1368484, -5.8418045, 5.7931557
41: 6.7078686, 13.2268858, 6.7359509, 13.2245541, -5.0393143, 5.0161591
42: -12.3983316, -3.4525018, -12.3780327, -3.4619799, -7.0630264, 7.0538902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6385799
time: 4.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6385803
time: 5.26 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5672455, -8.4783888, -21.5909901, -8.4790220, -10.4265289, 10.4420166
1: -21.4237137, -12.2342529, -21.4400692, -12.2368908, -5.2748699, 5.2991180
2: -12.3945704, -5.7748389, -12.4057627, -5.7779083, -4.2627106, 4.2760677
3: -12.0111580, -4.1600070, -12.0179863, -4.1672697, -5.3641853, 5.3732910
4: -10.2876625, 0.0246730, -10.3026886, 0.0049224, -6.0528030, 6.0681648
5: -13.5592213, -4.0388803, -13.5720196, -4.0451317, -6.1417809, 6.1530991
6: -8.3306007, 0.5306103, -8.3177471, 0.5451051, -6.4772415, 6.4517059
7: -32.1515274, -22.0571251, -32.1656647, -22.0568161, -5.8264046, 5.8478889
8: -18.8127232, -9.0715923, -18.8235111, -9.1058836, -5.2367687, 5.2551632
9: -5.3333635, 1.4026828, -5.3456097, 1.3941355, -4.0514412, 4.0648880
10: -36.1315498, -27.7596016, -36.1425705, -27.7643890, -5.2523727, 5.2674294
11: -55.1605225, -44.7897644, -55.1269302, -44.7889481, -4.9571018, 4.9567833
12: -11.5765877, -4.5835438, -11.5748186, -4.5759583, -6.2426682, 6.2409706
13: 0.8880055, 8.0301390, 0.8808190, 8.0137901, -5.3016396, 5.3179665
14: -71.0719147, -57.9545135, -71.0950623, -57.9581604, -8.2443771, 8.2720985
15: -8.9120235, 0.9227800, -8.9247675, 0.9020000, -4.8788910, 4.8981094
16: -33.5763931, -23.9675331, -33.5708160, -23.9673195, -6.4656296, 6.4705124
17: -88.6563797, -72.4192581, -88.6789856, -72.4196701, -8.1585312, 8.1860504
18: -4.1972079, 1.0632889, -4.1784062, 1.0637646, -3.4109573, 3.4045830
19: -30.5325451, -23.2012672, -30.5264053, -23.2013321, -4.6522999, 4.6515598
20: -11.1730614, -5.1576586, -11.1724339, -5.1511006, -4.9384880, 4.9290791
21: -43.5612335, -35.0513191, -43.5477371, -35.0519180, -4.2663956, 4.2693939
22: -27.0012093, -19.5390244, -27.0070457, -19.5395012, -4.3266430, 4.3360939
23: -20.8694973, -12.5062580, -20.8523483, -12.5072422, -4.7863121, 4.7862263
24: -16.8771172, -7.6407804, -16.8609352, -7.6399965, -7.1747971, 7.1716499
25: -14.6401129, -6.9574556, -14.6382627, -6.9579268, -4.1916962, 4.1967297
26: -14.6137590, -7.8181028, -14.6128912, -7.8143210, -6.5439606, 6.5451279
27: -14.6389408, -9.5302467, -14.6314735, -9.5305691, -4.0636635, 4.0637722
28: -10.0359812, -1.4237872, -10.0232248, -1.4187319, -6.1685066, 6.1528244
29: -45.5883331, -36.8221054, -45.5863686, -36.8217545, -5.0031853, 5.0113392
30: -32.2146606, -23.0048294, -32.1839066, -23.0037136, -5.0113373, 5.0163212
31: -32.2524033, -23.5173759, -32.2373619, -23.5171890, -6.3149338, 6.3016167
32: 7.7147226, 13.6665144, 7.7190766, 13.6802368, -4.1646423, 4.1465378
33: 4.6090717, 16.3037891, 4.6103926, 16.3190708, -6.6886272, 6.6733856
34: 20.5248680, 30.9823914, 20.5351562, 30.9973602, -5.7382221, 5.7224789
35: 16.4925289, 26.8572845, 16.5037994, 26.8678360, -5.4337673, 5.4239750
36: 28.8088570, 35.1176224, 28.8089371, 35.1256294, -3.4267530, 3.4187355
37: 11.0179739, 20.1096535, 11.0248508, 20.1182060, -5.9593430, 5.9498787
38: 34.8531952, 43.6908379, 34.8548050, 43.6875496, -6.0467110, 6.0354576
39: 8.9903316, 18.5086327, 8.9901352, 18.5050697, -6.5211487, 6.5198135
40: 15.7832565, 25.1224384, 15.7911005, 25.1386223, -5.8195763, 5.7947884
41: 6.7259016, 13.2174816, 6.7329211, 13.2274628, -5.0218811, 5.0083008
42: -12.3857994, -3.4617722, -12.3862467, -3.4491258, -7.0529480, 7.0422935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6338758
time: 5.09 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6338760
time: 5.28 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5873756, -8.4760866, -21.5909901, -8.4790220, -10.4474335, 10.4509048
1: -21.4321213, -12.2287636, -21.4400692, -12.2368908, -5.2850246, 5.3033829
2: -12.3986912, -5.7745886, -12.4057627, -5.7779083, -4.2662277, 4.2783852
3: -12.0143270, -4.1555634, -12.0179863, -4.1672697, -5.3670006, 5.3773499
4: -10.2981749, 0.0369308, -10.3026886, 0.0049224, -6.0632744, 6.0822983
5: -13.5651894, -4.0341926, -13.5720196, -4.0451317, -6.1489677, 6.1605873
6: -8.3528681, 0.5443628, -8.3177471, 0.5451051, -6.4991875, 6.4647865
7: -32.1578331, -22.0514069, -32.1656647, -22.0568161, -5.8344269, 5.8538780
8: -18.8172703, -9.0650339, -18.8235111, -9.1058836, -5.2427387, 5.2632122
9: -5.3468108, 1.4063028, -5.3456097, 1.3941355, -4.0654335, 4.0688267
10: -36.1405067, -27.7473030, -36.1425705, -27.7643890, -5.2615395, 5.2754059
11: -55.1675415, -44.7709808, -55.1269302, -44.7889481, -4.9653645, 4.9751759
12: -11.5829935, -4.5783935, -11.5748186, -4.5759583, -6.2453499, 6.2421761
13: 0.8782057, 8.0353088, 0.8808190, 8.0137901, -5.3116455, 5.3244476
14: -71.0884247, -57.9405632, -71.0950623, -57.9581604, -8.2607422, 8.2837677
15: -8.9242020, 0.9366841, -8.9247675, 0.9020000, -4.8913193, 4.9154968
16: -33.5911331, -23.9671917, -33.5708160, -23.9673195, -6.4826965, 6.4739265
17: -88.6778107, -72.3848724, -88.6789856, -72.4196701, -8.1820641, 8.2219925
18: -4.2013907, 1.0740848, -4.1784062, 1.0637646, -3.4156570, 3.4153042
19: -30.5339222, -23.2005119, -30.5264053, -23.2013321, -4.6546993, 4.6524048
20: -11.1740999, -5.1526041, -11.1724339, -5.1511006, -4.9401054, 4.9338741
21: -43.5629501, -35.0483246, -43.5477371, -35.0519180, -4.2691612, 4.2719288
22: -27.0084991, -19.5265179, -27.0070457, -19.5395012, -4.3347855, 4.3497715
23: -20.8755817, -12.5005035, -20.8523483, -12.5072422, -4.7949486, 4.7935276
24: -16.8822098, -7.6338596, -16.8609352, -7.6399965, -7.1796188, 7.1784210
25: -14.6443043, -6.9520712, -14.6382627, -6.9579268, -4.1971054, 4.2029781
26: -14.6220322, -7.7969537, -14.6128912, -7.8143210, -6.5529099, 6.5658035
27: -14.6419373, -9.5227213, -14.6314735, -9.5305691, -4.0671692, 4.0707359
28: -10.0377865, -1.4218202, -10.0232248, -1.4187319, -6.1722336, 6.1545563
29: -45.5967712, -36.8078270, -45.5863686, -36.8217545, -5.0129471, 5.0253029
30: -32.2183113, -22.9946671, -32.1839066, -23.0037136, -5.0151138, 5.0262547
31: -32.2602692, -23.5107574, -32.2373619, -23.5171890, -6.3254700, 6.3101921
32: 7.6959295, 13.6755581, 7.7190766, 13.6802368, -4.1839561, 4.1558933
33: 4.5854526, 16.3128586, 4.6103926, 16.3190708, -6.7172337, 6.6835899
34: 20.5147591, 30.9898796, 20.5351562, 30.9973602, -5.7500057, 5.7311287
35: 16.4767723, 26.8645210, 16.5037994, 26.8678360, -5.4521160, 5.4325771
36: 28.7952785, 35.1257210, 28.8089371, 35.1256294, -3.4408407, 3.4276648
37: 10.9971600, 20.1173115, 11.0248508, 20.1182060, -5.9827385, 5.9578094
38: 34.8387947, 43.7038383, 34.8548050, 43.6875496, -6.0611000, 6.0497551
39: 8.9639616, 18.5195007, 8.9901352, 18.5050697, -6.5480614, 6.5317612
40: 15.7626715, 25.1308022, 15.7911005, 25.1386223, -5.8413658, 5.8031425
41: 6.7078686, 13.2268858, 6.7329211, 13.2274628, -5.0408745, 5.0181274
42: -12.3983316, -3.4525018, -12.3862467, -3.4491258, -7.0647583, 7.0515747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6385805
time: 4.99 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6385804
time: 5.31 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5619202, -8.4773140, -21.6065216, -8.4775314, -10.4165421, 10.4790344
1: -21.4270687, -12.2364712, -21.4448509, -12.2345982, -5.2640228, 5.3067741
2: -12.3929873, -5.7769599, -12.4042301, -5.7796440, -4.2575073, 4.2835770
3: -12.0093021, -4.1673479, -12.0082998, -4.1722879, -5.3609848, 5.3684273
4: -10.2836637, 0.0053809, -10.3034935, 0.0140584, -6.0536308, 6.0540218
5: -13.5544376, -4.0442963, -13.5753832, -4.0436778, -6.1299934, 6.1653595
6: -8.3185015, 0.5380002, -8.3345490, 0.5489129, -6.4718704, 6.4757919
7: -32.1552505, -22.0553112, -32.1654129, -22.0575600, -5.8194313, 5.8458805
8: -18.8048077, -9.1060982, -18.8194313, -9.1049099, -5.2287312, 5.2231617
9: -5.3305693, 1.3936331, -5.3509741, 1.3953116, -4.0387936, 4.0615368
10: -36.1342239, -27.7654953, -36.1420822, -27.7659531, -5.2477264, 5.2775364
11: -55.1317596, -44.8025970, -55.1142654, -44.8063583, -4.9471779, 4.9682598
12: -11.5788498, -4.5906925, -11.5739937, -4.5803232, -6.2421074, 6.2388573
13: 0.8906822, 8.0121489, 0.8812381, 8.0142202, -5.3007088, 5.3053322
14: -71.0820084, -57.9611969, -71.1082993, -57.9469109, -8.2327805, 8.2787247
15: -8.9091282, 0.9019375, -8.9289417, 0.9135528, -4.8832703, 4.8902397
16: -33.5558128, -23.9755249, -33.5651779, -23.9900322, -6.4482574, 6.4952736
17: -88.6771698, -72.4199905, -88.6889191, -72.4026337, -8.1951637, 8.2036514
18: -4.1770797, 1.0589998, -4.1795034, 1.0691953, -3.3965721, 3.4023781
19: -30.5245628, -23.2039490, -30.5139713, -23.2195415, -4.6469593, 4.6585293
20: -11.1713247, -5.1568689, -11.1672792, -5.1550875, -4.9442863, 4.9305496
21: -43.5464859, -35.0578613, -43.5320969, -35.0761681, -4.2627544, 4.2835426
22: -27.0048599, -19.5391235, -27.0077610, -19.5372467, -4.3370399, 4.3445568
23: -20.8512859, -12.5133362, -20.8513718, -12.5160065, -4.7733574, 4.7857590
24: -16.8588409, -7.6453123, -16.8620834, -7.6435452, -7.1581345, 7.1685677
25: -14.6369247, -6.9584475, -14.6330605, -6.9691849, -4.1934490, 4.2011166
26: -14.6177588, -7.8184299, -14.6186495, -7.7965941, -6.5701904, 6.5392342
27: -14.6302242, -9.5350361, -14.6244287, -9.5388842, -4.0623608, 4.0664883
28: -10.0220985, -1.4306667, -10.0220070, -1.4201647, -6.1698532, 6.1490707
29: -45.5831375, -36.8256836, -45.5817337, -36.8308411, -5.0047455, 5.0268631
30: -32.1859398, -23.0199184, -32.1765594, -23.0157852, -4.9891758, 5.0106258
31: -32.2348633, -23.5174713, -32.2298279, -23.5328236, -6.3072853, 6.3162117
32: 7.7183456, 13.6730137, 7.7039404, 13.6884041, -4.1664848, 4.1643219
33: 4.6096482, 16.3102932, 4.5978980, 16.3244820, -6.6920853, 6.6858177
34: 20.5353756, 30.9838066, 20.5525742, 30.9858131, -5.7425308, 5.7320271
35: 16.5048599, 26.8592873, 16.5106392, 26.8619385, -5.4375191, 5.4357243
36: 28.8102207, 35.1238976, 28.8091507, 35.1249008, -3.4347057, 3.4324932
37: 11.0252705, 20.1128159, 11.0106869, 20.1216049, -5.9566727, 5.9644203
38: 34.8629417, 43.6871758, 34.8619308, 43.6850815, -6.0475616, 6.0340767
39: 8.9976864, 18.5058441, 8.9733686, 18.5147591, -6.5208054, 6.5276756
40: 15.7926989, 25.1246166, 15.7818089, 25.1449242, -5.8172035, 5.7957439
41: 6.7330136, 13.2223425, 6.7179971, 13.2335491, -5.0209656, 5.0264015
42: -12.3867254, -3.4558735, -12.3908043, -3.4529898, -7.0593605, 7.0533333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5853165, upper bound: 3.6381775
time: 5.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5853165, upper bound: 3.6381777
time: 5.34 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5619202, -8.4773140, -21.6110001, -8.4767599, -10.4167175, 10.4828110
1: -21.4270687, -12.2364712, -21.4481125, -12.2311363, -5.2655315, 5.3081913
2: -12.3929873, -5.7769599, -12.4096527, -5.7776737, -4.2533379, 4.2828999
3: -12.0093021, -4.1673479, -12.0208273, -4.1629062, -5.3579292, 5.3691750
4: -10.2836637, 0.0053809, -10.3124943, 0.0171833, -6.0509758, 6.0558338
5: -13.5544376, -4.0442963, -13.5776272, -4.0404482, -6.1325378, 6.1671944
6: -8.3185015, 0.5380002, -8.3399906, 0.5583075, -6.4745293, 6.4748650
7: -32.1552505, -22.0553112, -32.1716995, -22.0493050, -5.8236656, 5.8493214
8: -18.8048077, -9.1060982, -18.8273201, -9.0993786, -5.2286625, 5.2244339
9: -5.3305693, 1.3936331, -5.3587289, 1.3977078, -4.0432587, 4.0728283
10: -36.1342239, -27.7654953, -36.1511040, -27.7521687, -5.2477741, 5.2740898
11: -55.1317596, -44.8025970, -55.1336517, -44.7707825, -4.9450474, 4.9492207
12: -11.5788498, -4.5906925, -11.5812292, -4.5710626, -6.2448196, 6.2412033
13: 0.8906822, 8.0121489, 0.8715006, 8.0188313, -5.3003540, 5.3088188
14: -71.0820084, -57.9611969, -71.1110458, -57.9442139, -8.2377396, 8.2833557
15: -8.9091282, 0.9019375, -8.9361000, 0.9159389, -4.8770275, 4.8861217
16: -33.5558128, -23.9755249, -33.5851669, -23.9673901, -6.4509277, 6.4946594
17: -88.6771698, -72.4199905, -88.6999054, -72.3840256, -8.1977615, 8.2009926
18: -4.1770797, 1.0589998, -4.1824565, 1.0743344, -3.4003353, 3.4043789
19: -30.5245628, -23.2039490, -30.5277481, -23.2007751, -4.6446190, 4.6513977
20: -11.1713247, -5.1568689, -11.1734114, -5.1454167, -4.9456062, 4.9288406
21: -43.5464859, -35.0578613, -43.5494041, -35.0493240, -4.2547169, 4.2669048
22: -27.0048599, -19.5391235, -27.0143318, -19.5270081, -4.3393307, 4.3430061
23: -20.8512859, -12.5133362, -20.8583488, -12.5020084, -4.7745075, 4.7809410
24: -16.8588409, -7.6453123, -16.8659077, -7.6336226, -7.1627731, 7.1684456
25: -14.6369247, -6.9584475, -14.6423645, -6.9527545, -4.1938648, 4.1943188
26: -14.6177588, -7.8184299, -14.6208391, -7.7929163, -6.5727844, 6.5401382
27: -14.6302242, -9.5350361, -14.6343555, -9.5233107, -4.0627117, 4.0626259
28: -10.0220985, -1.4306667, -10.0249376, -1.4172479, -6.1734161, 6.1521606
29: -45.5831375, -36.8256836, -45.5945816, -36.8076553, -5.0059185, 5.0172958
30: -32.1859398, -23.0199184, -32.1875534, -22.9942093, -4.9918823, 5.0016232
31: -32.2348633, -23.5174713, -32.2451019, -23.5110188, -6.3034554, 6.3076553
32: 7.7183456, 13.6730137, 7.7003231, 13.6889763, -4.1667061, 4.1676712
33: 4.6096482, 16.3102932, 4.5861449, 16.3276997, -6.6918259, 6.6974945
34: 20.5353756, 30.9838066, 20.5251770, 31.0044098, -5.7340050, 5.7317047
35: 16.5048599, 26.8592873, 16.4881592, 26.8745708, -5.4278297, 5.4367332
36: 28.8102207, 35.1238976, 28.7954254, 35.1334839, -3.4320736, 3.4328003
37: 11.0252705, 20.1128159, 11.0041676, 20.1254883, -5.9577522, 5.9694214
38: 34.8629417, 43.6871758, 34.8407860, 43.7003021, -6.0478897, 6.0391045
39: 8.9976864, 18.5058441, 8.9640369, 18.5156174, -6.5235405, 6.5384254
40: 15.7926989, 25.1246166, 15.7706671, 25.1467209, -5.8167801, 5.8058052
41: 6.7330136, 13.2223425, 6.7149720, 13.2364779, -5.0225029, 5.0283585
42: -12.3867254, -3.4558735, -12.3989906, -3.4401145, -7.0611038, 7.0510178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5853165, upper bound: 3.6381773
time: 5.11 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5853165, upper bound: 3.6381780
time: 6.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5711842, -8.4744730, -21.6063957, -8.4774961, -10.4484406, 10.4766312
1: -21.4287968, -12.2341938, -21.4448547, -12.2348795, -5.2702866, 5.3079147
2: -12.3960075, -5.7740884, -12.4043198, -5.7796717, -4.2668285, 4.2823830
3: -12.0133085, -4.1585083, -12.0084314, -4.1722260, -5.3649483, 5.3772697
4: -10.2943420, 0.0263026, -10.3039322, 0.0140448, -6.0638847, 6.0753708
5: -13.5609341, -4.0370030, -13.5755405, -4.0436478, -6.1396103, 6.1635399
6: -8.3328705, 0.5433073, -8.3345432, 0.5491323, -6.4865456, 6.4813080
7: -32.1562080, -22.0567131, -32.1654358, -22.0592957, -5.8224411, 5.8504105
8: -18.8164101, -9.0703316, -18.8199120, -9.1048880, -5.2400799, 5.2590847
9: -5.3350859, 1.4032512, -5.3511276, 1.3953470, -4.0431271, 4.0712051
10: -36.1397247, -27.7590675, -36.1422119, -27.7658997, -5.2534599, 5.2795315
11: -55.1669731, -44.7890625, -55.1142921, -44.8057556, -4.9831142, 4.9815121
12: -11.5809202, -4.5831518, -11.5740080, -4.5801487, -6.2430229, 6.2457161
13: 0.8846473, 8.0314884, 0.8809891, 8.0143003, -5.3068581, 5.3251877
14: -71.0861130, -57.9543648, -71.1083069, -57.9469070, -8.2463799, 8.2839355
15: -8.9220886, 0.9235744, -8.9293432, 0.9135480, -4.8962803, 4.9122200
16: -33.5792122, -23.9671097, -33.5652313, -23.9896507, -6.4719620, 6.5030441
17: -88.6774445, -72.4169769, -88.6888885, -72.4039154, -8.1953201, 8.2087555
18: -4.2007470, 1.0639167, -4.1795454, 1.0694110, -3.4204922, 3.4071770
19: -30.5335884, -23.2010078, -30.5140114, -23.2194405, -4.6565132, 4.6610661
20: -11.1734238, -5.1556897, -11.1672935, -5.1558475, -4.9439888, 4.9348373
21: -43.5633163, -35.0510635, -43.5321579, -35.0758705, -4.2798538, 4.2902603
22: -27.0083084, -19.5384941, -27.0075874, -19.5372486, -4.3409519, 4.3445587
23: -20.8707676, -12.5043163, -20.8514004, -12.5156994, -4.7931747, 4.7953415
24: -16.8787384, -7.6378627, -16.8621788, -7.6432018, -7.1782455, 7.1759796
25: -14.6409645, -6.9558024, -14.6331177, -6.9691086, -4.1980076, 4.2039280
26: -14.6221619, -7.8150148, -14.6187496, -7.7969379, -6.5703621, 6.5576286
27: -14.6416073, -9.5297642, -14.6245022, -9.5386877, -4.0733395, 4.0718250
28: -10.0368824, -1.4221604, -10.0220490, -1.4198699, -6.1699944, 6.1566391
29: -45.5960922, -36.8214569, -45.5817566, -36.8306503, -5.0177994, 5.0301819
30: -32.2182312, -23.0042248, -32.1765518, -23.0152531, -5.0220528, 5.0265827
31: -32.2542191, -23.5131187, -32.2299347, -23.5326157, -6.3280449, 6.3200798
32: 7.7133918, 13.6752224, 7.7039194, 13.6884727, -4.1716480, 4.1664562
33: 4.6069069, 16.3115788, 4.5986271, 16.3244991, -6.6977310, 6.6870499
34: 20.5231781, 30.9887352, 20.5525208, 30.9860115, -5.7549610, 5.7365570
35: 16.4910145, 26.8643875, 16.5105858, 26.8621578, -5.4515343, 5.4407616
36: 28.8081112, 35.1255417, 28.8090820, 35.1249466, -3.4367409, 3.4339886
37: 11.0157509, 20.1163387, 11.0106525, 20.1217194, -5.9666061, 5.9676361
38: 34.8522034, 43.7010612, 34.8615837, 43.6850853, -6.0585785, 6.0482826
39: 8.9883671, 18.5176048, 8.9731808, 18.5147476, -6.5299339, 6.5371056
40: 15.7818136, 25.1295052, 15.7816820, 25.1449509, -5.8272171, 5.8038120
41: 6.7241435, 13.2266836, 6.7179632, 13.2337084, -5.0299492, 5.0306969
42: -12.3867283, -3.4531779, -12.3905563, -3.4529684, -7.0595665, 7.0591621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5973362, upper bound: 3.6382294
time: 5.93 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5973362, upper bound: 3.6382295
time: 6.33 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5711842, -8.4744730, -21.6108627, -8.4767103, -10.4485855, 10.4803848
1: -21.4287968, -12.2341938, -21.4481621, -12.2314167, -5.2717934, 5.3093376
2: -12.3960075, -5.7740884, -12.4097290, -5.7776909, -4.2626686, 4.2816944
3: -12.0133085, -4.1585083, -12.0209656, -4.1628599, -5.3618965, 5.3780327
4: -10.2943420, 0.0263026, -10.3128624, 0.0171697, -6.0612259, 6.0771885
5: -13.5609341, -4.0370030, -13.5778141, -4.0404425, -6.1421585, 6.1653824
6: -8.3328705, 0.5433073, -8.3400021, 0.5585490, -6.4892006, 6.4803963
7: -32.1562080, -22.0567131, -32.1716919, -22.0510712, -5.8266869, 5.8538761
8: -18.8164101, -9.0703316, -18.8278160, -9.0993271, -5.2400112, 5.2603226
9: -5.3350859, 1.4032512, -5.3588872, 1.3977203, -4.0475883, 4.0824947
10: -36.1397247, -27.7590675, -36.1512375, -27.7521324, -5.2535095, 5.2760715
11: -55.1669731, -44.7890625, -55.1336327, -44.7702026, -4.9809856, 4.9624653
12: -11.5809202, -4.5831518, -11.5812206, -4.5708961, -6.2457352, 6.2480812
13: 0.8846473, 8.0314884, 0.8712600, 8.0189142, -5.3064842, 5.3286705
14: -71.0861130, -57.9543648, -71.1110840, -57.9441605, -8.2513428, 8.2885818
15: -8.9220886, 0.9235744, -8.9364929, 0.9159493, -4.8900776, 4.9081059
16: -33.5792122, -23.9671097, -33.5851936, -23.9670010, -6.4746284, 6.5024261
17: -88.6774445, -72.4169769, -88.6998596, -72.3852997, -8.1979065, 8.2061081
18: -4.2007470, 1.0639167, -4.1824980, 1.0745461, -3.4242535, 3.4091873
19: -30.5335884, -23.2010078, -30.5278130, -23.2006969, -4.6541557, 4.6539364
20: -11.1734238, -5.1556897, -11.1734371, -5.1461768, -4.9453011, 4.9331341
21: -43.5633163, -35.0510635, -43.5494194, -35.0490417, -4.2718182, 4.2736225
22: -27.0083084, -19.5384941, -27.0141525, -19.5270176, -4.3432446, 4.3430119
23: -20.8707676, -12.5043163, -20.8583794, -12.5017166, -4.7943420, 4.7905273
24: -16.8787384, -7.6378627, -16.8659897, -7.6332955, -7.1828918, 7.1758347
25: -14.6409645, -6.9558024, -14.6424141, -6.9526939, -4.1984234, 4.1971264
26: -14.6221619, -7.8150148, -14.6209316, -7.7932777, -6.5729561, 6.5585365
27: -14.6416073, -9.5297642, -14.6344376, -9.5231142, -4.0736923, 4.0679607
28: -10.0368824, -1.4221604, -10.0249949, -1.4169520, -6.1735573, 6.1597252
29: -45.5960922, -36.8214569, -45.5945930, -36.8074532, -5.0189667, 5.0206203
30: -32.2182312, -23.0042248, -32.1875534, -22.9937000, -5.0247555, 5.0175896
31: -32.2542191, -23.5131187, -32.2451744, -23.5108204, -6.3241615, 6.3115387
32: 7.7133918, 13.6752224, 7.7003317, 13.6890411, -4.1718674, 4.1698151
33: 4.6069069, 16.3115788, 4.5868616, 16.3277245, -6.6974754, 6.6987305
34: 20.5231781, 30.9887352, 20.5250931, 31.0045986, -5.7464333, 5.7362309
35: 16.4910145, 26.8643875, 16.4880943, 26.8747902, -5.4418316, 5.4417591
36: 28.8081112, 35.1255417, 28.7953720, 35.1335258, -3.4341011, 3.4342861
37: 11.0157509, 20.1163387, 11.0041037, 20.1255932, -5.9676743, 5.9726410
38: 34.8522034, 43.7010612, 34.8404312, 43.7003250, -6.0589256, 6.0533218
39: 8.9883671, 18.5176048, 8.9638443, 18.5155983, -6.5326767, 6.5478477
40: 15.7818136, 25.1295052, 15.7705803, 25.1467476, -5.8268013, 5.8138733
41: 6.7241435, 13.2266836, 6.7149405, 13.2366371, -5.0314980, 5.0326614
42: -12.3867283, -3.4531779, -12.3987503, -3.4401159, -7.0612946, 7.0568581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=77, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1769

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5973362, upper bound: 3.6382294
time: 6.11 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5973362, upper bound: 3.6382299
time: 5.18 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 13.41 seconds
IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6316468
IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6381818
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6310388
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6374970
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6166938
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6381821
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6310392
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6374973
IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6319993
IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5856678, upper bound: 3.6385325
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5899425, upper bound: 3.6382337
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5973362, upper bound: 3.6382337
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5856645, upper bound: 3.6189817
IS_A2_B1_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5856645, upper bound: 3.6255130
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5899425, upper bound: 3.6382340
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5973362, upper bound: 3.6382340
IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6338233
IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6338238
IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6385279
IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6385281
IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6338239
IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6338239
IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6385283
IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5749351, upper bound: 3.6385287
IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6338755
IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6338754
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6385799
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6385803
IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6338758
IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6338760
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6385805
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5869516, upper bound: 3.6385804
IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5853165, upper bound: 3.6381775
IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5853165, upper bound: 3.6381777
IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5853165, upper bound: 3.6381773
IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5853165, upper bound: 3.6381780
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5973362, upper bound: 3.6382294
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5973362, upper bound: 3.6382295
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5973362, upper bound: 3.6382294
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 13.41
Output dim: 38, lower bound: -3.5973362, upper bound: 3.6382299

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5742188, -8.4789572, -21.5497856, -8.4852219, -10.4135361, 10.3827820
1: -21.4290352, -12.2320938, -21.4366894, -12.2486811, -5.2382908, 5.2805252
2: -12.3949013, -5.7782068, -12.3972998, -5.7838650, -4.2467480, 4.2653179
3: -12.0095510, -4.1682668, -11.9997730, -4.1877441, -5.3485336, 5.3652534
4: -10.2857018, 0.0084291, -10.2805767, -0.0181570, -6.0234070, 6.0672684
5: -13.5576019, -4.0438442, -13.5624466, -4.0617056, -6.1181602, 6.1526680
6: -8.3326015, 0.5378714, -8.3043442, 0.5338780, -6.4622612, 6.4259109
7: -32.1557770, -22.0512791, -32.1558876, -22.0963402, -5.7980156, 5.8572140
8: -18.8046989, -9.1146021, -18.7982864, -9.1235867, -5.2075577, 5.2231712
9: -5.3412209, 1.3932781, -5.3183298, 1.3881464, -4.0475464, 4.0343647
10: -36.1330109, -27.7551689, -36.1345177, -27.7839279, -5.2228203, 5.2663612
11: -55.1211739, -44.7848816, -55.1151085, -44.8622856, -4.9153538, 5.0006351
12: -11.5786514, -4.5867538, -11.5653887, -4.5988941, -6.2226295, 6.2049026
13: 0.8850516, 8.0086393, 0.9012778, 7.9903574, -5.2786827, 5.2895775
14: -71.0823975, -57.9487801, -71.1004639, -57.9731674, -8.1990204, 8.2728996
15: -8.9086590, 0.9085894, -8.9118433, 0.8907309, -4.8580551, 4.9083996
16: -33.5579758, -23.9757271, -33.5399780, -24.0129795, -6.4533234, 6.4458008
17: -88.6756668, -72.3891602, -88.6761627, -72.4893112, -8.1301575, 8.2523537
18: -4.1684327, 1.0690997, -4.1644273, 1.0401332, -3.3754139, 3.4016590
19: -30.5212402, -23.2038021, -30.5037231, -23.2301750, -4.6386471, 4.6502113
20: -11.1717358, -5.1547604, -11.1632576, -5.1662717, -4.9172401, 4.9193630
21: -43.5403595, -35.0556870, -43.5255890, -35.0936317, -4.2531567, 4.2697506
22: -27.0025520, -19.5272312, -26.9951992, -19.5678177, -4.3133011, 4.3571720
23: -20.8487282, -12.5109901, -20.8216019, -12.5378475, -4.7752552, 4.7631721
24: -16.8535881, -7.6420646, -16.8345814, -7.6664748, -7.1449051, 7.1480103
25: -14.6369104, -6.9556646, -14.6076899, -6.9822035, -4.1916428, 4.1813431
26: -14.6165190, -7.8010273, -14.6107941, -7.8233008, -6.5259933, 6.5325241
27: -14.6268396, -9.5285769, -14.6139545, -9.5693150, -4.0422955, 4.0682087
28: -10.0195856, -1.4316912, -10.0152740, -1.4302133, -6.1392021, 6.1277084
29: -45.5766602, -36.8121262, -45.5669098, -36.8734894, -4.9727726, 5.0335655
30: -32.1762924, -23.0122414, -32.1748543, -23.0522118, -4.9691238, 5.0179501
31: -32.2324371, -23.5158768, -32.2030373, -23.5537529, -6.2937317, 6.2921829
32: 7.7026072, 13.6725559, 7.7256427, 13.6792336, -4.1730919, 4.1336842
33: 4.5920687, 16.3102798, 4.6619358, 16.3128242, -6.7235813, 6.6275177
34: 20.5322151, 30.9840355, 20.5945053, 30.9686184, -5.7584076, 5.7004795
35: 16.4964237, 26.8584499, 16.5636406, 26.8457928, -5.4655247, 5.3950672
36: 28.7978973, 35.1231842, 28.8407173, 35.1193771, -3.4545841, 3.4066343
37: 11.0117073, 20.1127853, 11.0700607, 20.1089287, -5.9769745, 5.9072876
38: 34.8505974, 43.6846619, 34.9211426, 43.6712532, -6.0665741, 6.0001564
39: 8.9739828, 18.5052071, 9.0359783, 18.5124321, -6.5580215, 6.4777832
40: 15.7768478, 25.1227722, 15.8213854, 25.1351929, -5.8276024, 5.7505436
41: 6.7209663, 13.2216759, 6.7500315, 13.2234631, -5.0255966, 4.9867020
42: -12.3980618, -3.4565544, -12.3669062, -3.4569907, -7.0578842, 7.0181503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1769

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6339338
time: 5.27 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6381818
time: 6.19 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5834599, -8.4761534, -21.5425224, -8.4853573, -10.4631119, 10.3714981
1: -21.4307556, -12.2297544, -21.4359550, -12.2514191, -5.2573013, 5.2806492
2: -12.3979378, -5.7753315, -12.3963223, -5.7868547, -4.2555847, 4.2656784
3: -12.0135527, -4.1594543, -11.9992208, -4.1956019, -5.3405609, 5.3735008
4: -10.2963762, 0.0293608, -10.2789803, -0.0388678, -6.0001907, 6.0871983
5: -13.5640945, -4.0365562, -13.5602779, -4.0681963, -6.1264725, 6.1516724
6: -8.3469887, 0.5431714, -8.2907400, 0.5338166, -6.4767647, 6.4106750
7: -32.1567383, -22.0526676, -32.1553078, -22.0963326, -5.7810059, 5.8588181
8: -18.8163052, -9.0788727, -18.7982502, -9.1588516, -5.1768646, 5.2586002
9: -5.3457656, 1.4028986, -5.3176231, 1.3790383, -4.0518799, 4.0430927
10: -36.1385002, -27.7487221, -36.1319923, -27.7898483, -5.2477951, 5.2649879
11: -55.1563911, -44.7713699, -55.0802917, -44.8630981, -4.9452286, 4.9788074
12: -11.5807209, -4.5792556, -11.5633354, -4.6021676, -6.2200851, 6.2071304
13: 0.8790204, 8.0279617, 0.9013723, 7.9729605, -5.2615471, 5.3090668
14: -71.0864716, -57.9418716, -71.0976791, -57.9784355, -8.2170410, 8.2739105
15: -8.9215736, 0.9301729, -8.9085445, 0.8695211, -4.8471050, 4.9267349
16: -33.5813904, -23.9673271, -33.5177307, -24.0135155, -6.5052071, 6.4305916
17: -88.6759720, -72.3861160, -88.6761856, -72.4919739, -8.1038742, 8.2589455
18: -4.1920924, 1.0740075, -4.1417656, 1.0403466, -3.3977566, 3.3839149
19: -30.5302658, -23.2008743, -30.4962692, -23.2306175, -4.6423187, 4.6449471
20: -11.1738319, -5.1535616, -11.1622257, -5.1662650, -4.9196205, 4.9272099
21: -43.5571709, -35.0488892, -43.5098877, -35.0947838, -4.2698498, 4.2607117
22: -27.0059948, -19.5266075, -26.9934959, -19.5689125, -4.3157616, 4.3555546
23: -20.8682251, -12.5019474, -20.8032722, -12.5394564, -4.7932014, 4.7540226
24: -16.8734665, -7.6346040, -16.8164062, -7.6660986, -7.1651077, 7.1369400
25: -14.6409473, -6.9530144, -14.6052351, -6.9831715, -4.1950779, 4.1797295
26: -14.6209106, -7.7976198, -14.6097927, -7.8249307, -6.5222321, 6.5543251
27: -14.6382122, -9.5233068, -14.6046715, -9.5702991, -4.0511837, 4.0649147
28: -10.0343723, -1.4231758, -10.0019512, -1.4317718, -6.1392441, 6.1392250
29: -45.5896301, -36.8078995, -45.5547943, -36.8737526, -4.9910774, 5.0242310
30: -32.2085609, -22.9965286, -32.1430855, -23.0560169, -4.9936333, 5.0020027
31: -32.2517738, -23.5114975, -32.1862183, -23.5542603, -6.3132935, 6.2795029
32: 7.6976581, 13.6747322, 7.7302217, 13.6790581, -4.1780396, 4.1232567
33: 4.5893106, 16.3115711, 4.6640296, 16.3121185, -6.7284832, 6.6127319
34: 20.5200005, 30.9889946, 20.6057014, 30.9685459, -5.7705307, 5.6809578
35: 16.4825993, 26.8635826, 16.5759163, 26.8461418, -5.4796505, 5.3808975
36: 28.7958012, 35.1248474, 28.8421669, 35.1188660, -3.4562721, 3.4017782
37: 11.0022554, 20.1163216, 11.0779409, 20.1082382, -5.9859848, 5.8927765
38: 34.8398132, 43.6985664, 34.9237823, 43.6579132, -6.0641441, 5.9959946
39: 8.9646778, 18.5169983, 9.0368147, 18.5009518, -6.5579834, 6.4857635
40: 15.7659302, 25.1276417, 15.8301744, 25.1310806, -5.8332138, 5.7521973
41: 6.7120681, 13.2260218, 6.7580085, 13.2230406, -5.0339432, 4.9773903
42: -12.3980570, -3.4538286, -12.3670769, -3.4589555, -7.0571404, 7.0165138

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1769

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6332456
time: 5.18 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6374970
time: 5.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5742188, -8.4789572, -21.5542889, -8.4844904, -10.4136887, 10.3865166
1: -21.4290352, -12.2320938, -21.4399872, -12.2451820, -5.2398052, 5.2819481
2: -12.3949013, -5.7782068, -12.4027195, -5.7818608, -4.2425861, 4.2646351
3: -12.0095510, -4.1682668, -12.0123081, -4.1783438, -5.3454742, 5.3660507
4: -10.2857018, 0.0084291, -10.2895489, -0.0150387, -6.0207558, 6.0690804
5: -13.5576019, -4.0438442, -13.5646915, -4.0585237, -6.1207008, 6.1544876
6: -8.3326015, 0.5378714, -8.3097410, 0.5432746, -6.4649315, 6.4249916
7: -32.1557770, -22.0512791, -32.1621475, -22.0881195, -5.8022385, 5.8606567
8: -18.8046989, -9.1146021, -18.8062229, -9.1180887, -5.2074738, 5.2244549
9: -5.3412209, 1.3932781, -5.3261600, 1.3905404, -4.0520477, 4.0455570
10: -36.1330109, -27.7551689, -36.1435089, -27.7701797, -5.2228470, 5.2628670
11: -55.1211739, -44.7848816, -55.1344910, -44.8267403, -4.9131889, 4.9815826
12: -11.5786514, -4.5867538, -11.5726290, -4.5896258, -6.2253456, 6.2072868
13: 0.8850516, 8.0086393, 0.8915400, 7.9949312, -5.2782822, 5.2930565
14: -71.0823975, -57.9487801, -71.1032257, -57.9704285, -8.2039871, 8.2775230
15: -8.9086590, 0.9085894, -8.9189930, 0.8931608, -4.8518181, 4.9042759
16: -33.5579758, -23.9757271, -33.5600052, -23.9903374, -6.4561462, 6.4451599
17: -88.6756668, -72.3891602, -88.6871490, -72.4707031, -8.1327248, 8.2496643
18: -4.1684327, 1.0690997, -4.1673861, 1.0452347, -3.3791389, 3.4036427
19: -30.5212402, -23.2038021, -30.5174904, -23.2114296, -4.6363087, 4.6430836
20: -11.1717358, -5.1547604, -11.1693745, -5.1565981, -4.9185638, 4.9176350
21: -43.5403595, -35.0556870, -43.5428925, -35.0668106, -4.2451191, 4.2531033
22: -27.0025520, -19.5272312, -27.0017948, -19.5575848, -4.3155861, 4.3556137
23: -20.8487282, -12.5109901, -20.8285637, -12.5238104, -4.7764244, 4.7583256
24: -16.8535881, -7.6420646, -16.8384247, -7.6565685, -7.1495361, 7.1478577
25: -14.6369104, -6.9556646, -14.6169662, -6.9658098, -4.1920624, 4.1745338
26: -14.6165190, -7.8010273, -14.6129608, -7.8196192, -6.5285683, 6.5334358
27: -14.6268396, -9.5285769, -14.6238871, -9.5537577, -4.0426083, 4.0643253
28: -10.0195856, -1.4316912, -10.0182056, -1.4273407, -6.1427536, 6.1307907
29: -45.5766602, -36.8121262, -45.5797729, -36.8503494, -4.9739056, 5.0239925
30: -32.1762924, -23.0122414, -32.1858444, -23.0306511, -4.9718056, 5.0089169
31: -32.2324371, -23.5158768, -32.2182655, -23.5319328, -6.2899094, 6.2835846
32: 7.7026072, 13.6725559, 7.7220559, 13.6798153, -4.1733093, 4.1370296
33: 4.5920687, 16.3102798, 4.6502810, 16.3160515, -6.7232285, 6.6392097
34: 20.5322151, 30.9840355, 20.5670834, 30.9871941, -5.7498589, 5.7001762
35: 16.4964237, 26.8584499, 16.5412064, 26.8584137, -5.4557629, 5.3960648
36: 28.7978973, 35.1231842, 28.8270187, 35.1279831, -3.4519291, 3.4069405
37: 11.0117073, 20.1127853, 11.0635157, 20.1127968, -5.9780807, 5.9122696
38: 34.8505974, 43.6846619, 34.8999596, 43.6865005, -6.0668755, 6.0052528
39: 8.9739828, 18.5052071, 9.0267048, 18.5132866, -6.5607376, 6.4885216
40: 15.7768478, 25.1227722, 15.8102894, 25.1369743, -5.8271599, 5.7605267
41: 6.7209663, 13.2216759, 6.7470183, 13.2263851, -5.0271530, 4.9886665
42: -12.3980618, -3.4565544, -12.3750868, -3.4441195, -7.0596199, 7.0158386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1769

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6339340
time: 6.09 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6381819
time: 7.03 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5834599, -8.4761534, -21.5470181, -8.4845743, -10.4632568, 10.3752441
1: -21.4307556, -12.2297544, -21.4392567, -12.2478981, -5.2588158, 5.2820740
2: -12.3979378, -5.7753315, -12.4017029, -5.7848673, -4.2514305, 4.2649918
3: -12.0135527, -4.1594543, -12.0117426, -4.1861687, -5.3374977, 5.3742828
4: -10.2963762, 0.0293608, -10.2879629, -0.0357738, -5.9975204, 6.0890026
5: -13.5640945, -4.0365562, -13.5625381, -4.0650082, -6.1290169, 6.1534958
6: -8.3469887, 0.5431714, -8.2961578, 0.5431879, -6.4794235, 6.4097672
7: -32.1567383, -22.0526676, -32.1615677, -22.0881195, -5.7852249, 5.8622856
8: -18.8163052, -9.0788727, -18.8061657, -9.1533127, -5.1767902, 5.2598553
9: -5.3457656, 1.4028986, -5.3254790, 1.3814548, -4.0563774, 4.0542870
10: -36.1385002, -27.7487221, -36.1409836, -27.7760696, -5.2478294, 5.2615166
11: -55.1563911, -44.7713699, -55.0997162, -44.8275681, -4.9430428, 4.9597530
12: -11.5807209, -4.5792556, -11.5705681, -4.5928936, -6.2228012, 6.2094955
13: 0.8790204, 8.0279617, 0.8916259, 7.9775629, -5.2611542, 5.3125648
14: -71.0864716, -57.9418716, -71.1004715, -57.9756851, -8.2220116, 8.2785492
15: -8.9215736, 0.9301729, -8.9156971, 0.8719544, -4.8408909, 4.9226093
16: -33.5813904, -23.9673271, -33.5377579, -23.9908581, -6.5080032, 6.4299469
17: -88.6759720, -72.3861160, -88.6871719, -72.4733887, -8.1064415, 8.2562637
18: -4.1920924, 1.0740075, -4.1446924, 1.0454412, -3.4014797, 3.3858719
19: -30.5302658, -23.2008743, -30.5100517, -23.2118874, -4.6399746, 4.6378174
20: -11.1738319, -5.1535616, -11.1683760, -5.1565895, -4.9209518, 4.9254837
21: -43.5571709, -35.0488892, -43.5271683, -35.0679703, -4.2618122, 4.2440681
22: -27.0059948, -19.5266075, -27.0000725, -19.5586758, -4.3180332, 4.3539886
23: -20.8682251, -12.5019474, -20.8102341, -12.5254612, -4.7943726, 4.7491837
24: -16.8734665, -7.6346040, -16.8202114, -7.6561999, -7.1697464, 7.1367645
25: -14.6409473, -6.9530144, -14.6145391, -6.9667692, -4.1954899, 4.1729164
26: -14.6209106, -7.7976198, -14.6119814, -7.8212638, -6.5248108, 6.5552330
27: -14.6382122, -9.5233068, -14.6146107, -9.5547371, -4.0514927, 4.0610428
28: -10.0343723, -1.4231758, -10.0049353, -1.4288294, -6.1428108, 6.1423149
29: -45.5896301, -36.8078995, -45.5676727, -36.8505783, -4.9922104, 5.0146351
30: -32.2085609, -22.9965286, -32.1540794, -23.0345001, -4.9963284, 4.9929962
31: -32.2517738, -23.5114975, -32.2014732, -23.5324326, -6.3094482, 6.2709045
32: 7.6976581, 13.6747322, 7.7266674, 13.6796055, -4.1782475, 4.1266003
33: 4.5893106, 16.3115711, 4.6523628, 16.3153458, -6.7281456, 6.6244164
34: 20.5200005, 30.9889946, 20.5782528, 30.9871483, -5.7619991, 5.6806583
35: 16.4825993, 26.8635826, 16.5535049, 26.8587723, -5.4698963, 5.3818932
36: 28.7958012, 35.1248474, 28.8284512, 35.1274529, -3.4536228, 3.4020767
37: 11.0022554, 20.1163216, 11.0714273, 20.1121216, -5.9870987, 5.8977699
38: 34.8398132, 43.6985664, 34.9025879, 43.6731949, -6.0644836, 6.0010986
39: 8.9646778, 18.5169983, 9.0275612, 18.5017967, -6.5607147, 6.4964943
40: 15.7659302, 25.1276417, 15.8190880, 25.1328545, -5.8327637, 5.7621746
41: 6.7120681, 13.2260218, 6.7549877, 13.2259598, -5.0355072, 4.9793510
42: -12.3980570, -3.4538286, -12.3752909, -3.4460938, -7.0588379, 7.0141907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1769

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6332459
time: 5.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6374973
time: 5.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5781021, -8.4788551, -21.5575657, -8.4828711, -10.4486160, 10.4198380
1: -21.4303780, -12.2310791, -21.4387474, -12.2481756, -5.2710648, 5.2944527
2: -12.3956661, -5.7774386, -12.3976717, -5.7846122, -4.2688599, 4.2794056
3: -12.0103178, -4.1644087, -12.0024109, -4.1835532, -5.3552399, 5.3657684
4: -10.2874737, 0.0160040, -10.2854366, -0.0160222, -6.0185204, 6.0590839
5: -13.5587149, -4.0415139, -13.5629740, -4.0609560, -6.1370506, 6.1602097
6: -8.3385096, 0.5390713, -8.3091440, 0.5379272, -6.4683571, 6.4335327
7: -32.1568871, -22.0499954, -32.1567993, -22.0902367, -5.8161240, 5.8596859
8: -18.8056488, -9.1007957, -18.8056774, -9.1176910, -5.1955681, 5.2172413
9: -5.3422594, 1.3966579, -5.3220520, 1.3896229, -4.0497723, 4.0328331
10: -36.1350098, -27.7537746, -36.1357193, -27.7852802, -5.2427368, 5.2688198
11: -55.1323166, -44.7844963, -55.1096420, -44.8585472, -4.9058075, 4.9689693
12: -11.5809269, -4.5859241, -11.5701399, -4.5982966, -6.2182236, 6.2053070
13: 0.8842248, 8.0159569, 0.8965760, 7.9951043, -5.2826958, 5.2943878
14: -71.0843048, -57.9474068, -71.1002045, -57.9766693, -8.2371559, 8.2838936
15: -8.9112701, 0.9151077, -8.9146233, 0.8894377, -4.8590298, 4.9015236
16: -33.5677299, -23.9756050, -33.5436783, -24.0078697, -6.4502029, 6.4409027
17: -88.6775436, -72.3879013, -88.6782990, -72.4832687, -8.1518021, 8.2557602
18: -4.1777048, 1.0691819, -4.1673174, 1.0436270, -3.3761158, 3.3914509
19: -30.5249138, -23.2034569, -30.5065117, -23.2277412, -4.6377258, 4.6438866
20: -11.1720009, -5.1537991, -11.1622334, -5.1640692, -4.9303093, 4.9320011
21: -43.5461426, -35.0551033, -43.5262451, -35.0901413, -4.2476501, 4.2570553
22: -27.0050240, -19.5271416, -26.9990940, -19.5655327, -4.3194389, 4.3556976
23: -20.8561096, -12.5095081, -20.8251686, -12.5319643, -4.7728882, 4.7600479
24: -16.8623276, -7.6413236, -16.8429832, -7.6584578, -7.1512871, 7.1511841
25: -14.6402349, -6.9547219, -14.6149569, -6.9773302, -4.1932430, 4.1893444
26: -14.6176291, -7.8003669, -14.6087284, -7.8215451, -6.5463715, 6.5579681
27: -14.6305561, -9.5279999, -14.6159077, -9.5664682, -4.0426350, 4.0656471
28: -10.0230169, -1.4303138, -10.0109835, -1.4280664, -6.1501923, 6.1415596
29: -45.5838127, -36.8120422, -45.5743866, -36.8670959, -4.9822998, 5.0310287
30: -32.1860123, -23.0103569, -32.1724129, -23.0475998, -4.9587746, 4.9937668
31: -32.2408981, -23.5151539, -32.2111969, -23.5477066, -6.3039589, 6.3031120
32: 7.7008858, 13.6733494, 7.7246017, 13.6801348, -4.1791019, 4.1420841
33: 4.5882063, 16.3115215, 4.6514268, 16.3175011, -6.7260571, 6.6410065
34: 20.5269756, 30.9848976, 20.5897636, 30.9726181, -5.7557373, 5.6992626
35: 16.4906235, 26.8594131, 16.5581551, 26.8508625, -5.4625015, 5.3963127
36: 28.7973709, 35.1240768, 28.8394184, 35.1204529, -3.4551430, 3.4141865
37: 11.0066671, 20.1137829, 11.0621481, 20.1126556, -5.9784775, 5.9161949
38: 34.8495407, 43.6899261, 34.9186249, 43.6720734, -6.0574913, 6.0041199
39: 8.9733000, 18.5077343, 9.0353279, 18.5060101, -6.5506592, 6.4861031
40: 15.7735825, 25.1259422, 15.8176908, 25.1392059, -5.8332748, 5.7664719
41: 6.7167492, 13.2225609, 6.7445178, 13.2269545, -5.0289383, 4.9923897
42: -12.3983374, -3.4552188, -12.3679466, -3.4564207, -7.0582275, 7.0260429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6242553
time: 4.81 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6316467
time: 5.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5781021, -8.4788551, -21.5668736, -8.4800482, -10.4518814, 10.4296494
1: -21.4303780, -12.2310791, -21.4404831, -12.2458649, -5.2734947, 5.2956867
2: -12.3956661, -5.7774386, -12.4006672, -5.7817583, -4.2703209, 4.2783756
3: -12.0103178, -4.1644087, -12.0064087, -4.1747160, -5.3639641, 5.3697548
4: -10.2874737, 0.0160040, -10.2961206, 0.0048765, -6.0394669, 6.0695763
5: -13.5587149, -4.0415139, -13.5694857, -4.0536203, -6.1370010, 6.1614418
6: -8.3385096, 0.5390713, -8.3235226, 0.5432085, -6.4739571, 6.4479713
7: -32.1568871, -22.0499954, -32.1577988, -22.0916500, -5.8208008, 5.8628502
8: -18.8056488, -9.1007957, -18.8172379, -9.0819645, -5.2309418, 5.2288361
9: -5.3422594, 1.3966579, -5.3265719, 1.3992379, -4.0592842, 4.0372219
10: -36.1350098, -27.7537746, -36.1411972, -27.7788143, -5.2453346, 5.2736130
11: -55.1323166, -44.7844963, -55.1448517, -44.8450546, -4.9194660, 5.0043125
12: -11.5809269, -4.5859241, -11.5722342, -4.5908079, -6.2256927, 6.2060509
13: 0.8842248, 8.0159569, 0.8905485, 8.0144196, -5.3022957, 5.3005791
14: -71.0843048, -57.9474068, -71.1042862, -57.9698105, -8.2445412, 8.2889557
15: -8.9112701, 0.9151077, -8.9275780, 0.9110227, -4.8805866, 4.9147816
16: -33.5677299, -23.9756050, -33.5671082, -23.9994545, -6.4589691, 6.4642067
17: -88.6775436, -72.3879013, -88.6785812, -72.4802246, -8.1583366, 8.2560081
18: -4.1777048, 1.0691819, -4.1909728, 1.0485427, -3.3810177, 3.4150848
19: -30.5249138, -23.2034569, -30.5155487, -23.2248268, -4.6407509, 4.6532841
20: -11.1720009, -5.1537991, -11.1643181, -5.1628656, -4.9281387, 4.9333954
21: -43.5461426, -35.0551033, -43.5430450, -35.0833435, -4.2545204, 4.2738972
22: -27.0050240, -19.5271416, -27.0025387, -19.5648956, -4.3194866, 4.3594761
23: -20.8561096, -12.5095081, -20.8446732, -12.5229340, -4.7826252, 4.7795753
24: -16.8623276, -7.6413236, -16.8628826, -7.6509871, -7.1587830, 7.1710052
25: -14.6402349, -6.9547219, -14.6189976, -6.9746609, -4.1961117, 4.1935883
26: -14.6176291, -7.8003669, -14.6131105, -7.8181305, -6.5479050, 6.5624771
27: -14.6305561, -9.5279999, -14.6272936, -9.5611753, -4.0479946, 4.0764580
28: -10.0230169, -1.4303138, -10.0257921, -1.4195311, -6.1539116, 6.1432419
29: -45.5838127, -36.8120422, -45.5873299, -36.8628502, -4.9865532, 5.0438786
30: -32.1860123, -23.0103569, -32.2047234, -23.0319252, -4.9750595, 5.0260925
31: -32.2408981, -23.5151539, -32.2305565, -23.5432892, -6.3083572, 6.3216553
32: 7.7008858, 13.6733494, 7.7196436, 13.6823416, -4.1812840, 4.1471767
33: 4.5882063, 16.3115215, 4.6487021, 16.3187828, -6.7272911, 6.6466827
34: 20.5269756, 30.9848976, 20.5774841, 30.9776115, -5.7604294, 5.7116356
35: 16.4906235, 26.8594131, 16.5442982, 26.8559837, -5.4675751, 5.4100990
36: 28.7973709, 35.1240768, 28.8372974, 35.1220856, -3.4565115, 3.4162312
37: 11.0066671, 20.1137829, 11.0526733, 20.1161880, -5.9820480, 5.9259949
38: 34.8495407, 43.6899261, 34.9078751, 43.6859741, -6.0713615, 6.0152512
39: 8.9733000, 18.5077343, 9.0260067, 18.5177498, -6.5600548, 6.4955864
40: 15.7735825, 25.1259422, 15.8067713, 25.1440849, -5.8381691, 5.7772865
41: 6.7167492, 13.2225609, 6.7356405, 13.2313128, -5.0333099, 5.0012169
42: -12.3983374, -3.4552188, -12.3679533, -3.4537067, -7.0609589, 7.0267372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6307889
time: 5.62 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6381817
time: 5.89 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5838375, -8.4760742, -21.5474911, -8.4833565, -10.4860001, 10.4052582
1: -21.4268017, -12.2288179, -21.4237309, -12.2570524, -5.2647705, 5.2787876
2: -12.3964128, -5.7746916, -12.3928432, -5.7880578, -4.2760239, 4.2718182
3: -12.0110798, -4.1559644, -11.9967308, -4.1896458, -5.3493462, 5.3689690
4: -10.2920723, 0.0363891, -10.2762089, -0.0291009, -6.0100060, 6.0712509
5: -13.5623865, -4.0344014, -13.5592480, -4.0667191, -6.1402512, 6.1519547
6: -8.3521137, 0.5389824, -8.2976503, 0.5277033, -6.4726257, 6.4227943
7: -32.1531563, -22.0514946, -32.1439705, -22.1004906, -5.8073502, 5.8508663
8: -18.8133812, -9.0653715, -18.8059864, -9.1252918, -5.1948357, 5.2529621
9: -5.3438401, 1.4061282, -5.3144493, 1.3839298, -4.0447979, 4.0339565
10: -36.1357040, -27.7474442, -36.1250229, -27.7933388, -5.2343597, 5.2588310
11: -55.1664848, -44.7712936, -55.1069527, -44.8480453, -4.9491577, 4.9780693
12: -11.5825930, -4.5787020, -11.5683765, -4.5954232, -6.2208366, 6.2104073
13: 0.8817129, 8.0348959, 0.9021168, 7.9884820, -5.2786293, 5.3084145
14: -71.0820007, -57.9406357, -71.0832214, -57.9842911, -8.2378159, 8.2701950
15: -8.9174709, 0.9364209, -8.9050636, 0.8763442, -4.8509941, 4.9138584
16: -33.5891266, -23.9672909, -33.5366440, -23.9998360, -6.4794807, 6.4391174
17: -88.6684570, -72.3856735, -88.6519928, -72.5025406, -8.1259804, 8.2355385
18: -4.2001371, 1.0738747, -4.1642928, 1.0448816, -3.3998852, 3.3929577
19: -30.5337601, -23.2016602, -30.5051384, -23.2277222, -4.6479359, 4.6458015
20: -11.1740179, -5.1551142, -11.1589699, -5.1705856, -4.9244537, 4.9359989
21: -43.5625763, -35.0493622, -43.5245895, -35.0867500, -4.2672939, 4.2616215
22: -27.0058155, -19.5266953, -26.9929657, -19.5714493, -4.3157310, 4.3497639
23: -20.8752708, -12.5039425, -20.8184052, -12.5353394, -4.7889786, 4.7588425
24: -16.8817863, -7.6372252, -16.8370914, -7.6611743, -7.1683197, 7.1493340
25: -14.6440887, -6.9549170, -14.6098719, -6.9845963, -4.1903934, 4.1838799
26: -14.6209755, -7.7980275, -14.6086912, -7.8245983, -6.5383339, 6.5794525
27: -14.6415129, -9.5229921, -14.6161671, -9.5637264, -4.0552883, 4.0702801
28: -10.0376511, -1.4248657, -10.0071974, -1.4298108, -6.1474457, 6.1476135
29: -45.5936127, -36.8080444, -45.5659828, -36.8692017, -4.9908333, 5.0247078
30: -32.2180176, -22.9961948, -32.1694832, -23.0404739, -4.9986687, 5.0062294
31: -32.2597198, -23.5143852, -32.2049065, -23.5535889, -6.3191452, 6.2966957
32: 7.6963968, 13.6720028, 7.7330198, 13.6719723, -4.1757774, 4.1325397
33: 4.5860834, 16.3069630, 4.6669784, 16.3014507, -6.7151337, 6.6220627
34: 20.5152855, 30.9857712, 20.5981693, 30.9656868, -5.7603912, 5.6914978
35: 16.4771767, 26.8598003, 16.5683746, 26.8430023, -5.4680786, 5.3863544
36: 28.7955551, 35.1223869, 28.8461609, 35.1121178, -3.4485989, 3.4067888
37: 10.9978514, 20.1130028, 11.0738297, 20.1032600, -5.9784126, 5.9032173
38: 34.8391647, 43.7000809, 34.9189072, 43.6612930, -6.0572052, 6.0144501
39: 8.9645805, 18.5147705, 9.0412750, 18.4929276, -6.5461655, 6.4845734
40: 15.7631521, 25.1264000, 15.8271093, 25.1272850, -5.8305130, 5.7610416
41: 6.7085056, 13.2231932, 6.7531557, 13.2204943, -5.0308418, 4.9844017
42: -12.3980169, -3.4566917, -12.3597832, -3.4675467, -7.0481949, 7.0221863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 535

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5827478, upper bound: 3.6374971
time: 5.90 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5892050, upper bound: 3.6374971
time: 5.01 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5871468, -8.4760504, -21.5588932, -8.4801388, -10.4849319, 10.4310455
1: -21.4317951, -12.2287779, -21.4387474, -12.2481365, -5.2782936, 5.2815475
2: -12.3985348, -5.7745876, -12.3992319, -5.7846093, -4.2798309, 4.2748699
3: -12.0141163, -4.1556058, -12.0052786, -4.1824937, -5.3599586, 5.3693085
4: -10.2978401, 0.0368910, -10.2935038, -0.0157654, -6.0287361, 6.0748940
5: -13.5650082, -4.0341997, -13.5667820, -4.0600300, -6.1495743, 6.1520576
6: -8.3528175, 0.5440377, -8.3099442, 0.5420675, -6.4814262, 6.4393616
7: -32.1575623, -22.0514336, -32.1563568, -22.0914612, -5.8211479, 5.8518677
8: -18.8170567, -9.0650625, -18.8164959, -9.1171684, -5.2069607, 5.2542019
9: -5.3466206, 1.4062850, -5.3253126, 1.3903956, -4.0548325, 4.0366001
10: -36.1402130, -27.7473354, -36.1377754, -27.7838821, -5.2498741, 5.2573204
11: -55.1672516, -44.7710266, -55.1092224, -44.8447571, -4.9550991, 4.9801540
12: -11.5829849, -4.5784841, -11.5702610, -4.5942335, -6.2226257, 6.2146873
13: 0.8784010, 8.0352507, 0.8912199, 7.9970307, -5.2903023, 5.3136444
14: -71.0879364, -57.9405365, -71.1001968, -57.9732513, -8.2563553, 8.2701416
15: -8.9237976, 0.9366622, -8.9230146, 0.8898787, -4.8720798, 4.9147892
16: -33.5907478, -23.9672394, -33.5435944, -23.9989948, -6.4800377, 6.4543419
17: -88.6772461, -72.3849335, -88.6767197, -72.4813080, -8.1557770, 8.2423973
18: -4.2012939, 1.0740733, -4.1679454, 1.0488513, -3.4048595, 3.3955460
19: -30.5338936, -23.2006264, -30.5080376, -23.2248383, -4.6503048, 4.6471100
20: -11.1740637, -5.1527362, -11.1633110, -5.1630268, -4.9311256, 4.9371223
21: -43.5628891, -35.0484047, -43.5272598, -35.0838470, -4.2691498, 4.2646618
22: -27.0083466, -19.5265179, -27.0003166, -19.5654373, -4.3241806, 4.3526669
23: -20.8755779, -12.5007210, -20.8262978, -12.5250912, -4.7914619, 4.7701797
24: -16.8821831, -7.6340547, -16.8447266, -7.6512175, -7.1737404, 7.1597748
25: -14.6442699, -6.9522309, -14.6165314, -6.9760571, -4.1921158, 4.1934853
26: -14.6217909, -7.7970419, -14.6114645, -7.8198085, -6.5509605, 6.5829849
27: -14.6419001, -9.5227995, -14.6179314, -9.5619411, -4.0579338, 4.0733433
28: -10.0377474, -1.4219873, -10.0124912, -1.4215581, -6.1563492, 6.1490135
29: -45.5965652, -36.8078423, -45.5746078, -36.8624802, -4.9994869, 5.0276890
30: -32.2183075, -22.9948025, -32.1728897, -23.0352650, -5.0005016, 5.0099030
31: -32.2601891, -23.5109787, -32.2136002, -23.5435867, -6.3226929, 6.3086548
32: 7.6959648, 13.6753321, 7.7242036, 13.6814251, -4.1806889, 4.1443634
33: 4.5855026, 16.3124619, 4.6507797, 16.3170052, -6.7186966, 6.6443367
34: 20.5148048, 30.9896049, 20.5886765, 30.9766960, -5.7663403, 5.7046165
35: 16.4768162, 26.8642292, 16.5565033, 26.8554344, -5.4720783, 5.4026794
36: 28.7952881, 35.1255188, 28.8381310, 35.1209106, -3.4507999, 3.4178944
37: 10.9972439, 20.1170654, 11.0605659, 20.1146832, -5.9825821, 5.9207077
38: 34.8388367, 43.7035980, 34.9101830, 43.6718788, -6.0611992, 6.0262833
39: 8.9640331, 18.5191917, 9.0264950, 18.5053253, -6.5499954, 6.5037079
40: 15.7627220, 25.1305332, 15.8150063, 25.1390915, -5.8330631, 5.7776871
41: 6.7079449, 13.2266426, 6.7435780, 13.2301245, -5.0370369, 4.9972229
42: -12.3983021, -3.4527769, -12.3682947, -3.4564810, -7.0534286, 7.0340195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 535

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5901413, upper bound: 3.6374971
time: 4.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5965988, upper bound: 3.6374971
time: 5.93 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5838375, -8.4760742, -21.5519924, -8.4825897, -10.4861603, 10.4089966
1: -21.4268017, -12.2288179, -21.4270515, -12.2535915, -5.2662849, 5.2802181
2: -12.3964128, -5.7746916, -12.3982563, -5.7860742, -4.2718658, 4.2711220
3: -12.0110798, -4.1559644, -12.0092449, -4.1802311, -5.3462830, 5.3697586
4: -10.2920723, 0.0363891, -10.2851849, -0.0259869, -6.0073433, 6.0730629
5: -13.5623865, -4.0344014, -13.5615282, -4.0635214, -6.1428108, 6.1537857
6: -8.3521137, 0.5389824, -8.3030729, 0.5370712, -6.4753075, 6.4218941
7: -32.1531563, -22.0514946, -32.1502266, -22.0922432, -5.8115730, 5.8543472
8: -18.8133812, -9.0653715, -18.8138809, -9.1197929, -5.1947556, 5.2542000
9: -5.3438401, 1.4061282, -5.3223438, 1.3863189, -4.0493011, 4.0451603
10: -36.1357040, -27.7474442, -36.1339989, -27.7795906, -5.2343884, 5.2553501
11: -55.1664848, -44.7712936, -55.1263847, -44.8124924, -4.9469872, 4.9590225
12: -11.5825930, -4.5787020, -11.5756168, -4.5861325, -6.2235527, 6.2127571
13: 0.8817129, 8.0348959, 0.8923808, 7.9930806, -5.2782440, 5.3119049
14: -71.0820007, -57.9406357, -71.0859680, -57.9815712, -8.2427864, 8.2748375
15: -8.9174709, 0.9364209, -8.9122276, 0.8787646, -4.8447781, 4.9097366
16: -33.5891266, -23.9672909, -33.5566750, -23.9771957, -6.4822884, 6.4384804
17: -88.6684570, -72.3856735, -88.6629486, -72.4840088, -8.1285629, 8.2328682
18: -4.2001371, 1.0738747, -4.1672220, 1.0499740, -3.4036102, 3.3949566
19: -30.5337601, -23.2016602, -30.5189323, -23.2089672, -4.6455956, 4.6386738
20: -11.1740179, -5.1551142, -11.1651163, -5.1609173, -4.9257851, 4.9342842
21: -43.5625763, -35.0493622, -43.5419159, -35.0599518, -4.2592621, 4.2449741
22: -27.0058155, -19.5266953, -26.9995518, -19.5612068, -4.3180084, 4.3482056
23: -20.8752708, -12.5039425, -20.8253651, -12.5213442, -4.7901573, 4.7540016
24: -16.8817863, -7.6372252, -16.8409023, -7.6512671, -7.1729698, 7.1491776
25: -14.6440887, -6.9549170, -14.6191769, -6.9681592, -4.1908112, 4.1770668
26: -14.6209755, -7.7980275, -14.6108694, -7.8209400, -6.5409203, 6.5803490
27: -14.6415129, -9.5229921, -14.6261139, -9.5481672, -4.0556087, 4.0664062
28: -10.0376511, -1.4248657, -10.0101566, -1.4268898, -6.1509895, 6.1507034
29: -45.5936127, -36.8080444, -45.5788345, -36.8460464, -4.9919624, 5.0151291
30: -32.2180176, -22.9961948, -32.1804657, -23.0188866, -5.0013542, 4.9972153
31: -32.2597198, -23.5143852, -32.2201691, -23.5317516, -6.3152847, 6.2881165
32: 7.6963968, 13.6720028, 7.7294531, 13.6725426, -4.1760025, 4.1358814
33: 4.5860834, 16.3069630, 4.6554079, 16.3047009, -6.7147942, 6.6337662
34: 20.5152855, 30.9857712, 20.5707760, 30.9842911, -5.7518654, 5.6911850
35: 16.4771767, 26.8598003, 16.5459328, 26.8556175, -5.4583187, 5.3873539
36: 28.7955551, 35.1223869, 28.8324490, 35.1207008, -3.4459457, 3.4070854
37: 10.9978514, 20.1130028, 11.0673189, 20.1071739, -5.9795265, 5.9082069
38: 34.8391647, 43.7000809, 34.8977242, 43.6765823, -6.0575371, 6.0195618
39: 8.9645805, 18.5147705, 9.0320015, 18.4937820, -6.5488930, 6.4953117
40: 15.7631521, 25.1264000, 15.8160601, 25.1290703, -5.8300781, 5.7710285
41: 6.7085056, 13.2231932, 6.7501192, 13.2234240, -5.0323868, 4.9863739
42: -12.3980169, -3.4566917, -12.3679895, -3.4546616, -7.0499229, 7.0198746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 535

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5827478, upper bound: 3.6374974
time: 5.18 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5892050, upper bound: 3.6374974
time: 4.98 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5871468, -8.4760504, -21.5633850, -8.4793863, -10.4850998, 10.4347839
1: -21.4317951, -12.2287779, -21.4420433, -12.2446709, -5.2798100, 5.2829704
2: -12.3985348, -5.7745876, -12.4046326, -5.7826629, -4.2756729, 4.2741776
3: -12.0141163, -4.1556058, -12.0177965, -4.1730700, -5.3568878, 5.3701057
4: -10.2978401, 0.0368910, -10.3024540, -0.0126201, -6.0260582, 6.0767059
5: -13.5650082, -4.0341997, -13.5690422, -4.0568419, -6.1521530, 6.1538887
6: -8.3528175, 0.5440377, -8.3153582, 0.5514654, -6.4840813, 6.4384422
7: -32.1575623, -22.0514336, -32.1626282, -22.0832138, -5.8253708, 5.8553410
8: -18.8170567, -9.0650625, -18.8244076, -9.1116285, -5.2068901, 5.2554398
9: -5.3466206, 1.4062850, -5.3331938, 1.3927984, -4.0593300, 4.0477886
10: -36.1402130, -27.7473354, -36.1467667, -27.7700996, -5.2499161, 5.2538395
11: -55.1672516, -44.7710266, -55.1286240, -44.8092346, -4.9529343, 4.9611111
12: -11.5829849, -4.5784841, -11.5774860, -4.5849566, -6.2253380, 6.2170486
13: 0.8784010, 8.0352507, 0.8814968, 8.0016289, -5.2899094, 5.3171349
14: -71.0879364, -57.9405365, -71.1029358, -57.9705048, -8.2613106, 8.2747879
15: -8.9237976, 0.9366622, -8.9301624, 0.8922954, -4.8658676, 4.9106636
16: -33.5907478, -23.9672394, -33.5636024, -23.9763393, -6.4828453, 6.4537125
17: -88.6772461, -72.3849335, -88.6877060, -72.4627228, -8.1583824, 8.2397270
18: -4.2012939, 1.0740733, -4.1709132, 1.0539563, -3.4085865, 3.3975391
19: -30.5338936, -23.2006264, -30.5217991, -23.2060814, -4.6479607, 4.6399822
20: -11.1740637, -5.1527362, -11.1694651, -5.1533070, -4.9324684, 4.9354019
21: -43.5628891, -35.0484047, -43.5445290, -35.0570221, -4.2611160, 4.2480202
22: -27.0083466, -19.5265179, -27.0068893, -19.5551929, -4.3264580, 4.3511086
23: -20.8755779, -12.5007210, -20.8332634, -12.5110922, -4.7926292, 4.7653465
24: -16.8821831, -7.6340547, -16.8485413, -7.6413093, -7.1783829, 7.1596413
25: -14.6442699, -6.9522309, -14.6258259, -6.9596310, -4.1925335, 4.1866856
26: -14.6217909, -7.7970419, -14.6136503, -7.8161321, -6.5535355, 6.5838966
27: -14.6419001, -9.5227995, -14.6278744, -9.5463848, -4.0582504, 4.0694714
28: -10.0377474, -1.4219873, -10.0154343, -1.4186246, -6.1599236, 6.1521034
29: -45.5965652, -36.8078423, -45.5874596, -36.8393326, -5.0006237, 5.0181084
30: -32.2183075, -22.9948025, -32.1838531, -23.0137196, -5.0031872, 5.0008869
31: -32.2601891, -23.5109787, -32.2288284, -23.5217552, -6.3188324, 6.3000870
32: 7.6959648, 13.6753321, 7.7206144, 13.6819839, -4.1809063, 4.1477127
33: 4.5855026, 16.3124619, 4.6390867, 16.3202400, -6.7183628, 6.6560326
34: 20.5148048, 30.9896049, 20.5612335, 30.9953156, -5.7578087, 5.7043247
35: 16.4768162, 26.8642292, 16.5340748, 26.8680649, -5.4623184, 5.4036827
36: 28.7952881, 35.1255188, 28.8244152, 35.1294937, -3.4481392, 3.4181948
37: 10.9972439, 20.1170654, 11.0540447, 20.1185741, -5.9837074, 5.9257011
38: 34.8388367, 43.7035980, 34.8890228, 43.6871338, -6.0615387, 6.0313835
39: 8.9640331, 18.5191917, 9.0171928, 18.5061798, -6.5527191, 6.5144501
40: 15.7627220, 25.1305332, 15.8039246, 25.1408501, -5.8326283, 5.7876759
41: 6.7079449, 13.2266426, 6.7405543, 13.2330618, -5.0385895, 4.9991798
42: -12.3983021, -3.4527769, -12.3764849, -3.4435911, -7.0551643, 7.0317154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 535

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5901413, upper bound: 3.6374974
time: 4.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5965988, upper bound: 3.6374974
time: 4.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -21.5579834, -8.4811563, -21.5679703, -8.4849968, -10.3779984, 10.3881760
1: -21.4219856, -12.2365484, -21.4330368, -12.2429752, -5.2604065, 5.2784271
2: -12.3915615, -5.7777085, -12.3970137, -5.7820106, -4.2516594, 4.2615948
3: -12.0071468, -4.1688685, -11.9988441, -4.1900616, -5.3500595, 5.3633575
4: -10.2769680, 0.0037479, -10.2780800, -0.0219492, -6.0203667, 6.0519867
5: -13.5527449, -4.0461941, -13.5626421, -4.0566649, -6.1249161, 6.1405258
6: -8.3162031, 0.5253098, -8.2925739, 0.5265012, -6.4545403, 6.4274712
7: -32.1505814, -22.0557404, -32.1576996, -22.0683174, -5.8139687, 5.8328876
8: -18.8011246, -9.1073551, -18.7963161, -9.1539602, -5.1830521, 5.2271919
9: -5.3288035, 1.3930535, -5.3294868, 1.3802533, -4.0309143, 4.0448704
10: -36.1260567, -27.7660561, -36.1269646, -27.7834549, -5.2434235, 5.2596512
11: -55.1253281, -44.8033066, -55.0775528, -44.8422432, -4.9321861, 4.9320316
12: -11.5745134, -4.5910568, -11.5606747, -4.5935268, -6.2369347, 6.2256737
13: 0.8940356, 8.0107975, 0.9013391, 7.9844122, -5.2710304, 5.2912903
14: -71.0678329, -57.9613991, -71.0888062, -57.9643173, -8.2146683, 8.2422829
15: -8.8990755, 0.9011693, -8.9017487, 0.8785944, -4.8504963, 4.8815117
16: -33.5529327, -23.9759464, -33.5236397, -24.0038528, -6.4362946, 6.4371605
17: -88.6560822, -72.4222717, -88.6654892, -72.4461136, -8.1429291, 8.1800995
18: -4.1735067, 1.0583792, -4.1485982, 1.0500364, -3.3882084, 3.3707523
19: -30.5235023, -23.2042046, -30.5005798, -23.2254486, -4.6473389, 4.6454811
20: -11.1709690, -5.1588526, -11.1651707, -5.1633296, -4.9249420, 4.9210777
21: -43.5443840, -35.0580826, -43.5126877, -35.0892105, -4.2624531, 4.2608757
22: -26.9977493, -19.5396614, -26.9931145, -19.5526733, -4.3215580, 4.3305073
23: -20.8500271, -12.5152779, -20.8215218, -12.5363426, -4.7659187, 4.7575417
24: -16.8572483, -7.6482162, -16.8278732, -7.6655345, -7.1451416, 7.1353798
25: -14.6360760, -6.9600935, -14.6172752, -6.9818368, -4.1860104, 4.1885567
26: -14.6093721, -7.8215179, -14.6084795, -7.8230209, -6.5136223, 6.5141449
27: -14.6275578, -9.5355358, -14.6077700, -9.5544033, -4.0500507, 4.0486965
28: -10.0212345, -1.4322876, -10.0095654, -1.4325421, -6.1446495, 6.1375198
29: -45.5753860, -36.8263397, -45.5529099, -36.8557968, -4.9884377, 4.9973335
30: -32.1823769, -23.0205536, -32.1420631, -23.0461082, -4.9809570, 4.9774952
31: -32.2330093, -23.5217361, -32.1939430, -23.5494747, -6.2918015, 6.2792587
32: 7.7196803, 13.6643209, 7.7287741, 13.6766987, -4.1555882, 4.1319828
33: 4.6117835, 16.3025074, 4.6348710, 16.3101654, -6.6816635, 6.6499557
34: 20.5370865, 30.9774494, 20.5801067, 30.9698429, -5.7377892, 5.7006969
35: 16.5063782, 26.8521919, 16.5462265, 26.8450661, -5.4335041, 5.3981209
36: 28.8109665, 35.1159744, 28.8264122, 35.1144905, -3.4248323, 3.4141798
37: 11.0274429, 20.1061268, 11.0491848, 20.1071854, -5.9489403, 5.9233208
38: 34.8639641, 43.6769333, 34.8896561, 43.6574020, -6.0201874, 6.0137177
39: 8.9996548, 18.4968891, 9.0095472, 18.4988823, -6.5064240, 6.4905128
40: 15.7941332, 25.1175671, 15.8171120, 25.1276131, -5.7997589, 5.7634640
41: 6.7347732, 13.2131405, 6.7507710, 13.2167454, -5.0080986, 4.9874496
42: -12.3857851, -3.4645004, -12.3771009, -3.4650826, -7.0472069, 7.0377655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6260930
time: 4.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6334849
time: 5.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.5579834, -8.4811563, -21.5848198, -8.4798470, -10.4171524, 10.4394989
1: -21.4219856, -12.2365484, -21.4365387, -12.2401314, -5.2780743, 5.2962017
2: -12.3915615, -5.7777085, -12.4002247, -5.7799158, -4.2692833, 4.2776642
3: -12.0071468, -4.1688685, -12.0052891, -4.1770725, -5.3566246, 5.3636475
4: -10.2769680, 0.0037479, -10.2933006, 0.0010507, -6.0211983, 6.0449371
5: -13.5527449, -4.0461941, -13.5695114, -4.0485811, -6.1362305, 6.1526070
6: -8.3162031, 0.5253098, -8.3117342, 0.5354959, -6.4598961, 6.4423370
7: -32.1505814, -22.0557404, -32.1593552, -22.0636520, -5.8243523, 5.8396664
8: -18.8011246, -9.1073551, -18.8150597, -9.1123924, -5.1960907, 5.2179718
9: -5.3288035, 1.3930535, -5.3375883, 1.3913524, -4.0327682, 4.0439529
10: -36.1260567, -27.7660561, -36.1333847, -27.7783470, -5.2487869, 5.2687683
11: -55.1253281, -44.8033066, -55.1070633, -44.8250465, -4.9232922, 4.9352798
12: -11.5745134, -4.5910568, -11.5674973, -4.5855122, -6.2389297, 6.2257462
13: 0.8940356, 8.0107975, 0.8908373, 8.0084352, -5.2879982, 5.2945786
14: -71.0678329, -57.9613991, -71.0921555, -57.9609985, -8.2378578, 8.2617683
15: -8.8990755, 0.9011693, -8.9170990, 0.8989048, -4.8533192, 4.8801537
16: -33.5529327, -23.9759464, -33.5503540, -23.9903507, -6.4392319, 6.4527359
17: -88.6560822, -72.4222717, -88.6673355, -72.4370117, -8.1551819, 8.1831551
18: -4.1735067, 1.0583792, -4.1749363, 1.0584300, -3.3832836, 3.3839226
19: -30.5235023, -23.2042046, -30.5123615, -23.2201767, -4.6450996, 4.6483974
20: -11.1709690, -5.1588526, -11.1661997, -5.1600647, -4.9372292, 4.9340820
21: -43.5443840, -35.0580826, -43.5300789, -35.0790405, -4.2573071, 4.2628365
22: -26.9977493, -19.5396614, -27.0002861, -19.5497589, -4.3204327, 4.3325710
23: -20.8500271, -12.5152779, -20.8445320, -12.5216122, -4.7652245, 4.7643623
24: -16.8572483, -7.6482162, -16.8561382, -7.6502113, -7.1499825, 7.1530037
25: -14.6360760, -6.9600935, -14.6285496, -6.9744334, -4.1866684, 4.1927853
26: -14.6093721, -7.8215179, -14.6105719, -7.8179474, -6.5406151, 6.5427513
27: -14.6275578, -9.5355358, -14.6210728, -9.5463486, -4.0523148, 4.0561447
28: -10.0212345, -1.4322876, -10.0200129, -1.4220256, -6.1644058, 6.1546707
29: -45.5753860, -36.8263397, -45.5731659, -36.8451042, -4.9889927, 5.0072803
30: -32.1823769, -23.0205536, -32.1719017, -23.0259132, -4.9756660, 4.9807415
31: -32.2330093, -23.5217361, -32.2213097, -23.5392494, -6.2980232, 6.3014221
32: 7.7196803, 13.6643209, 7.7228394, 13.6795845, -4.1612530, 4.1408176
33: 4.6117835, 16.3025074, 4.6217232, 16.3157864, -6.6832104, 6.6555634
34: 20.5370865, 30.9774494, 20.5631657, 30.9785690, -5.7343063, 5.7055016
35: 16.5063782, 26.8521919, 16.5268993, 26.8549728, -5.4294548, 5.4034424
36: 28.8109665, 35.1159744, 28.8230095, 35.1169815, -3.4273434, 3.4161997
37: 11.0274429, 20.1061268, 11.0318642, 20.1141624, -5.9482880, 5.9333420
38: 34.8639641, 43.6769333, 34.8764076, 43.6718445, -6.0236015, 6.0161247
39: 8.9996548, 18.4968891, 8.9996738, 18.5038872, -6.5075111, 6.4996185
40: 15.7941332, 25.1175671, 15.8024569, 25.1361923, -5.8068314, 5.7765312
41: 6.7347732, 13.2131405, 6.7364345, 13.2243643, -5.0113373, 4.9972038
42: -12.3857851, -3.4645004, -12.3781538, -3.4620640, -7.0509644, 7.0390816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6260929
time: 4.91 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6334851
time: 4.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5781021, -8.4788551, -21.5679703, -8.4849968, -10.3989258, 10.3970718
1: -21.4303780, -12.2310791, -21.4330368, -12.2429752, -5.2705498, 5.2826900
2: -12.3956661, -5.7774386, -12.3970137, -5.7820106, -4.2551804, 4.2639122
3: -12.0103178, -4.1644087, -11.9988441, -4.1900616, -5.3528709, 5.3673935
4: -10.2874737, 0.0160040, -10.2780800, -0.0219492, -6.0308571, 6.0661201
5: -13.5587149, -4.0415139, -13.5626421, -4.0566649, -6.1320992, 6.1480103
6: -8.3385096, 0.5390713, -8.2925739, 0.5265012, -6.4764862, 6.4406013
7: -32.1568871, -22.0499954, -32.1576996, -22.0683174, -5.8219833, 5.8388844
8: -18.8056488, -9.1007957, -18.7963161, -9.1539602, -5.1890297, 5.2352390
9: -5.3422594, 1.3966579, -5.3294868, 1.3802533, -4.0448856, 4.0488071
10: -36.1350098, -27.7537746, -36.1269646, -27.7834549, -5.2525978, 5.2676430
11: -55.1323166, -44.7844963, -55.0775528, -44.8422432, -4.9404449, 4.9504299
12: -11.5809269, -4.5859241, -11.5606747, -4.5935268, -6.2396126, 6.2268639
13: 0.8842248, 8.0159569, 0.9013391, 7.9844122, -5.2810440, 5.2977829
14: -71.0843048, -57.9474068, -71.0888062, -57.9643173, -8.2310295, 8.2539711
15: -8.9112701, 0.9151077, -8.9017487, 0.8785944, -4.8629246, 4.8989067
16: -33.5677299, -23.9756050, -33.5236397, -24.0038528, -6.4532509, 6.4405899
17: -88.6775436, -72.3879013, -88.6654892, -72.4461136, -8.1664658, 8.2160683
18: -4.1777048, 1.0691819, -4.1485982, 1.0500364, -3.3929100, 3.3814735
19: -30.5249138, -23.2034569, -30.5005798, -23.2254486, -4.6497345, 4.6463337
20: -11.1720009, -5.1537991, -11.1651707, -5.1633296, -4.9265709, 4.9258995
21: -43.5461426, -35.0551033, -43.5126877, -35.0892105, -4.2652206, 4.2634163
22: -27.0050240, -19.5271416, -26.9931145, -19.5526733, -4.3296852, 4.3441906
23: -20.8561096, -12.5095081, -20.8215218, -12.5363426, -4.7745552, 4.7648792
24: -16.8623276, -7.6413236, -16.8278732, -7.6655345, -7.1499863, 7.1421661
25: -14.6402349, -6.9547219, -14.6172752, -6.9818368, -4.1914253, 4.1948128
26: -14.6176291, -7.8003669, -14.6084795, -7.8230209, -6.5225792, 6.5348129
27: -14.6305561, -9.5279999, -14.6077700, -9.5544033, -4.0535641, 4.0556679
28: -10.0230169, -1.4303138, -10.0095654, -1.4325421, -6.1483917, 6.1392555
29: -45.5838127, -36.8120422, -45.5529099, -36.8557968, -4.9981880, 5.0113029
30: -32.1860123, -23.0103569, -32.1420631, -23.0461082, -4.9847221, 4.9874496
31: -32.2408981, -23.5151539, -32.1939430, -23.5494747, -6.3023300, 6.2878380
32: 7.7008858, 13.6733494, 7.7287741, 13.6766987, -4.1749058, 4.1413517
33: 4.5882063, 16.3115215, 4.6348710, 16.3101654, -6.7103043, 6.6601448
34: 20.5269756, 30.9848976, 20.5801067, 30.9698429, -5.7495766, 5.7093277
35: 16.4906235, 26.8594131, 16.5462265, 26.8450661, -5.4519176, 5.4067249
36: 28.7973709, 35.1240768, 28.8264122, 35.1144905, -3.4389372, 3.4231081
37: 11.0066671, 20.1137829, 11.0491848, 20.1071854, -5.9723129, 5.9312592
38: 34.8495407, 43.6899261, 34.8896561, 43.6574020, -6.0345802, 6.0279579
39: 8.9733000, 18.5077343, 9.0095472, 18.4988823, -6.5333862, 6.5024529
40: 15.7735825, 25.1259422, 15.8171120, 25.1276131, -5.8215752, 5.7718163
41: 6.7167492, 13.2225609, 6.7507710, 13.2167454, -5.0270767, 4.9972610
42: -12.3983374, -3.4552188, -12.3771009, -3.4650826, -7.0590324, 7.0470734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5745822, upper bound: 3.6307845
time: 4.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5745822, upper bound: 3.6381770
time: 6.08 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5781021, -8.4788551, -21.5848198, -8.4798470, -10.4380798, 10.4483795
1: -21.4303780, -12.2310791, -21.4365387, -12.2401314, -5.2882175, 5.3004608
2: -12.3956661, -5.7774386, -12.4002247, -5.7799158, -4.2728081, 4.2799778
3: -12.0103178, -4.1644087, -12.0052891, -4.1770725, -5.3594360, 5.3676872
4: -10.2874737, 0.0160040, -10.2933006, 0.0010507, -6.0316849, 6.0590630
5: -13.5587149, -4.0415139, -13.5695114, -4.0485811, -6.1434174, 6.1600990
6: -8.3385096, 0.5390713, -8.3117342, 0.5354959, -6.4818420, 6.4554634
7: -32.1568871, -22.0499954, -32.1593552, -22.0636520, -5.8323708, 5.8456631
8: -18.8056488, -9.1007957, -18.8150597, -9.1123924, -5.2020645, 5.2260208
9: -5.3422594, 1.3966579, -5.3375883, 1.3913524, -4.0467377, 4.0478954
10: -36.1350098, -27.7537746, -36.1333847, -27.7783470, -5.2579689, 5.2767620
11: -55.1323166, -44.7844963, -55.1070633, -44.8250465, -4.9315491, 4.9536781
12: -11.5809269, -4.5859241, -11.5674973, -4.5855122, -6.2416153, 6.2269325
13: 0.8842248, 8.0159569, 0.8908373, 8.0084352, -5.2980118, 5.3010750
14: -71.0843048, -57.9474068, -71.0921555, -57.9609985, -8.2542152, 8.2734528
15: -8.9112701, 0.9151077, -8.9170990, 0.8989048, -4.8657455, 4.8975487
16: -33.5677299, -23.9756050, -33.5503540, -23.9903507, -6.4561806, 6.4561653
17: -88.6775436, -72.3879013, -88.6673355, -72.4370117, -8.1787186, 8.2191124
18: -4.1777048, 1.0691819, -4.1749363, 1.0584300, -3.3879852, 3.3946476
19: -30.5249138, -23.2034569, -30.5123615, -23.2201767, -4.6474953, 4.6492500
20: -11.1720009, -5.1537991, -11.1661997, -5.1600647, -4.9388618, 4.9388885
21: -43.5461426, -35.0551033, -43.5300789, -35.0790405, -4.2600689, 4.2653809
22: -27.0050240, -19.5271416, -27.0002861, -19.5497589, -4.3285618, 4.3462505
23: -20.8561096, -12.5095081, -20.8445320, -12.5216122, -4.7738609, 4.7716999
24: -16.8623276, -7.6413236, -16.8561382, -7.6502113, -7.1548195, 7.1597824
25: -14.6402349, -6.9547219, -14.6285496, -6.9744334, -4.1920815, 4.1990452
26: -14.6176291, -7.8003669, -14.6105719, -7.8179474, -6.5495644, 6.5634193
27: -14.6305561, -9.5279999, -14.6210728, -9.5463486, -4.0558243, 4.0631142
28: -10.0230169, -1.4303138, -10.0200129, -1.4220256, -6.1681519, 6.1563950
29: -45.5838127, -36.8120422, -45.5731659, -36.8451042, -4.9987469, 5.0212498
30: -32.1860123, -23.0103569, -32.1719017, -23.0259132, -4.9794350, 4.9906940
31: -32.2408981, -23.5151539, -32.2213097, -23.5392494, -6.3085480, 6.3099976
32: 7.7008858, 13.6733494, 7.7228394, 13.6795845, -4.1805706, 4.1501865
33: 4.5882063, 16.3115215, 4.6217232, 16.3157864, -6.7118969, 6.6657562
34: 20.5269756, 30.9848976, 20.5631657, 30.9785690, -5.7460861, 5.7141361
35: 16.4906235, 26.8594131, 16.5268993, 26.8549728, -5.4478703, 5.4120464
36: 28.7973709, 35.1240768, 28.8230095, 35.1169815, -3.4414501, 3.4251289
37: 11.0066671, 20.1137829, 11.0318642, 20.1141624, -5.9716568, 5.9412689
38: 34.8495407, 43.6899261, 34.8764076, 43.6718445, -6.0379982, 6.0303612
39: 8.9733000, 18.5077343, 8.9996738, 18.5038872, -6.5344620, 6.5115547
40: 15.7735825, 25.1259422, 15.8024569, 25.1361923, -5.8286552, 5.7848797
41: 6.7167492, 13.2225609, 6.7364345, 13.2243643, -5.0303154, 5.0070114
42: -12.3983374, -3.4552188, -12.3781538, -3.4620640, -7.0627747, 7.0483932

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5745822, upper bound: 3.6307843
time: 5.99 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5745792, upper bound: 3.6251582
time: 6.15 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -21.5579834, -8.4811563, -21.5724335, -8.4842739, -10.3782043, 10.3919449
1: -21.4219856, -12.2365484, -21.4363556, -12.2395029, -5.2619133, 5.2798576
2: -12.3915615, -5.7777085, -12.4024239, -5.7800450, -4.2475033, 4.2609081
3: -12.0071468, -4.1688685, -12.0113773, -4.1806560, -5.3470192, 5.3641281
4: -10.2769680, 0.0037479, -10.2870455, -0.0188769, -6.0177155, 6.0538044
5: -13.5527449, -4.0461941, -13.5649290, -4.0534735, -6.1275024, 6.1423454
6: -8.3162031, 0.5253098, -8.2979946, 0.5358833, -6.4571991, 6.4265633
7: -32.1505814, -22.0557404, -32.1639786, -22.0601234, -5.8182068, 5.8363342
8: -18.8011246, -9.1073551, -18.8042393, -9.1484690, -5.1829720, 5.2284679
9: -5.3288035, 1.3930535, -5.3373280, 1.3826716, -4.0353851, 4.0561085
10: -36.1260567, -27.7660561, -36.1359711, -27.7696667, -5.2434635, 5.2561951
11: -55.1253281, -44.8033066, -55.0969620, -44.8067627, -4.9300270, 4.9129772
12: -11.5745134, -4.5910568, -11.5679131, -4.5842657, -6.2396545, 6.2280426
13: 0.8940356, 8.0107975, 0.8916062, 7.9890151, -5.2706642, 5.2947845
14: -71.0678329, -57.9613991, -71.0915527, -57.9615631, -8.2196350, 8.2469101
15: -8.8990755, 0.9011693, -8.9089165, 0.8810186, -4.8442554, 4.8773861
16: -33.5529327, -23.9759464, -33.5436478, -23.9812031, -6.4389839, 6.4365311
17: -88.6560822, -72.4222717, -88.6764832, -72.4274750, -8.1455193, 8.1774292
18: -4.1735067, 1.0583792, -4.1515331, 1.0551367, -3.3919392, 3.3727245
19: -30.5235023, -23.2042046, -30.5143356, -23.2066956, -4.6449947, 4.6383591
20: -11.1709690, -5.1588526, -11.1713057, -5.1536303, -4.9262695, 4.9193726
21: -43.5443840, -35.0580826, -43.5299988, -35.0624313, -4.2544231, 4.2442360
22: -26.9977493, -19.5396614, -26.9996872, -19.5424328, -4.3238392, 4.3289585
23: -20.8500271, -12.5152779, -20.8285027, -12.5223312, -4.7670975, 4.7527275
24: -16.8572483, -7.6482162, -16.8316956, -7.6555991, -7.1497879, 7.1352730
25: -14.6360760, -6.9600935, -14.6265678, -6.9654036, -4.1864414, 4.1817627
26: -14.6093721, -7.8215179, -14.6106510, -7.8193498, -6.5162163, 6.5150528
27: -14.6275578, -9.5355358, -14.6177101, -9.5388498, -4.0503826, 4.0448456
28: -10.0212345, -1.4322876, -10.0125237, -1.4296120, -6.1482162, 6.1406097
29: -45.5753860, -36.8263397, -45.5657921, -36.8326302, -4.9895992, 4.9877567
30: -32.1823769, -23.0205536, -32.1530762, -23.0245304, -4.9836521, 4.9684792
31: -32.2330093, -23.5217361, -32.2091675, -23.5276451, -6.2880020, 6.2706909
32: 7.7196803, 13.6643209, 7.7251992, 13.6772833, -4.1558132, 4.1353321
33: 4.6117835, 16.3025074, 4.6231813, 16.3134079, -6.6813869, 6.6616478
34: 20.5370865, 30.9774494, 20.5526772, 30.9884262, -5.7292633, 5.7003727
35: 16.5063782, 26.8521919, 16.5237503, 26.8576813, -5.4238033, 5.3991203
36: 28.8109665, 35.1159744, 28.8126926, 35.1230736, -3.4222012, 3.4144859
37: 11.0274429, 20.1061268, 11.0426617, 20.1110706, -5.9500237, 5.9283180
38: 34.8639641, 43.6769333, 34.8684692, 43.6726456, -6.0205193, 6.0187569
39: 8.9996548, 18.4968891, 9.0003071, 18.4997540, -6.5091743, 6.5012207
40: 15.7941332, 25.1175671, 15.8060102, 25.1293697, -5.7993240, 5.7734890
41: 6.7347732, 13.2131405, 6.7477283, 13.2196817, -5.0096512, 4.9894142
42: -12.3857851, -3.4645004, -12.3853006, -3.4522071, -7.0489464, 7.0354347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6260929
time: 5.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6334852
time: 5.26 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.5579834, -8.4811563, -21.5893040, -8.4791241, -10.4173355, 10.4432602
1: -21.4219856, -12.2365484, -21.4398346, -12.2366695, -5.2795830, 5.2976246
2: -12.3915615, -5.7777085, -12.4056311, -5.7779608, -4.2651291, 4.2769775
3: -12.0071468, -4.1688685, -12.0178127, -4.1676788, -5.3535843, 5.3644180
4: -10.2769680, 0.0037479, -10.3022442, 0.0041616, -6.0185242, 6.0467472
5: -13.5527449, -4.0461941, -13.5717993, -4.0453963, -6.1388130, 6.1544342
6: -8.3162031, 0.5253098, -8.3171616, 0.5448667, -6.4625626, 6.4414253
7: -32.1505814, -22.0557404, -32.1656075, -22.0554199, -5.8285904, 5.8431187
8: -18.8011246, -9.1073551, -18.8230095, -9.1068678, -5.1960220, 5.2192421
9: -5.3288035, 1.3930535, -5.3454218, 1.3937551, -4.0372314, 4.0551987
10: -36.1260567, -27.7660561, -36.1423874, -27.7645817, -5.2488289, 5.2653122
11: -55.1253281, -44.8033066, -55.1264343, -44.7895393, -4.9211445, 4.9162312
12: -11.5745134, -4.5910568, -11.5747194, -4.5762444, -6.2416420, 6.2281113
13: 0.8940356, 8.0107975, 0.8810883, 8.0130281, -5.2876129, 5.2980843
14: -71.0678329, -57.9613991, -71.0949478, -57.9582748, -8.2428436, 8.2664070
15: -8.8990755, 0.9011693, -8.9242620, 0.9013116, -4.8470821, 4.8760414
16: -33.5529327, -23.9759464, -33.5703239, -23.9677238, -6.4419098, 6.4521141
17: -88.6560822, -72.4222717, -88.6783066, -72.4184952, -8.1577682, 8.1804962
18: -4.1735067, 1.0583792, -4.1778874, 1.0635507, -3.3870163, 3.3859291
19: -30.5235023, -23.2042046, -30.5261154, -23.2014236, -4.6427555, 4.6412754
20: -11.1709690, -5.1588526, -11.1723633, -5.1503844, -4.9385643, 4.9323730
21: -43.5443840, -35.0580826, -43.5473938, -35.0522118, -4.2492752, 4.2461987
22: -26.9977493, -19.5396614, -27.0068665, -19.5395107, -4.3227234, 4.3310280
23: -20.8500271, -12.5152779, -20.8515205, -12.5075779, -4.7664070, 4.7595444
24: -16.8572483, -7.6482162, -16.8599396, -7.6403017, -7.1546249, 7.1528778
25: -14.6360760, -6.9600935, -14.6378460, -6.9580021, -4.1871071, 4.1859856
26: -14.6093721, -7.8215179, -14.6127462, -7.8142643, -6.5431938, 6.5436592
27: -14.6275578, -9.5355358, -14.6310034, -9.5307779, -4.0526447, 4.0522919
28: -10.0212345, -1.4322876, -10.0229874, -1.4191103, -6.1679649, 6.1577530
29: -45.5753860, -36.8263397, -45.5860214, -36.8219452, -4.9901524, 4.9977188
30: -32.1823769, -23.0205536, -32.1828918, -23.0043488, -4.9783497, 4.9717255
31: -32.2330093, -23.5217361, -32.2365685, -23.5173988, -6.2942085, 6.2928734
32: 7.7196803, 13.6643209, 7.7192507, 13.6801529, -4.1614799, 4.1441669
33: 4.6117835, 16.3025074, 4.6100254, 16.3190460, -6.6829472, 6.6672478
34: 20.5370865, 30.9774494, 20.5357323, 30.9971485, -5.7257557, 5.7051773
35: 16.5063782, 26.8521919, 16.5044403, 26.8676128, -5.4197464, 5.4044380
36: 28.8109665, 35.1159744, 28.8093147, 35.1255836, -3.4247046, 3.4165020
37: 11.0274429, 20.1061268, 11.0253296, 20.1180496, -5.9493675, 5.9383354
38: 34.8639641, 43.6769333, 34.8552246, 43.6871071, -6.0239296, 6.0211639
39: 8.9996548, 18.4968891, 8.9903574, 18.5047798, -6.5102615, 6.5103531
40: 15.7941332, 25.1175671, 15.7913923, 25.1379929, -5.8064003, 5.7865276
41: 6.7347732, 13.2131405, 6.7334027, 13.2272882, -5.0128746, 4.9991760
42: -12.3857851, -3.4645004, -12.3863640, -3.4491861, -7.0526848, 7.0367622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6260931
time: 5.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6334854
time: 5.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5781021, -8.4788551, -21.5724335, -8.4842739, -10.3991241, 10.4008331
1: -21.4303780, -12.2310791, -21.4363556, -12.2395029, -5.2720642, 5.2841206
2: -12.3956661, -5.7774386, -12.4024239, -5.7800450, -4.2510281, 4.2632294
3: -12.0103178, -4.1644087, -12.0113773, -4.1806560, -5.3498268, 5.3681908
4: -10.2874737, 0.0160040, -10.2870455, -0.0188769, -6.0281906, 6.0679359
5: -13.5587149, -4.0415139, -13.5649290, -4.0534735, -6.1346855, 6.1498337
6: -8.3385096, 0.5390713, -8.2979946, 0.5358833, -6.4791412, 6.4396820
7: -32.1568871, -22.0499954, -32.1639786, -22.0601234, -5.8262215, 5.8423347
8: -18.8056488, -9.1007957, -18.8042393, -9.1484690, -5.1889458, 5.2365131
9: -5.3422594, 1.3966579, -5.3373280, 1.3826716, -4.0493889, 4.0600433
10: -36.1350098, -27.7537746, -36.1359711, -27.7696667, -5.2526417, 5.2641754
11: -55.1323166, -44.7844963, -55.0969620, -44.8067627, -4.9382801, 4.9313717
12: -11.5809269, -4.5859241, -11.5679131, -4.5842657, -6.2423325, 6.2292328
13: 0.8842248, 8.0159569, 0.8916062, 7.9890151, -5.2806702, 5.3012733
14: -71.0843048, -57.9474068, -71.0915527, -57.9615631, -8.2360001, 8.2585907
15: -8.9112701, 0.9151077, -8.9089165, 0.8810186, -4.8566837, 4.8947792
16: -33.5677299, -23.9756050, -33.5436478, -23.9812031, -6.4560623, 6.4399605
17: -88.6775436, -72.3879013, -88.6764832, -72.4274750, -8.1690521, 8.2133789
18: -4.1777048, 1.0691819, -4.1515331, 1.0551367, -3.3966446, 3.3834496
19: -30.5249138, -23.2034569, -30.5143356, -23.2066956, -4.6473942, 4.6392040
20: -11.1720009, -5.1537991, -11.1713057, -5.1536303, -4.9278984, 4.9241638
21: -43.5461426, -35.0551033, -43.5299988, -35.0624313, -4.2571869, 4.2467728
22: -27.0050240, -19.5271416, -26.9996872, -19.5424328, -4.3319683, 4.3426285
23: -20.8561096, -12.5095081, -20.8285027, -12.5223312, -4.7757378, 4.7600403
24: -16.8623276, -7.6413236, -16.8316956, -7.6555991, -7.1546326, 7.1420250
25: -14.6402349, -6.9547219, -14.6265678, -6.9654036, -4.1918545, 4.1880112
26: -14.6176291, -7.8003669, -14.6106510, -7.8193498, -6.5251732, 6.5357208
27: -14.6305561, -9.5279999, -14.6177101, -9.5388498, -4.0538960, 4.0518036
28: -10.0230169, -1.4303138, -10.0125237, -1.4296120, -6.1519547, 6.1423454
29: -45.5838127, -36.8120422, -45.5657921, -36.8326302, -4.9993420, 5.0017128
30: -32.1860123, -23.0103569, -32.1530762, -23.0245304, -4.9874172, 4.9784164
31: -32.2408981, -23.5151539, -32.2091675, -23.5276451, -6.2985229, 6.2792511
32: 7.7008858, 13.6733494, 7.7251992, 13.6772833, -4.1751270, 4.1447010
33: 4.5882063, 16.3115215, 4.6231813, 16.3134079, -6.7099590, 6.6718407
34: 20.5269756, 30.9848976, 20.5526772, 30.9884262, -5.7410431, 5.7090168
35: 16.4906235, 26.8594131, 16.5237503, 26.8576813, -5.4421654, 5.4077263
36: 28.7973709, 35.1240768, 28.8126926, 35.1230736, -3.4362850, 3.4234123
37: 11.0066671, 20.1137829, 11.0426617, 20.1110706, -5.9734306, 5.9362450
38: 34.8495407, 43.6899261, 34.8684692, 43.6726456, -6.0349121, 6.0330620
39: 8.9733000, 18.5077343, 9.0003071, 18.4997540, -6.5361137, 6.5131645
40: 15.7735825, 25.1259422, 15.8060102, 25.1293697, -5.8211327, 5.7818413
41: 6.7167492, 13.2225609, 6.7477283, 13.2196817, -5.0286331, 4.9992256
42: -12.3983374, -3.4552188, -12.3853006, -3.4522071, -7.0607491, 7.0447540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5745822, upper bound: 3.6307843
time: 5.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5745822, upper bound: 3.6381771
time: 5.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5781021, -8.4788551, -21.5893040, -8.4791241, -10.4382477, 10.4521408
1: -21.4303780, -12.2310791, -21.4398346, -12.2366695, -5.2897339, 5.3018856
2: -12.3956661, -5.7774386, -12.4056311, -5.7779608, -4.2686539, 4.2792969
3: -12.0103178, -4.1644087, -12.0178127, -4.1676788, -5.3563881, 5.3684769
4: -10.2874737, 0.0160040, -10.3022442, 0.0041616, -6.0289993, 6.0608807
5: -13.5587149, -4.0415139, -13.5717993, -4.0453963, -6.1459961, 6.1619225
6: -8.3385096, 0.5390713, -8.3171616, 0.5448667, -6.4844971, 6.4545441
7: -32.1568871, -22.0499954, -32.1656075, -22.0554199, -5.8366165, 5.8491116
8: -18.8056488, -9.1007957, -18.8230095, -9.1068678, -5.2019920, 5.2272854
9: -5.3422594, 1.3966579, -5.3454218, 1.3937551, -4.0512333, 4.0591373
10: -36.1350098, -27.7537746, -36.1423874, -27.7645817, -5.2580070, 5.2732925
11: -55.1323166, -44.7844963, -55.1264343, -44.7895393, -4.9293995, 4.9346294
12: -11.5809269, -4.5859241, -11.5747194, -4.5762444, -6.2443352, 6.2292976
13: 0.8842248, 8.0159569, 0.8810883, 8.0130281, -5.2976227, 5.3045769
14: -71.0843048, -57.9474068, -71.0949478, -57.9582748, -8.2592087, 8.2780838
15: -8.9112701, 0.9151077, -8.9242620, 0.9013116, -4.8595123, 4.8934326
16: -33.5677299, -23.9756050, -33.5703239, -23.9677238, -6.4589882, 6.4555397
17: -88.6775436, -72.3879013, -88.6783066, -72.4184952, -8.1813049, 8.2164307
18: -4.1777048, 1.0691819, -4.1778874, 1.0635507, -3.3917179, 3.3966541
19: -30.5249138, -23.2034569, -30.5261154, -23.2014236, -4.6451550, 4.6421204
20: -11.1720009, -5.1537991, -11.1723633, -5.1503844, -4.9401932, 4.9371490
21: -43.5461426, -35.0551033, -43.5473938, -35.0522118, -4.2520370, 4.2487354
22: -27.0050240, -19.5271416, -27.0068665, -19.5395107, -4.3308563, 4.3446980
23: -20.8561096, -12.5095081, -20.8515205, -12.5075779, -4.7750435, 4.7668610
24: -16.8623276, -7.6413236, -16.8599396, -7.6403017, -7.1594696, 7.1596375
25: -14.6402349, -6.9547219, -14.6378460, -6.9580021, -4.1925201, 4.1922359
26: -14.6176291, -7.8003669, -14.6127462, -7.8142643, -6.5521507, 6.5643272
27: -14.6305561, -9.5279999, -14.6310034, -9.5307779, -4.0561523, 4.0592480
28: -10.0230169, -1.4303138, -10.0229874, -1.4191103, -6.1717110, 6.1594734
29: -45.5838127, -36.8120422, -45.5860214, -36.8219452, -4.9998989, 5.0116692
30: -32.1860123, -23.0103569, -32.1828918, -23.0043488, -4.9821186, 4.9816589
31: -32.2408981, -23.5151539, -32.2365685, -23.5173988, -6.3047295, 6.3014336
32: 7.7008858, 13.6733494, 7.7192507, 13.6801529, -4.1807938, 4.1535320
33: 4.5882063, 16.3115215, 4.6100254, 16.3190460, -6.7115536, 6.6774445
34: 20.5269756, 30.9848976, 20.5357323, 30.9971485, -5.7375431, 5.7138252
35: 16.4906235, 26.8594131, 16.5044403, 26.8676128, -5.4381084, 5.4130440
36: 28.7973709, 35.1240768, 28.8093147, 35.1255836, -3.4387922, 3.4254322
37: 11.0066671, 20.1137829, 11.0253296, 20.1180496, -5.9727707, 5.9462662
38: 34.8495407, 43.6899261, 34.8552246, 43.6871071, -6.0383148, 6.0354691
39: 8.9733000, 18.5077343, 8.9903574, 18.5047798, -6.5371933, 6.5222969
40: 15.7735825, 25.1259422, 15.7913923, 25.1379929, -5.8282051, 5.7948761
41: 6.7167492, 13.2225609, 6.7334027, 13.2272882, -5.0318642, 5.0089874
42: -12.3983374, -3.4552188, -12.3863640, -3.4491861, -7.0644875, 7.0460777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5745822, upper bound: 3.6307845
time: 6.00 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5745792, upper bound: 3.6251583
time: 6.25 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -21.5672455, -8.4783888, -21.5678844, -8.4849710, -10.4099121, 10.3857422
1: -21.4237137, -12.2342529, -21.4330940, -12.2432528, -5.2666626, 5.2795734
2: -12.3945704, -5.7748389, -12.3971043, -5.7820091, -4.2609901, 4.2603912
3: -12.0111580, -4.1600070, -11.9989748, -4.1900549, -5.3540192, 5.3722115
4: -10.2876625, 0.0246730, -10.2784653, -0.0219605, -6.0306282, 6.0733414
5: -13.5592213, -4.0388803, -13.5628099, -4.0566530, -6.1345558, 6.1387138
6: -8.3306007, 0.5306103, -8.2925930, 0.5267129, -6.4692268, 6.4330025
7: -32.1515274, -22.0571251, -32.1577148, -22.0700989, -5.8169937, 5.8374386
8: -18.8127232, -9.0715923, -18.7968178, -9.1539326, -5.1943932, 5.2631130
9: -5.3333635, 1.4026828, -5.3296499, 1.3802854, -4.0352554, 4.0545311
10: -36.1315498, -27.7596016, -36.1271095, -27.7833977, -5.2491608, 5.2616539
11: -55.1605225, -44.7897644, -55.0776215, -44.8416748, -4.9681187, 4.9452782
12: -11.5765877, -4.5835438, -11.5606689, -4.5933619, -6.2378540, 6.2325363
13: 0.8880055, 8.0301390, 0.9011034, 7.9844589, -5.2771759, 5.3111496
14: -71.0719147, -57.9545135, -71.0888519, -57.9642792, -8.2282791, 8.2474976
15: -8.9120235, 0.9227800, -8.9021587, 0.8786244, -4.8635273, 4.9034882
16: -33.5763931, -23.9675331, -33.5236664, -24.0034561, -6.4600105, 6.4449463
17: -88.6563797, -72.4192581, -88.6654434, -72.4473419, -8.1430740, 8.1852036
18: -4.1972079, 1.0632889, -4.1486230, 1.0502489, -3.4121284, 3.3755569
19: -30.5325451, -23.2012672, -30.5006027, -23.2253666, -4.6569099, 4.6480217
20: -11.1730614, -5.1576586, -11.1651716, -5.1640816, -4.9246254, 4.9253693
21: -43.5612335, -35.0513191, -43.5127258, -35.0889320, -4.2795601, 4.2675972
22: -27.0012093, -19.5390244, -26.9929504, -19.5526791, -4.3255043, 4.3305149
23: -20.8694973, -12.5062580, -20.8215523, -12.5360260, -4.7857552, 4.7671394
24: -16.8771172, -7.6407804, -16.8279610, -7.6652002, -7.1652603, 7.1428223
25: -14.6401129, -6.9574556, -14.6173153, -6.9817605, -4.1905785, 4.1913719
26: -14.6137590, -7.8181028, -14.6085854, -7.8233547, -6.5138130, 6.5325356
27: -14.6389408, -9.5302467, -14.6078348, -9.5542107, -4.0610485, 4.0540333
28: -10.0359812, -1.4237872, -10.0095959, -1.4322555, -6.1447945, 6.1450729
29: -45.5883331, -36.8221054, -45.5529327, -36.8555908, -5.0014725, 5.0006561
30: -32.2146606, -23.0048294, -32.1421051, -23.0455818, -5.0138321, 4.9934406
31: -32.2524033, -23.5173759, -32.1940231, -23.5492897, -6.3125725, 6.2831230
32: 7.7147226, 13.6665144, 7.7287788, 13.6767883, -4.1607513, 4.1341324
33: 4.6090717, 16.3037891, 4.6356068, 16.3102055, -6.6873703, 6.6511879
34: 20.5248680, 30.9823914, 20.5800419, 30.9700508, -5.7502270, 5.7052422
35: 16.4925289, 26.8572845, 16.5461464, 26.8452873, -5.4475269, 5.4031506
36: 28.8088570, 35.1176224, 28.8263588, 35.1145134, -3.4268665, 3.4156771
37: 11.0179739, 20.1096535, 11.0491600, 20.1073151, -5.9588890, 5.9265251
38: 34.8531952, 43.6908379, 34.8892975, 43.6574020, -6.0312080, 6.0279312
39: 8.9903316, 18.5086327, 9.0093861, 18.4988804, -6.5155602, 6.4999504
40: 15.7832565, 25.1224384, 15.8170156, 25.1276302, -5.8097725, 5.7715321
41: 6.7259016, 13.2174816, 6.7507286, 13.2168999, -5.0170746, 4.9917336
42: -12.3857994, -3.4617722, -12.3768597, -3.4650753, -7.0473862, 7.0435982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6261447
time: 4.89 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6335370
time: 5.02 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.5672455, -8.4783888, -21.5847187, -8.4797974, -10.4490433, 10.4370728
1: -21.4237137, -12.2342529, -21.4365692, -12.2404108, -5.2843285, 5.2973461
2: -12.3945704, -5.7748389, -12.4002924, -5.7799354, -4.2786179, 4.2764549
3: -12.0111580, -4.1600070, -12.0054216, -4.1770658, -5.3605919, 5.3725090
4: -10.2876625, 0.0246730, -10.2937126, 0.0010668, -6.0314445, 6.0662899
5: -13.5592213, -4.0388803, -13.5696793, -4.0485582, -6.1459312, 6.1507912
6: -8.3306007, 0.5306103, -8.3117695, 0.5357151, -6.4745789, 6.4478760
7: -32.1515274, -22.0571251, -32.1593475, -22.0653934, -5.8273735, 5.8442001
8: -18.8127232, -9.0715923, -18.8155632, -9.1123924, -5.2074356, 5.2538967
9: -5.3333635, 1.4026828, -5.3377371, 1.3913652, -4.0371037, 4.0536156
10: -36.1315498, -27.7596016, -36.1335106, -27.7783089, -5.2546082, 5.2707558
11: -55.1605225, -44.7897644, -55.1070557, -44.8244629, -4.9592323, 4.9485245
12: -11.5765877, -4.5835438, -11.5674801, -4.5853252, -6.2398415, 6.2325974
13: 0.8880055, 8.0301390, 0.8905934, 8.0085125, -5.2941399, 5.3144341
14: -71.0719147, -57.9545135, -71.0921936, -57.9609489, -8.2514725, 8.2669983
15: -8.9120235, 0.9227800, -8.9175110, 0.8989143, -4.8663464, 4.9021435
16: -33.5763931, -23.9675331, -33.5503578, -23.9899712, -6.4629402, 6.4605217
17: -88.6563797, -72.4192581, -88.6672745, -72.4383087, -8.1553383, 8.1882935
18: -4.1972079, 1.0632889, -4.1749716, 1.0586429, -3.4072132, 3.3887272
19: -30.5325451, -23.2012672, -30.5123940, -23.2200775, -4.6546440, 4.6509361
20: -11.1730614, -5.1576586, -11.1662407, -5.1608210, -4.9369354, 4.9383736
21: -43.5612335, -35.0513191, -43.5301132, -35.0787659, -4.2744122, 4.2695599
22: -27.0012093, -19.5390244, -27.0001221, -19.5497475, -4.3243465, 4.3325710
23: -20.8694973, -12.5062580, -20.8445683, -12.5213070, -4.7850590, 4.7739506
24: -16.8771172, -7.6407804, -16.8561974, -7.6499062, -7.1701126, 7.1604462
25: -14.6401129, -6.9574556, -14.6285992, -6.9743471, -4.1912422, 4.1956081
26: -14.6137590, -7.8181028, -14.6106653, -7.8182821, -6.5408020, 6.5611496
27: -14.6389408, -9.5302467, -14.6211596, -9.5461397, -4.0633068, 4.0614929
28: -10.0359812, -1.4237872, -10.0200710, -1.4217129, -6.1645470, 6.1622810
29: -45.5883331, -36.8221054, -45.5731812, -36.8449173, -5.0020237, 5.0106163
30: -32.2146606, -23.0048294, -32.1719093, -23.0253754, -5.0085373, 4.9966717
31: -32.2524033, -23.5173759, -32.2214355, -23.5390263, -6.3187714, 6.3052864
32: 7.7147226, 13.6665144, 7.7228246, 13.6796646, -4.1664200, 4.1429710
33: 4.6090717, 16.3037891, 4.6224365, 16.3158188, -6.6888504, 6.6567917
34: 20.5248680, 30.9823914, 20.5631161, 30.9787388, -5.7467327, 5.7100487
35: 16.4925289, 26.8572845, 16.5268326, 26.8551922, -5.4434738, 5.4084797
36: 28.8088570, 35.1176224, 28.8229656, 35.1170044, -3.4293795, 3.4177113
37: 11.0179739, 20.1096535, 11.0318098, 20.1142864, -5.9582367, 5.9365463
38: 34.8531952, 43.6908379, 34.8760490, 43.6718559, -6.0346146, 6.0303383
39: 8.9903316, 18.5086327, 8.9994946, 18.5038986, -6.5166512, 6.5090408
40: 15.7832565, 25.1224384, 15.8023605, 25.1362190, -5.8168430, 5.7845974
41: 6.7259016, 13.2174816, 6.7363825, 13.2245331, -5.0203018, 5.0014954
42: -12.3857994, -3.4617722, -12.3779039, -3.4620495, -7.0511246, 7.0449524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6261449
time: 4.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6335369
time: 4.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5873756, -8.4760866, -21.5678844, -8.4849710, -10.4308243, 10.3946381
1: -21.4321213, -12.2287636, -21.4330940, -12.2432528, -5.2768173, 5.2838383
2: -12.3986912, -5.7745886, -12.3971043, -5.7820091, -4.2645111, 4.2627106
3: -12.0143270, -4.1555634, -11.9989748, -4.1900549, -5.3568268, 5.3762436
4: -10.2981749, 0.0369308, -10.2784653, -0.0219605, -6.0411148, 6.0874786
5: -13.5651894, -4.0341926, -13.5628099, -4.0566530, -6.1417389, 6.1461906
6: -8.3528681, 0.5443628, -8.2925930, 0.5267129, -6.4911842, 6.4460945
7: -32.1578331, -22.0514069, -32.1577148, -22.0700989, -5.8250046, 5.8434391
8: -18.8172703, -9.0650339, -18.7968178, -9.1539326, -5.2003708, 5.2711639
9: -5.3468108, 1.4063028, -5.3296499, 1.3802854, -4.0492172, 4.0584679
10: -36.1405067, -27.7473030, -36.1271095, -27.7833977, -5.2583275, 5.2696362
11: -55.1675415, -44.7709808, -55.0776215, -44.8416748, -4.9763813, 4.9636841
12: -11.5829935, -4.5783935, -11.5606689, -4.5933619, -6.2405205, 6.2337379
13: 0.8782057, 8.0353088, 0.9011034, 7.9844589, -5.2871933, 5.3176346
14: -71.0884247, -57.9405632, -71.0888519, -57.9642792, -8.2446327, 8.2591667
15: -8.9242020, 0.9366841, -8.9021587, 0.8786244, -4.8759518, 4.9208775
16: -33.5911331, -23.9671917, -33.5236664, -24.0034561, -6.4769516, 6.4483643
17: -88.6778107, -72.3848724, -88.6654434, -72.4473419, -8.1666107, 8.2211685
18: -4.2013907, 1.0740848, -4.1486230, 1.0502489, -3.4168282, 3.3862801
19: -30.5339222, -23.2005119, -30.5006027, -23.2253666, -4.6592979, 4.6488705
20: -11.1740999, -5.1526041, -11.1651716, -5.1640816, -4.9262428, 4.9301910
21: -43.5629501, -35.0483246, -43.5127258, -35.0889320, -4.2823277, 4.2701378
22: -27.0084991, -19.5265179, -26.9929504, -19.5526791, -4.3336411, 4.3442020
23: -20.8755817, -12.5005035, -20.8215523, -12.5360260, -4.7943916, 4.7744560
24: -16.8822098, -7.6338596, -16.8279610, -7.6652002, -7.1700897, 7.1496162
25: -14.6443043, -6.9520712, -14.6173153, -6.9817605, -4.1959896, 4.1976242
26: -14.6220322, -7.7969537, -14.6085854, -7.8233547, -6.5227547, 6.5532074
27: -14.6419373, -9.5227213, -14.6078348, -9.5542107, -4.0645561, 4.0610104
28: -10.0377865, -1.4218202, -10.0095959, -1.4322555, -6.1485214, 6.1468124
29: -45.5967712, -36.8078270, -45.5529327, -36.8555908, -5.0112343, 5.0146275
30: -32.2183113, -22.9946671, -32.1421051, -23.0455818, -5.0176010, 5.0033913
31: -32.2602692, -23.5107574, -32.1940231, -23.5492897, -6.3231010, 6.2917061
32: 7.6959295, 13.6755581, 7.7287788, 13.6767883, -4.1800690, 4.1434860
33: 4.5854526, 16.3128586, 4.6356068, 16.3102055, -6.7160110, 6.6613846
34: 20.5147591, 30.9898796, 20.5800419, 30.9700508, -5.7620068, 5.7138767
35: 16.4767723, 26.8645210, 16.5461464, 26.8452873, -5.4659271, 5.4117508
36: 28.7952785, 35.1257210, 28.8263588, 35.1145134, -3.4409695, 3.4246073
37: 10.9971600, 20.1173115, 11.0491600, 20.1073151, -5.9822426, 5.9344673
38: 34.8387947, 43.7038383, 34.8892975, 43.6574020, -6.0455933, 6.0421638
39: 8.9639616, 18.5195007, 9.0093861, 18.4988804, -6.5425034, 6.5118980
40: 15.7626715, 25.1308022, 15.8170156, 25.1276302, -5.8315849, 5.7798882
41: 6.7078686, 13.2268858, 6.7507286, 13.2168999, -5.0360527, 5.0015602
42: -12.3983316, -3.4525018, -12.3768597, -3.4650753, -7.0592041, 7.0528908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6308365
time: 4.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6382292
time: 4.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5873756, -8.4760866, -21.5847187, -8.4797974, -10.4699631, 10.4459686
1: -21.4321213, -12.2287636, -21.4365692, -12.2404108, -5.2944794, 5.3016109
2: -12.3986912, -5.7745886, -12.4002924, -5.7799354, -4.2821465, 4.2787704
3: -12.0143270, -4.1555634, -12.0054216, -4.1770658, -5.3634109, 5.3765373
4: -10.2981749, 0.0369308, -10.2937126, 0.0010668, -6.0419312, 6.0804234
5: -13.5651894, -4.0341926, -13.5696793, -4.0485582, -6.1531143, 6.1582756
6: -8.3528681, 0.5443628, -8.3117695, 0.5357151, -6.4965286, 6.4609680
7: -32.1578331, -22.0514069, -32.1593475, -22.0653934, -5.8353958, 5.8501892
8: -18.8172703, -9.0650339, -18.8155632, -9.1123924, -5.2134094, 5.2619476
9: -5.3468108, 1.4063028, -5.3377371, 1.3913652, -4.0510654, 4.0575542
10: -36.1405067, -27.7473030, -36.1335106, -27.7783089, -5.2637787, 5.2787495
11: -55.1675415, -44.7709808, -55.1070557, -44.8244629, -4.9674969, 4.9669266
12: -11.5829935, -4.5783935, -11.5674801, -4.5853252, -6.2425232, 6.2338028
13: 0.8782057, 8.0353088, 0.8905934, 8.0085125, -5.3041573, 5.3209190
14: -71.0884247, -57.9405632, -71.0921936, -57.9609489, -8.2678146, 8.2786636
15: -8.9242020, 0.9366841, -8.9175110, 0.8989143, -4.8787689, 4.9195328
16: -33.5911331, -23.9671917, -33.5503578, -23.9899712, -6.4798889, 6.4639397
17: -88.6778107, -72.3848724, -88.6672745, -72.4383087, -8.1788712, 8.2242508
18: -4.2013907, 1.0740848, -4.1749716, 1.0586429, -3.4119110, 3.3994541
19: -30.5339222, -23.2005119, -30.5123940, -23.2200775, -4.6570396, 4.6517906
20: -11.1740999, -5.1526041, -11.1662407, -5.1608210, -4.9385529, 4.9431839
21: -43.5629501, -35.0483246, -43.5301132, -35.0787659, -4.2771816, 4.2721004
22: -27.0084991, -19.5265179, -27.0001221, -19.5497475, -4.3324871, 4.3462543
23: -20.8755817, -12.5005035, -20.8445683, -12.5213070, -4.7936993, 4.7812691
24: -16.8822098, -7.6338596, -16.8561974, -7.6499062, -7.1749344, 7.1672363
25: -14.6443043, -6.9520712, -14.6285992, -6.9743471, -4.1966553, 4.2018623
26: -14.6220322, -7.7969537, -14.6106653, -7.8182821, -6.5497475, 6.5818253
27: -14.6419373, -9.5227213, -14.6211596, -9.5461397, -4.0668087, 4.0684662
28: -10.0377865, -1.4218202, -10.0200710, -1.4217129, -6.1682777, 6.1640053
29: -45.5967712, -36.8078270, -45.5731812, -36.8449173, -5.0117855, 5.0245876
30: -32.2183113, -22.9946671, -32.1719093, -23.0253754, -5.0123062, 5.0066299
31: -32.2602692, -23.5107574, -32.2214355, -23.5390263, -6.3293037, 6.3138733
32: 7.6959295, 13.6755581, 7.7228246, 13.6796646, -4.1857300, 4.1523247
33: 4.5854526, 16.3128586, 4.6224365, 16.3158188, -6.7175331, 6.6669922
34: 20.5147591, 30.9898796, 20.5631161, 30.9787388, -5.7585163, 5.7186737
35: 16.4767723, 26.8645210, 16.5268326, 26.8551922, -5.4618740, 5.4170780
36: 28.7952785, 35.1257210, 28.8229656, 35.1170044, -3.4434843, 3.4266424
37: 10.9971600, 20.1173115, 11.0318098, 20.1142864, -5.9815903, 5.9444771
38: 34.8387947, 43.7038383, 34.8760490, 43.6718559, -6.0490036, 6.0445709
39: 8.9639616, 18.5195007, 8.9994946, 18.5038986, -6.5435791, 6.5209961
40: 15.7626715, 25.1308022, 15.8023605, 25.1362190, -5.8386555, 5.7929516
41: 6.7078686, 13.2268858, 6.7363825, 13.2245331, -5.0392914, 5.0113258
42: -12.3983316, -3.4525018, -12.3779039, -3.4620495, -7.0629501, 7.0542374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6308367
time: 5.19 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6382294
time: 5.16 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -21.5672455, -8.4783888, -21.5723629, -8.4842157, -10.4100876, 10.3895187
1: -21.4237137, -12.2342529, -21.4364014, -12.2397718, -5.2681732, 5.2810040
2: -12.3945704, -5.7748389, -12.4025021, -5.7800579, -4.2568417, 4.2597065
3: -12.0111580, -4.1600070, -12.0115013, -4.1806507, -5.3509750, 5.3729706
4: -10.2876625, 0.0246730, -10.2874098, -0.0188563, -6.0279732, 6.0751572
5: -13.5592213, -4.0388803, -13.5651035, -4.0534716, -6.1371346, 6.1405449
6: -8.3306007, 0.5306103, -8.2980213, 0.5360940, -6.4718781, 6.4320908
7: -32.1515274, -22.0571251, -32.1639748, -22.0618744, -5.8212204, 5.8409176
8: -18.8127232, -9.0715923, -18.8047466, -9.1484432, -5.1943207, 5.2643547
9: -5.3333635, 1.4026828, -5.3374877, 1.3826954, -4.0397186, 4.0657673
10: -36.1315498, -27.7596016, -36.1361084, -27.7696495, -5.2491856, 5.2581882
11: -55.1605225, -44.7897644, -55.0969887, -44.8061562, -4.9659615, 4.9262352
12: -11.5765877, -4.5835438, -11.5678997, -4.5841112, -6.2405510, 6.2349014
13: 0.8880055, 8.0301390, 0.8913711, 7.9890881, -5.2768021, 5.3146400
14: -71.0719147, -57.9545135, -71.0915985, -57.9615669, -8.2332268, 8.2521286
15: -8.9120235, 0.9227800, -8.9093456, 0.8810124, -4.8573151, 4.8993626
16: -33.5763931, -23.9675331, -33.5436630, -23.9808235, -6.4626961, 6.4443207
17: -88.6563797, -72.4192581, -88.6764297, -72.4287720, -8.1456528, 8.1825523
18: -4.1972079, 1.0632889, -4.1515746, 1.0553567, -3.4158688, 3.3775272
19: -30.5325451, -23.2012672, -30.5143719, -23.2066040, -4.6545582, 4.6408882
20: -11.1730614, -5.1576586, -11.1713238, -5.1544156, -4.9259720, 4.9236660
21: -43.5612335, -35.0513191, -43.5300140, -35.0621414, -4.2715321, 4.2509613
22: -27.0012093, -19.5390244, -26.9995136, -19.5424309, -4.3277740, 4.3289528
23: -20.8694973, -12.5062580, -20.8285332, -12.5219765, -4.7869415, 4.7623215
24: -16.8771172, -7.6407804, -16.8317719, -7.6552944, -7.1699295, 7.1427002
25: -14.6401129, -6.9574556, -14.6266289, -6.9653306, -4.1910038, 4.1845703
26: -14.6137590, -7.8181028, -14.6107607, -7.8196950, -6.5163918, 6.5334435
27: -14.6389408, -9.5302467, -14.6177883, -9.5386524, -4.0613785, 4.0501709
28: -10.0359812, -1.4237872, -10.0125294, -1.4293417, -6.1483498, 6.1481552
29: -45.5883331, -36.8221054, -45.5658188, -36.8324280, -5.0026207, 4.9910889
30: -32.2146606, -23.0048294, -32.1531029, -23.0239906, -5.0165272, 4.9844532
31: -32.2524033, -23.5173759, -32.2092667, -23.5274448, -6.3087273, 6.2745628
32: 7.7147226, 13.6665144, 7.7251773, 13.6773643, -4.1609764, 4.1374798
33: 4.6090717, 16.3037891, 4.6239252, 16.3134346, -6.6871109, 6.6628723
34: 20.5248680, 30.9823914, 20.5526218, 30.9886265, -5.7416954, 5.7049179
35: 16.4925289, 26.8572845, 16.5237045, 26.8579082, -5.4378242, 5.4041481
36: 28.8088570, 35.1176224, 28.8126507, 35.1231155, -3.4242306, 3.4159689
37: 11.0179739, 20.1096535, 11.0426188, 20.1111870, -5.9599686, 5.9315300
38: 34.8531952, 43.6908379, 34.8681183, 43.6726570, -6.0315437, 6.0329742
39: 8.9903316, 18.5086327, 9.0000820, 18.4997368, -6.5183144, 6.5106621
40: 15.7832565, 25.1224384, 15.8059187, 25.1294174, -5.8093586, 5.7815495
41: 6.7259016, 13.2174816, 6.7477241, 13.2198448, -5.0186195, 4.9937019
42: -12.3857994, -3.4617722, -12.3850622, -3.4521940, -7.0491257, 7.0412941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6261450
time: 4.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6335371
time: 5.32 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -21.5672455, -8.4783888, -21.5891876, -8.4790573, -10.4492188, 10.4408417
1: -21.4237137, -12.2342529, -21.4398766, -12.2369738, -5.2858372, 5.2987671
2: -12.3945704, -5.7748389, -12.4057016, -5.7779808, -4.2744637, 4.2757683
3: -12.0111580, -4.1600070, -12.0179386, -4.1676483, -5.3575401, 5.3732605
4: -10.2876625, 0.0246730, -10.3026304, 0.0041404, -6.0287857, 6.0681210
5: -13.5592213, -4.0388803, -13.5719433, -4.0453429, -6.1485100, 6.1526260
6: -8.3306007, 0.5306103, -8.3171768, 0.5451018, -6.4772415, 6.4469414
7: -32.1515274, -22.0571251, -32.1656036, -22.0571938, -5.8316040, 5.8476715
8: -18.8127232, -9.0715923, -18.8234901, -9.1068306, -5.2073593, 5.2551346
9: -5.3333635, 1.4026828, -5.3455830, 1.3937838, -4.0415726, 4.0648632
10: -36.1315498, -27.7596016, -36.1425133, -27.7645493, -5.2546482, 5.2673073
11: -55.1605225, -44.7897644, -55.1264496, -44.7889748, -4.9570866, 4.9294872
12: -11.5765877, -4.5835438, -11.5747108, -4.5760660, -6.2425537, 6.2349892
13: 0.8880055, 8.0301390, 0.8808296, 8.0131073, -5.2937584, 5.3179398
14: -71.0719147, -57.9545135, -71.0949707, -57.9582291, -8.2564240, 8.2716255
15: -8.9120235, 0.9227800, -8.9246998, 0.9013143, -4.8601360, 4.8980274
16: -33.5763931, -23.9675331, -33.5703735, -23.9673347, -6.4656181, 6.4598846
17: -88.6563797, -72.4192581, -88.6782608, -72.4197006, -8.1579132, 8.1856384
18: -4.1972079, 1.0632889, -4.1779275, 1.0637596, -3.4109554, 3.3907261
19: -30.5325451, -23.2012672, -30.5261612, -23.2013397, -4.6522903, 4.6438160
20: -11.1730614, -5.1576586, -11.1723824, -5.1511436, -4.9382629, 4.9366722
21: -43.5612335, -35.0513191, -43.5474396, -35.0519333, -4.2663765, 4.2529259
22: -27.0012093, -19.5390244, -27.0067005, -19.5395126, -4.3266220, 4.3310242
23: -20.8694973, -12.5062580, -20.8515396, -12.5072870, -4.7862434, 4.7691422
24: -16.8771172, -7.6407804, -16.8600311, -7.6399951, -7.1747513, 7.1603088
25: -14.6401129, -6.9574556, -14.6379108, -6.9579325, -4.1916656, 4.1888065
26: -14.6137590, -7.8181028, -14.6128550, -7.8146191, -6.5433769, 6.5620461
27: -14.6389408, -9.5302467, -14.6310949, -9.5305891, -4.0636368, 4.0576305
28: -10.0359812, -1.4237872, -10.0230312, -1.4188213, -6.1681061, 6.1653671
29: -45.5883331, -36.8221054, -45.5860481, -36.8217468, -5.0031776, 5.0010471
30: -32.2146606, -23.0048294, -32.1828308, -23.0038185, -5.0112228, 4.9876823
31: -32.2524033, -23.5173759, -32.2366714, -23.5171967, -6.3149338, 6.2967415
32: 7.7147226, 13.6665144, 7.7192421, 13.6802254, -4.1666489, 4.1463108
33: 4.6090717, 16.3037891, 4.6107407, 16.3190479, -6.6885967, 6.6684799
34: 20.5248680, 30.9823914, 20.5356884, 30.9973469, -5.7381973, 5.7097263
35: 16.4925289, 26.8572845, 16.5044193, 26.8678398, -5.4337616, 5.4094753
36: 28.8088570, 35.1176224, 28.8092594, 35.1256104, -3.4267387, 3.4180117
37: 11.0179739, 20.1096535, 11.0253191, 20.1181793, -5.9593163, 5.9415436
38: 34.8531952, 43.6908379, 34.8548508, 43.6871338, -6.0349579, 6.0353851
39: 8.9903316, 18.5086327, 8.9901943, 18.5047569, -6.5193939, 6.5197639
40: 15.7832565, 25.1224384, 15.7912979, 25.1380196, -5.8164330, 5.7945976
41: 6.7259016, 13.2174816, 6.7333636, 13.2274494, -5.0218582, 5.0034676
42: -12.3857994, -3.4617722, -12.3861265, -3.4491959, -7.0528793, 7.0426292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6261452
time: 4.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6335373
time: 4.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5873756, -8.4760866, -21.5723629, -8.4842157, -10.4309998, 10.3983994
1: -21.4321213, -12.2287636, -21.4364014, -12.2397718, -5.2783279, 5.2852726
2: -12.3986912, -5.7745886, -12.4025021, -5.7800579, -4.2603588, 4.2620182
3: -12.0143270, -4.1555634, -12.0115013, -4.1806507, -5.3537903, 5.3770256
4: -10.2981749, 0.0369308, -10.2874098, -0.0188563, -6.0384483, 6.0892925
5: -13.5651894, -4.0341926, -13.5651035, -4.0534716, -6.1443214, 6.1480217
6: -8.3528681, 0.5443628, -8.2980213, 0.5360940, -6.4938278, 6.4451714
7: -32.1578331, -22.0514069, -32.1639748, -22.0618744, -5.8292427, 5.8469028
8: -18.8172703, -9.0650339, -18.8047466, -9.1484432, -5.2002907, 5.2724037
9: -5.3468108, 1.4063028, -5.3374877, 1.3826954, -4.0537090, 4.0697041
10: -36.1405067, -27.7473030, -36.1361084, -27.7696495, -5.2583523, 5.2661610
11: -55.1675415, -44.7709808, -55.0969887, -44.8061562, -4.9742260, 4.9446297
12: -11.5829935, -4.5783935, -11.5678997, -4.5841112, -6.2432251, 6.2361069
13: 0.8782057, 8.0353088, 0.8913711, 7.9890881, -5.2868080, 5.3211212
14: -71.0884247, -57.9405632, -71.0915985, -57.9615669, -8.2495880, 8.2638054
15: -8.9242020, 0.9366841, -8.9093456, 0.8810124, -4.8697433, 4.9167500
16: -33.5911331, -23.9671917, -33.5436630, -23.9808235, -6.4797630, 6.4477386
17: -88.6778107, -72.3848724, -88.6764297, -72.4287720, -8.1691856, 8.2184982
18: -4.2013907, 1.0740848, -4.1515746, 1.0553567, -3.4205666, 3.3882484
19: -30.5339222, -23.2005119, -30.5143719, -23.2066040, -4.6569538, 4.6417313
20: -11.1740999, -5.1526041, -11.1713238, -5.1544156, -4.9275894, 4.9284649
21: -43.5629501, -35.0483246, -43.5300140, -35.0621414, -4.2742958, 4.2534943
22: -27.0084991, -19.5265179, -26.9995136, -19.5424309, -4.3359127, 4.3426304
23: -20.8755817, -12.5005035, -20.8285332, -12.5219765, -4.7955818, 4.7696228
24: -16.8822098, -7.6338596, -16.8317719, -7.6552944, -7.1747437, 7.1494713
25: -14.6443043, -6.9520712, -14.6266289, -6.9653306, -4.1964149, 4.1908169
26: -14.6220322, -7.7969537, -14.6107607, -7.8196950, -6.5253448, 6.5541191
27: -14.6419373, -9.5227213, -14.6177883, -9.5386524, -4.0648842, 4.0571327
28: -10.0377865, -1.4218202, -10.0125294, -1.4293417, -6.1520767, 6.1498871
29: -45.5967712, -36.8078270, -45.5658188, -36.8324280, -5.0123825, 5.0050526
30: -32.2183113, -22.9946671, -32.1531029, -23.0239906, -5.0203037, 4.9943867
31: -32.2602692, -23.5107574, -32.2092667, -23.5274448, -6.3192520, 6.2831383
32: 7.6959295, 13.6755581, 7.7251773, 13.6773643, -4.1802864, 4.1468353
33: 4.5854526, 16.3128586, 4.6239252, 16.3134346, -6.7156792, 6.6730728
34: 20.5147591, 30.9898796, 20.5526218, 30.9886265, -5.7534790, 5.7135677
35: 16.4767723, 26.8645210, 16.5237045, 26.8579082, -5.4561729, 5.4127522
36: 28.7952785, 35.1257210, 28.8126507, 35.1231155, -3.4383144, 3.4248981
37: 10.9971600, 20.1173115, 11.0426188, 20.1111870, -5.9833565, 5.9394569
38: 34.8387947, 43.7038383, 34.8681183, 43.6726570, -6.0459366, 6.0472717
39: 8.9639616, 18.5195007, 9.0000820, 18.4997368, -6.5452309, 6.5226097
40: 15.7626715, 25.1308022, 15.8059187, 25.1294174, -5.8311443, 5.7899036
41: 6.7078686, 13.2268858, 6.7477241, 13.2198448, -5.0376091, 5.0035286
42: -12.3983316, -3.4525018, -12.3850622, -3.4521940, -7.0609283, 7.0505753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6308369
time: 5.32 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6382295
time: 5.29 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5873756, -8.4760866, -21.5891876, -8.4790573, -10.4701309, 10.4497223
1: -21.4321213, -12.2287636, -21.4398766, -12.2369738, -5.2959995, 5.3030338
2: -12.3986912, -5.7745886, -12.4057016, -5.7779808, -4.2779808, 4.2780857
3: -12.0143270, -4.1555634, -12.0179386, -4.1676483, -5.3603516, 5.3773232
4: -10.2981749, 0.0369308, -10.3026304, 0.0041404, -6.0392609, 6.0822544
5: -13.5651894, -4.0341926, -13.5719433, -4.0453429, -6.1557007, 6.1601105
6: -8.3528681, 0.5443628, -8.3171768, 0.5451018, -6.4991913, 6.4600258
7: -32.1578331, -22.0514069, -32.1656036, -22.0571938, -5.8396225, 5.8536568
8: -18.8172703, -9.0650339, -18.8234901, -9.1068306, -5.2133331, 5.2631836
9: -5.3468108, 1.4063028, -5.3455830, 1.3937838, -4.0555630, 4.0688038
10: -36.1405067, -27.7473030, -36.1425133, -27.7645493, -5.2638111, 5.2752838
11: -55.1675415, -44.7709808, -55.1264496, -44.7889748, -4.9653492, 4.9478798
12: -11.5829935, -4.5783935, -11.5747108, -4.5760660, -6.2452354, 6.2361908
13: 0.8782057, 8.0353088, 0.8808296, 8.0131073, -5.3037605, 5.3244171
14: -71.0884247, -57.9405632, -71.0949707, -57.9582291, -8.2727814, 8.2832909
15: -8.9242020, 0.9366841, -8.9246998, 0.9013143, -4.8725605, 4.9154148
16: -33.5911331, -23.9671917, -33.5703735, -23.9673347, -6.4826851, 6.4633064
17: -88.6778107, -72.3848724, -88.6782608, -72.4197006, -8.1814461, 8.2215691
18: -4.2013907, 1.0740848, -4.1779275, 1.0637596, -3.4156532, 3.4014511
19: -30.5339222, -23.2005119, -30.5261612, -23.2013397, -4.6546936, 4.6446629
20: -11.1740999, -5.1526041, -11.1723824, -5.1511436, -4.9398804, 4.9414558
21: -43.5629501, -35.0483246, -43.5474396, -35.0519333, -4.2691422, 4.2554588
22: -27.0084991, -19.5265179, -27.0067005, -19.5395126, -4.3347626, 4.3447018
23: -20.8755817, -12.5005035, -20.8515396, -12.5072870, -4.7948837, 4.7764416
24: -16.8822098, -7.6338596, -16.8600311, -7.6399951, -7.1795731, 7.1670799
25: -14.6443043, -6.9520712, -14.6379108, -6.9579325, -4.1970730, 4.1950531
26: -14.6220322, -7.7969537, -14.6128550, -7.8146191, -6.5523300, 6.5827217
27: -14.6419373, -9.5227213, -14.6310949, -9.5305891, -4.0671425, 4.0645905
28: -10.0377865, -1.4218202, -10.0230312, -1.4188213, -6.1718330, 6.1670952
29: -45.5967712, -36.8078270, -45.5860481, -36.8217468, -5.0129395, 5.0150089
30: -32.2183113, -22.9946671, -32.1828308, -23.0038185, -5.0149956, 4.9976177
31: -32.2602692, -23.5107574, -32.2366714, -23.5171967, -6.3254662, 6.3053169
32: 7.6959295, 13.6755581, 7.7192421, 13.6802254, -4.1859589, 4.1556664
33: 4.5854526, 16.3128586, 4.6107407, 16.3190479, -6.7172031, 6.6786766
34: 20.5147591, 30.9898796, 20.5356884, 30.9973469, -5.7499809, 5.7183743
35: 16.4767723, 26.8645210, 16.5044193, 26.8678398, -5.4521103, 5.4180756
36: 28.7952785, 35.1257210, 28.8092594, 35.1256104, -3.4408264, 3.4269409
37: 10.9971600, 20.1173115, 11.0253191, 20.1181793, -5.9827080, 5.9494743
38: 34.8387947, 43.7038383, 34.8548508, 43.6871338, -6.0493469, 6.0496864
39: 8.9639616, 18.5195007, 8.9901943, 18.5047569, -6.5462990, 6.5317116
40: 15.7626715, 25.1308022, 15.7912979, 25.1380196, -5.8382149, 5.8029537
41: 6.7078686, 13.2268858, 6.7333636, 13.2274494, -5.0408478, 5.0132942
42: -12.3983316, -3.4525018, -12.3861265, -3.4491959, -7.0646896, 7.0519180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1599

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6308370
time: 5.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6382297
time: 5.48 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5619202, -8.4773140, -21.5878696, -8.4826908, -10.4001160, 10.4265366
1: -21.4270687, -12.2364712, -21.4411697, -12.2375145, -5.2573280, 5.2886429
2: -12.3929873, -5.7769599, -12.4009800, -5.7817779, -4.2516346, 4.2672272
3: -12.0093021, -4.1673479, -12.0018339, -4.1856570, -5.3477631, 5.3680916
4: -10.2836637, 0.0053809, -10.2882757, -0.0097315, -6.0287971, 6.0610142
5: -13.5544376, -4.0442963, -13.5684280, -4.0519595, -6.1253586, 6.1528015
6: -8.3185015, 0.5380002, -8.3148232, 0.5399177, -6.4665184, 6.4561691
7: -32.1552505, -22.0553112, -32.1637421, -22.0626106, -5.8142319, 5.8388672
8: -18.8048077, -9.1060982, -18.8006477, -9.1474400, -5.1862984, 5.2323685
9: -5.3305693, 1.3936331, -5.3428578, 1.3838549, -4.0270710, 4.0624218
10: -36.1342239, -27.7654953, -36.1356354, -27.7712212, -5.2445469, 5.2682972
11: -55.1317596, -44.8025970, -55.0843201, -44.8235550, -4.9560432, 4.9377136
12: -11.5788498, -4.5906925, -11.5670853, -4.5884728, -6.2399864, 6.2327995
13: 0.8906822, 8.0121489, 0.8917536, 7.9895201, -5.2758865, 5.3020325
14: -71.0820084, -57.9611969, -71.1048050, -57.9503479, -8.2216301, 8.2587662
15: -8.9091282, 0.9019375, -8.9135075, 0.8925285, -4.8616905, 4.8914986
16: -33.5558128, -23.9755249, -33.5380478, -24.0035286, -6.4453201, 6.4690857
17: -88.6771698, -72.4199905, -88.6863708, -72.4117584, -8.1822853, 8.2001572
18: -4.1770797, 1.0589998, -4.1526899, 1.0607822, -3.4014854, 3.3753452
19: -30.5245628, -23.2039490, -30.5019302, -23.2247849, -4.6491871, 4.6478634
20: -11.1713247, -5.1568689, -11.1661692, -5.1583943, -4.9317627, 4.9251423
21: -43.5464859, -35.0578613, -43.5143890, -35.0863495, -4.2678833, 4.2651138
22: -27.0048599, -19.5391235, -27.0002518, -19.5401688, -4.3381424, 4.3374271
23: -20.8512859, -12.5133362, -20.8275509, -12.5307693, -4.7739868, 4.7618561
24: -16.8588409, -7.6453123, -16.8329353, -7.6588340, -7.1532593, 7.1396217
25: -14.6369247, -6.9584475, -14.6214428, -6.9765959, -4.1927662, 4.1889591
26: -14.6177588, -7.8184299, -14.6165056, -7.8019571, -6.5426254, 6.5275497
27: -14.6302242, -9.5350361, -14.6107349, -9.5469666, -4.0600758, 4.0528870
28: -10.0220985, -1.4306667, -10.0113220, -1.4307612, -6.1497002, 6.1444321
29: -45.5831375, -36.8256836, -45.5611534, -36.8415108, -5.0041828, 5.0066071
30: -32.1859398, -23.0199184, -32.1457367, -23.0361004, -4.9943485, 4.9787540
31: -32.2348633, -23.5174713, -32.2017479, -23.5430756, -6.3010635, 6.2891731
32: 7.7183456, 13.6730137, 7.7100325, 13.6855288, -4.1628132, 4.1552753
33: 4.6096482, 16.3102932, 4.6113958, 16.3188210, -6.6904869, 6.6753159
34: 20.5353756, 30.9838066, 20.5700607, 30.9770622, -5.7460155, 5.7144794
35: 16.5048599, 26.8592873, 16.5305405, 26.8520203, -5.4415798, 5.4159069
36: 28.8102207, 35.1238976, 28.8128319, 35.1223984, -3.4321766, 3.4297295
37: 11.0252705, 20.1128159, 11.0284348, 20.1145973, -5.9572983, 5.9460793
38: 34.8629417, 43.6871758, 34.8752899, 43.6701813, -6.0324020, 6.0315933
39: 8.9976864, 18.5058441, 8.9833326, 18.5094299, -6.5179596, 6.5185089
40: 15.7926989, 25.1246166, 15.7966433, 25.1357193, -5.8069859, 5.7824650
41: 6.7330136, 13.2223425, 6.7327785, 13.2259235, -5.0177040, 5.0118141
42: -12.3867254, -3.4558735, -12.3896332, -3.4560645, -7.0555649, 7.0523262

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 535

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5781204, upper bound: 3.6374407
time: 5.40 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5845772, upper bound: 3.6374407
time: 5.10 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5619202, -8.4773140, -21.6047039, -8.4775391, -10.4392548, 10.4778366
1: -21.4270687, -12.2364712, -21.4446526, -12.2346458, -5.2750015, 5.3064194
2: -12.3929873, -5.7769599, -12.4041920, -5.7796865, -4.2692528, 4.2832775
3: -12.0093021, -4.1673479, -12.0082598, -4.1726427, -5.3543243, 5.3683891
4: -10.2836637, 0.0053809, -10.3034468, 0.0132430, -6.0296211, 6.0539742
5: -13.5544376, -4.0442963, -13.5752926, -4.0439162, -6.1366730, 6.1648674
6: -8.3185015, 0.5380002, -8.3339720, 0.5489074, -6.4718666, 6.4710388
7: -32.1552505, -22.0553112, -32.1653862, -22.0579166, -5.8246307, 5.8456554
8: -18.8048077, -9.1060982, -18.8194084, -9.1058626, -5.1993275, 5.2231464
9: -5.3305693, 1.3936331, -5.3509321, 1.3949559, -4.0289135, 4.0615082
10: -36.1342239, -27.7654953, -36.1420479, -27.7660942, -5.2498970, 5.2774067
11: -55.1317596, -44.8025970, -55.1137772, -44.8063583, -4.9471588, 4.9409618
12: -11.5788498, -4.5906925, -11.5739193, -4.5804272, -6.2419891, 6.2328339
13: 0.8906822, 8.0121489, 0.8812569, 8.0135498, -5.2928429, 5.3053093
14: -71.0820084, -57.9611969, -71.1081772, -57.9470367, -8.2448196, 8.2782516
15: -8.9091282, 0.9019375, -8.9288549, 0.9128876, -4.8645210, 4.8901539
16: -33.5558128, -23.9755249, -33.5647278, -23.9900265, -6.4482498, 6.4846458
17: -88.6771698, -72.4199905, -88.6882248, -72.4027023, -8.1945419, 8.2032127
18: -4.1770797, 1.0589998, -4.1790328, 1.0691831, -3.3965645, 3.3885231
19: -30.5245628, -23.2039490, -30.5137100, -23.2195339, -4.6469574, 4.6507854
20: -11.1713247, -5.1568689, -11.1672325, -5.1551409, -4.9440613, 4.9381371
21: -43.5464859, -35.0578613, -43.5317993, -35.0761452, -4.2627296, 4.2670708
22: -27.0048599, -19.5391235, -27.0074253, -19.5372581, -4.3370247, 4.3394947
23: -20.8512859, -12.5133362, -20.8505898, -12.5160732, -4.7732773, 4.7686749
24: -16.8588409, -7.6453123, -16.8611832, -7.6435428, -7.1581039, 7.1572304
25: -14.6369247, -6.9584475, -14.6327190, -6.9692068, -4.1934204, 4.1931896
26: -14.6177588, -7.8184299, -14.6186247, -7.7968864, -6.5696068, 6.5561676
27: -14.6302242, -9.5350361, -14.6240463, -9.5388927, -4.0623341, 4.0603333
28: -10.0220985, -1.4306667, -10.0217810, -1.4202302, -6.1694412, 6.1615486
29: -45.5831375, -36.8256836, -45.5813866, -36.8308449, -5.0047398, 5.0165596
30: -32.1859398, -23.0199184, -32.1755257, -23.0158844, -4.9890747, 4.9820004
31: -32.2348633, -23.5174713, -32.2291222, -23.5328388, -6.3072815, 6.3113289
32: 7.7183456, 13.6730137, 7.7040863, 13.6883984, -4.1684799, 4.1641159
33: 4.6096482, 16.3102932, 4.5982404, 16.3244286, -6.6920509, 6.6809158
34: 20.5353756, 30.9838066, 20.5530891, 30.9858055, -5.7425098, 5.7192726
35: 16.5048599, 26.8592873, 16.5112572, 26.8619270, -5.4375114, 5.4212265
36: 28.8102207, 35.1238976, 28.8094749, 35.1248894, -3.4346924, 3.4317493
37: 11.0252705, 20.1128159, 11.0111580, 20.1215935, -5.9566383, 5.9560699
38: 34.8629417, 43.6871758, 34.8620186, 43.6846542, -6.0358200, 6.0339966
39: 8.9976864, 18.5058441, 8.9734097, 18.5144520, -6.5190468, 6.5276108
40: 15.7926989, 25.1246166, 15.7819853, 25.1442947, -5.8140736, 5.7955456
41: 6.7330136, 13.2223425, 6.7184629, 13.2335129, -5.0209312, 5.0215530
42: -12.3867254, -3.4558735, -12.3906507, -3.4530299, -7.0593033, 7.0536575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 535

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5781204, upper bound: 3.6374406
time: 4.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5845772, upper bound: 3.6374409
time: 5.15 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5619202, -8.4773140, -21.5923271, -8.4819803, -10.4002762, 10.4302979
1: -21.4270687, -12.2364712, -21.4444542, -12.2340145, -5.2588329, 5.2900715
2: -12.3929873, -5.7769599, -12.4064054, -5.7798128, -4.2474670, 4.2665367
3: -12.0093021, -4.1673479, -12.0143566, -4.1762633, -5.3447189, 5.3688622
4: -10.2836637, 0.0053809, -10.2972317, -0.0066376, -6.0261459, 6.0628319
5: -13.5544376, -4.0442963, -13.5706854, -4.0487690, -6.1278992, 6.1546326
6: -8.3185015, 0.5380002, -8.3202543, 0.5493152, -6.4691696, 6.4552574
7: -32.1552505, -22.0553112, -32.1700058, -22.0543938, -5.8184738, 5.8423386
8: -18.8048077, -9.1060982, -18.8085709, -9.1419315, -5.1862221, 5.2336369
9: -5.3305693, 1.3936331, -5.3506198, 1.3862579, -4.0315342, 4.0737038
10: -36.1342239, -27.7654953, -36.1446228, -27.7574158, -5.2445908, 5.2648315
11: -55.1317596, -44.8025970, -55.1037025, -44.7879829, -4.9539127, 4.9186668
12: -11.5788498, -4.5906925, -11.5742970, -4.5791969, -6.2426949, 6.2351608
13: 0.8906822, 8.0121489, 0.8820283, 7.9941177, -5.2755089, 5.3055000
14: -71.0820084, -57.9611969, -71.1075592, -57.9476166, -8.2265892, 8.2633972
15: -8.9091282, 0.9019375, -8.9206600, 0.8949623, -4.8554611, 4.8873787
16: -33.5558128, -23.9755249, -33.5580215, -23.9808884, -6.4479866, 6.4684830
17: -88.6771698, -72.4199905, -88.6973572, -72.3931732, -8.1848679, 8.1974983
18: -4.1770797, 1.0589998, -4.1556153, 1.0659182, -3.4052467, 3.3773232
19: -30.5245628, -23.2039490, -30.5157185, -23.2060471, -4.6468525, 4.6407318
20: -11.1713247, -5.1568689, -11.1722994, -5.1487079, -4.9330902, 4.9234467
21: -43.5464859, -35.0578613, -43.5316772, -35.0595436, -4.2598495, 4.2484722
22: -27.0048599, -19.5391235, -27.0068226, -19.5299377, -4.3404255, 4.3358727
23: -20.8512859, -12.5133362, -20.8345432, -12.5167942, -4.7751446, 4.7570343
24: -16.8588409, -7.6453123, -16.8367615, -7.6489339, -7.1578979, 7.1394882
25: -14.6369247, -6.9584475, -14.6307402, -6.9601908, -4.1931858, 4.1821690
26: -14.6177588, -7.8184299, -14.6186914, -7.7983093, -6.5452118, 6.5284653
27: -14.6302242, -9.5350361, -14.6206760, -9.5314035, -4.0604210, 4.0490265
28: -10.0220985, -1.4306667, -10.0142717, -1.4278408, -6.1532631, 6.1475067
29: -45.5831375, -36.8256836, -45.5739937, -36.8183212, -5.0053539, 4.9970360
30: -32.1859398, -23.0199184, -32.1567039, -23.0145130, -4.9970703, 4.9697495
31: -32.2348633, -23.5174713, -32.2169952, -23.5212708, -6.2972298, 6.2805977
32: 7.7183456, 13.6730137, 7.7064400, 13.6860943, -4.1630344, 4.1586208
33: 4.6096482, 16.3102932, 4.5996361, 16.3220558, -6.6902256, 6.6869926
34: 20.5353756, 30.9838066, 20.5425854, 30.9956856, -5.7374687, 5.7141438
35: 16.5048599, 26.8592873, 16.5080452, 26.8646355, -5.4318790, 5.4169140
36: 28.8102207, 35.1238976, 28.7991314, 35.1309814, -3.4295521, 3.4300356
37: 11.0252705, 20.1128159, 11.0219002, 20.1184731, -5.9583740, 5.9510956
38: 34.8629417, 43.6871758, 34.8540726, 43.6854248, -6.0327301, 6.0366287
39: 8.9976864, 18.5058441, 8.9739857, 18.5102654, -6.5207214, 6.5292549
40: 15.7926989, 25.1246166, 15.7854872, 25.1374893, -5.8065662, 5.7925606
41: 6.7330136, 13.2223425, 6.7297411, 13.2288504, -5.0192413, 5.0137749
42: -12.3867254, -3.4558735, -12.3978395, -3.4431930, -7.0572891, 7.0500221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 535

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5781204, upper bound: 3.6374407
time: 4.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5845772, upper bound: 3.6374410
time: 4.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5619202, -8.4773140, -21.6091633, -8.4767857, -10.4394150, 10.4816284
1: -21.4270687, -12.2364712, -21.4479408, -12.2312059, -5.2765064, 5.3078384
2: -12.3929873, -5.7769599, -12.4095917, -5.7777228, -4.2650986, 4.2826023
3: -12.0093021, -4.1673479, -12.0208025, -4.1632552, -5.3512688, 5.3691444
4: -10.2836637, 0.0053809, -10.3124342, 0.0163522, -6.0269547, 6.0557899
5: -13.5544376, -4.0442963, -13.5775528, -4.0407043, -6.1392059, 6.1667137
6: -8.3185015, 0.5380002, -8.3394146, 0.5583106, -6.4745216, 6.4701233
7: -32.1552505, -22.0553112, -32.1716309, -22.0497189, -5.8288612, 5.8491001
8: -18.8048077, -9.1060982, -18.8272839, -9.1003571, -5.1992722, 5.2244091
9: -5.3305693, 1.3936331, -5.3586860, 1.3973298, -4.0333843, 4.0727978
10: -36.1342239, -27.7654953, -36.1510391, -27.7523251, -5.2499599, 5.2739487
11: -55.1317596, -44.8025970, -55.1331406, -44.7707863, -4.9450264, 4.9219227
12: -11.5788498, -4.5906925, -11.5810995, -4.5711637, -6.2446861, 6.2352066
13: 0.8906822, 8.0121489, 0.8715194, 8.0181570, -5.2924728, 5.3087769
14: -71.0820084, -57.9611969, -71.1109314, -57.9442978, -8.2497864, 8.2828865
15: -8.9091282, 0.9019375, -8.9360313, 0.9152737, -4.8582802, 4.8860359
16: -33.5558128, -23.9755249, -33.5847092, -23.9673920, -6.4509239, 6.4840317
17: -88.6771698, -72.4199905, -88.6991806, -72.3841248, -8.1971321, 8.2005577
18: -4.1770797, 1.0589998, -4.1819782, 1.0743146, -3.4003239, 3.3905315
19: -30.5245628, -23.2039490, -30.5274811, -23.2007961, -4.6446152, 4.6436558
20: -11.1713247, -5.1568689, -11.1733675, -5.1454549, -4.9453773, 4.9364300
21: -43.5464859, -35.0578613, -43.5490875, -35.0493393, -4.2546997, 4.2504292
22: -27.0048599, -19.5391235, -27.0140038, -19.5270195, -4.3393173, 4.3379498
23: -20.8512859, -12.5133362, -20.8575611, -12.5020275, -4.7744408, 4.7638588
24: -16.8588409, -7.6453123, -16.8650341, -7.6336436, -7.1627426, 7.1570930
25: -14.6369247, -6.9584475, -14.6420307, -6.9527841, -4.1938381, 4.1863918
26: -14.6177588, -7.8184299, -14.6207790, -7.7932167, -6.5722084, 6.5570679
27: -14.6302242, -9.5350361, -14.6339798, -9.5233383, -4.0626850, 4.0564842
28: -10.0220985, -1.4306667, -10.0247593, -1.4173279, -6.1730118, 6.1646347
29: -45.5831375, -36.8256836, -45.5942421, -36.8076477, -5.0059128, 5.0070038
30: -32.1859398, -23.0199184, -32.1865196, -22.9943352, -4.9917755, 4.9729977
31: -32.2348633, -23.5174713, -32.2444077, -23.5110092, -6.3034401, 6.3027878
32: 7.7183456, 13.6730137, 7.7005019, 13.6889601, -4.1687088, 4.1674519
33: 4.6096482, 16.3102932, 4.5864739, 16.3276711, -6.6917877, 6.6926155
34: 20.5353756, 30.9838066, 20.5256996, 31.0043640, -5.7339764, 5.7189484
35: 16.5048599, 26.8592873, 16.4887657, 26.8745575, -5.4278259, 5.4222317
36: 28.8102207, 35.1238976, 28.7957382, 35.1334686, -3.4320621, 3.4320564
37: 11.0252705, 20.1128159, 11.0046043, 20.1254692, -5.9577217, 5.9610825
38: 34.8629417, 43.6871758, 34.8408775, 43.6998825, -6.0361443, 6.0390320
39: 8.9976864, 18.5058441, 8.9640675, 18.5153046, -6.5218010, 6.5383873
40: 15.7926989, 25.1246166, 15.7708788, 25.1460876, -5.8136368, 5.8056107
41: 6.7330136, 13.2223425, 6.7153955, 13.2364578, -5.0224762, 5.0235329
42: -12.3867254, -3.4558735, -12.3988676, -3.4401846, -7.0610390, 7.0513535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 535

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5781204, upper bound: 3.6374409
time: 6.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5845772, upper bound: 3.6374412
time: 5.48 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5711842, -8.4744730, -21.5877476, -8.4826612, -10.4319992, 10.4241104
1: -21.4287968, -12.2341938, -21.4412079, -12.2377625, -5.2635880, 5.2897968
2: -12.3960075, -5.7740884, -12.4010563, -5.7817831, -4.2609615, 4.2660179
3: -12.0133085, -4.1585083, -12.0019712, -4.1856327, -5.3517265, 5.3769417
4: -10.2943420, 0.0263026, -10.2886906, -0.0097404, -6.0390511, 6.0823593
5: -13.5609341, -4.0370030, -13.5685959, -4.0519390, -6.1349831, 6.1509933
6: -8.3328705, 0.5433073, -8.3148460, 0.5401521, -6.4812012, 6.4617043
7: -32.1562080, -22.0567131, -32.1637268, -22.0643501, -5.8172531, 5.8434238
8: -18.8164101, -9.0703316, -18.8011436, -9.1474409, -5.1976471, 5.2682838
9: -5.3350859, 1.4032512, -5.3430104, 1.3838885, -4.0314045, 4.0720921
10: -36.1397247, -27.7590675, -36.1357841, -27.7711639, -5.2502747, 5.2702904
11: -55.1669731, -44.7890625, -55.0843239, -44.8229485, -4.9919796, 4.9509583
12: -11.5809202, -4.5831518, -11.5670872, -4.5883160, -6.2408943, 6.2396660
13: 0.8846473, 8.0314884, 0.8915039, 7.9895926, -5.2820244, 5.3218727
14: -71.0861130, -57.9543648, -71.1048508, -57.9502945, -8.2352409, 8.2639847
15: -8.9220886, 0.9235744, -8.9139061, 0.8925457, -4.8747139, 4.9134903
16: -33.5792122, -23.9671097, -33.5380630, -24.0031471, -6.4690247, 6.4768562
17: -88.6774445, -72.4169769, -88.6863174, -72.4130325, -8.1824379, 8.2052689
18: -4.2007470, 1.0639167, -4.1527233, 1.0610015, -3.4253941, 3.3801537
19: -30.5335884, -23.2010078, -30.5019836, -23.2247162, -4.6587582, 4.6504059
20: -11.1734238, -5.1556897, -11.1661787, -5.1591415, -4.9314766, 4.9294243
21: -43.5633163, -35.0510635, -43.5144043, -35.0860672, -4.2849846, 4.2718296
22: -27.0083084, -19.5384941, -27.0000668, -19.5401649, -4.3420830, 4.3374233
23: -20.8707676, -12.5043163, -20.8275795, -12.5304642, -4.7938099, 4.7714424
24: -16.8787384, -7.6378627, -16.8329887, -7.6585159, -7.1733704, 7.1470184
25: -14.6409645, -6.9558024, -14.6214695, -6.9765368, -4.1973190, 4.1917763
26: -14.6221619, -7.8150148, -14.6166124, -7.8023052, -6.5428047, 6.5459518
27: -14.6416073, -9.5297642, -14.6108246, -9.5467644, -4.0710506, 4.0582218
28: -10.0368824, -1.4221604, -10.0113382, -1.4304683, -6.1498375, 6.1519852
29: -45.5960922, -36.8214569, -45.5611725, -36.8413239, -5.0172310, 5.0099220
30: -32.2182312, -23.0042248, -32.1457024, -23.0355473, -5.0272331, 4.9947090
31: -32.2542191, -23.5131187, -32.2018356, -23.5428848, -6.3218117, 6.2930374
32: 7.7133918, 13.6752224, 7.7100277, 13.6856079, -4.1679745, 4.1574116
33: 4.6069069, 16.3115788, 4.6121330, 16.3188477, -6.6962090, 6.6765594
34: 20.5231781, 30.9887352, 20.5699844, 30.9772453, -5.7584286, 5.7190018
35: 16.4910145, 26.8643875, 16.5304661, 26.8522320, -5.4555836, 5.4209423
36: 28.8081112, 35.1255417, 28.8127899, 35.1224289, -3.4342089, 3.4312229
37: 11.0157509, 20.1163387, 11.0284004, 20.1147194, -5.9672394, 5.9493027
38: 34.8522034, 43.7010612, 34.8749237, 43.6701965, -6.0434265, 6.0458031
39: 8.9883671, 18.5176048, 8.9831514, 18.5094299, -6.5270920, 6.5279312
40: 15.7818136, 25.1295052, 15.7965565, 25.1357307, -5.8169975, 5.7905426
41: 6.7241435, 13.2266836, 6.7327466, 13.2260857, -5.0266838, 5.0161018
42: -12.3867283, -3.4531779, -12.3893929, -3.4560480, -7.0557785, 7.0581703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 535

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5901413, upper bound: 3.6374927
time: 5.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5965987, upper bound: 3.6374927
time: 5.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5711842, -8.4744730, -21.6045666, -8.4775028, -10.4711304, 10.4754562
1: -21.4287968, -12.2341938, -21.4446716, -12.2349319, -5.2812576, 5.3075619
2: -12.3960075, -5.7740884, -12.4042835, -5.7797060, -4.2785835, 4.2820854
3: -12.0133085, -4.1585083, -12.0083981, -4.1726260, -5.3582878, 5.3772392
4: -10.2943420, 0.0263026, -10.3038740, 0.0132291, -6.0398521, 6.0753231
5: -13.5609341, -4.0370030, -13.5754509, -4.0438929, -6.1463394, 6.1630592
6: -8.3328705, 0.5433073, -8.3339939, 0.5491508, -6.4865341, 6.4765625
7: -32.1562080, -22.0567131, -32.1653786, -22.0596619, -5.8276367, 5.8501854
8: -18.8164101, -9.0703316, -18.8199062, -9.1058445, -5.2106876, 5.2590618
9: -5.3350859, 1.4032512, -5.3510852, 1.3949630, -4.0332527, 4.0711746
10: -36.1397247, -27.7590675, -36.1421700, -27.7660599, -5.2557201, 5.2793980
11: -55.1669731, -44.7890625, -55.1137848, -44.8057823, -4.9830952, 4.9542046
12: -11.5809202, -4.5831518, -11.5738955, -4.5802855, -6.2429047, 6.2397118
13: 0.8846473, 8.0314884, 0.8809920, 8.0136185, -5.2989883, 5.3251534
14: -71.0861130, -57.9543648, -71.1082306, -57.9470177, -8.2584152, 8.2834625
15: -8.9220886, 0.9235744, -8.9292698, 0.9128528, -4.8775330, 4.9121361
16: -33.5792122, -23.9671097, -33.5647774, -23.9896679, -6.4719620, 6.4924240
17: -88.6774445, -72.4169769, -88.6881638, -72.4039993, -8.1946869, 8.2083359
18: -4.2007470, 1.0639167, -4.1790586, 1.0693982, -3.4204826, 3.3933277
19: -30.5335884, -23.2010078, -30.5137672, -23.2194481, -4.6565056, 4.6533279
20: -11.1734238, -5.1556897, -11.1672573, -5.1558895, -4.9437561, 4.9424114
21: -43.5633163, -35.0510635, -43.5318375, -35.0758705, -4.2798328, 4.2737885
22: -27.0083084, -19.5384941, -27.0072517, -19.5372696, -4.3409367, 4.3395004
23: -20.8707676, -12.5043163, -20.8506050, -12.5157480, -4.7931099, 4.7782536
24: -16.8787384, -7.6378627, -16.8612556, -7.6432467, -7.1782227, 7.1646500
25: -14.6409645, -6.9558024, -14.6327715, -6.9691286, -4.1979733, 4.1959991
26: -14.6221619, -7.8150148, -14.6187201, -7.7972336, -6.5697861, 6.5745621
27: -14.6416073, -9.5297642, -14.6241207, -9.5387039, -4.0733128, 4.0656796
28: -10.0368824, -1.4221604, -10.0218344, -1.4199193, -6.1695824, 6.1691666
29: -45.5960922, -36.8214569, -45.5814171, -36.8306656, -5.0177841, 5.0198822
30: -32.2182312, -23.0042248, -32.1755295, -23.0153542, -5.0219460, 4.9979496
31: -32.2542191, -23.5131187, -32.2292328, -23.5326118, -6.3280220, 6.3152008
32: 7.7133918, 13.6752224, 7.7040777, 13.6884804, -4.1736431, 4.1662521
33: 4.6069069, 16.3115788, 4.5989366, 16.3244705, -6.6976986, 6.6821594
34: 20.5231781, 30.9887352, 20.5530529, 30.9860039, -5.7549381, 5.7237949
35: 16.4910145, 26.8643875, 16.5111732, 26.8621521, -5.4515266, 5.4262581
36: 28.8081112, 35.1255417, 28.8094044, 35.1249313, -3.4367237, 3.4332676
37: 11.0157509, 20.1163387, 11.0110989, 20.1217003, -5.9665718, 5.9592934
38: 34.8522034, 43.7010612, 34.8616524, 43.6846466, -6.0468292, 6.0481987
39: 8.9883671, 18.5176048, 8.9732323, 18.5144348, -6.5281792, 6.5370560
40: 15.7818136, 25.1295052, 15.7819061, 25.1443367, -5.8240738, 5.8036137
41: 6.7241435, 13.2266836, 6.7184157, 13.2336807, -5.0299187, 5.0258484
42: -12.3867283, -3.4531779, -12.3904285, -3.4530292, -7.0595131, 7.0595131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 535

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5901413, upper bound: 3.6374929
time: 4.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5965987, upper bound: 3.6374929
time: 5.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -21.5711842, -8.4744730, -21.5922508, -8.4818926, -10.4321594, 10.4278946
1: -21.4287968, -12.2341938, -21.4444942, -12.2342796, -5.2650928, 5.2912159
2: -12.3960075, -5.7740884, -12.4064732, -5.7798166, -4.2568016, 4.2653313
3: -12.0133085, -4.1585083, -12.0145025, -4.1762099, -5.3486595, 5.3777084
4: -10.2943420, 0.0263026, -10.2975988, -0.0066381, -6.0363960, 6.0841751
5: -13.5609341, -4.0370030, -13.5708523, -4.0487432, -6.1375122, 6.1528206
6: -8.3328705, 0.5433073, -8.3202572, 0.5495266, -6.4838448, 6.4607887
7: -32.1562080, -22.0567131, -32.1700058, -22.0561485, -5.8214874, 5.8469048
8: -18.8164101, -9.0703316, -18.8090496, -9.1418982, -5.1975651, 5.2695198
9: -5.3350859, 1.4032512, -5.3507557, 1.3862669, -4.0358658, 4.0833683
10: -36.1397247, -27.7590675, -36.1447639, -27.7573967, -5.2503166, 5.2668285
11: -55.1669731, -44.7890625, -55.1036911, -44.7873917, -4.9898510, 4.9319191
12: -11.5809202, -4.5831518, -11.5742769, -4.5790429, -6.2436066, 6.2420311
13: 0.8846473, 8.0314884, 0.8817797, 7.9941907, -5.2816505, 5.3253441
14: -71.0861130, -57.9543648, -71.1076202, -57.9475822, -8.2402077, 8.2686234
15: -8.9220886, 0.9235744, -8.9210491, 0.8949709, -4.8685074, 4.9093666
16: -33.5792122, -23.9671097, -33.5580559, -23.9805069, -6.4716797, 6.4762421
17: -88.6774445, -72.4169769, -88.6972885, -72.3944550, -8.1850166, 8.2026100
18: -4.2007470, 1.0639167, -4.1556549, 1.0661297, -3.4291534, 3.3821259
19: -30.5335884, -23.2010078, -30.5157566, -23.2059689, -4.6564178, 4.6432724
20: -11.1734238, -5.1556897, -11.1723299, -5.1494894, -4.9327965, 4.9277287
21: -43.5633163, -35.0510635, -43.5317154, -35.0592613, -4.2769489, 4.2551937
22: -27.0083084, -19.5384941, -27.0066433, -19.5299225, -4.3443604, 4.3358727
23: -20.8707676, -12.5043163, -20.8345909, -12.5164604, -4.7949677, 4.7666283
24: -16.8787384, -7.6378627, -16.8368053, -7.6486158, -7.1780319, 7.1469078
25: -14.6409645, -6.9558024, -14.6307793, -6.9601107, -4.1977406, 4.1849728
26: -14.6221619, -7.8150148, -14.6187878, -7.7986345, -6.5453987, 6.5468483
27: -14.6416073, -9.5297642, -14.6207590, -9.5311975, -4.0714016, 4.0543556
28: -10.0368824, -1.4221604, -10.0143013, -1.4275469, -6.1534004, 6.1550713
29: -45.5960922, -36.8214569, -45.5740356, -36.8181381, -5.0183945, 5.0003624
30: -32.2182312, -23.0042248, -32.1566925, -23.0139542, -5.0299397, 4.9857159
31: -32.2542191, -23.5131187, -32.2170639, -23.5210648, -6.3179474, 6.2844696
32: 7.7133918, 13.6752224, 7.7064447, 13.6861572, -4.1681957, 4.1607704
33: 4.6069069, 16.3115788, 4.6003771, 16.3220730, -6.6959610, 6.6882477
34: 20.5231781, 30.9887352, 20.5425739, 30.9958763, -5.7499008, 5.7186680
35: 16.4910145, 26.8643875, 16.5079956, 26.8648701, -5.4458847, 5.4219475
36: 28.8081112, 35.1255417, 28.7990780, 35.1310196, -3.4315853, 3.4315205
37: 11.0157509, 20.1163387, 11.0218582, 20.1186028, -5.9683037, 5.9543114
38: 34.8522034, 43.7010612, 34.8537369, 43.6854134, -6.0437737, 6.0508232
39: 8.9883671, 18.5176048, 8.9738102, 18.5102501, -6.5298386, 6.5386887
40: 15.7818136, 25.1295052, 15.7854118, 25.1375237, -5.8165817, 5.8006325
41: 6.7241435, 13.2266836, 6.7297254, 13.2290354, -5.0282249, 5.0180664
42: -12.3867283, -3.4531779, -12.3975945, -3.4431853, -7.0574722, 7.0558548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 535

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5901413, upper bound: 3.6374931
time: 5.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5965987, upper bound: 3.6374928
time: 5.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -21.5711842, -8.4744730, -21.6090508, -8.4767179, -10.4713058, 10.4791870
1: -21.4287968, -12.2341938, -21.4479675, -12.2314796, -5.2827606, 5.3089828
2: -12.3960075, -5.7740884, -12.4096746, -5.7777185, -4.2744255, 4.2813931
3: -12.0133085, -4.1585083, -12.0209188, -4.1632476, -5.3552437, 5.3779984
4: -10.2943420, 0.0263026, -10.3128262, 0.0163774, -6.0372124, 6.0771332
5: -13.5609341, -4.0370030, -13.5777321, -4.0406971, -6.1488991, 6.1648941
6: -8.3328705, 0.5433073, -8.3394079, 0.5585302, -6.4891930, 6.4756508
7: -32.1562080, -22.0567131, -32.1716537, -22.0514793, -5.8318748, 5.8536472
8: -18.8164101, -9.0703316, -18.8278179, -9.1003246, -5.2106094, 5.2603016
9: -5.3350859, 1.4032512, -5.3588600, 1.3973736, -4.0377121, 4.0824680
10: -36.1397247, -27.7590675, -36.1511536, -27.7522850, -5.2557716, 5.2759323
11: -55.1669731, -44.7890625, -55.1331711, -44.7702103, -4.9809704, 4.9351749
12: -11.5809202, -4.5831518, -11.5810919, -4.5710030, -6.2456169, 6.2420807
13: 0.8846473, 8.0314884, 0.8712787, 8.0182381, -5.2986221, 5.3286400
14: -71.0861130, -57.9543648, -71.1109924, -57.9442215, -8.2633820, 8.2881050
15: -8.9220886, 0.9235744, -8.9363995, 0.9152679, -4.8713303, 4.9080143
16: -33.5792122, -23.9671097, -33.5847321, -23.9670105, -6.4746170, 6.4918060
17: -88.6774445, -72.4169769, -88.6991196, -72.3853912, -8.1972771, 8.2056847
18: -4.2007470, 1.0639167, -4.1820135, 1.0745227, -3.4242477, 3.3953285
19: -30.5335884, -23.2010078, -30.5275383, -23.2006950, -4.6541519, 4.6461964
20: -11.1734238, -5.1556897, -11.1733980, -5.1462383, -4.9450798, 4.9407234
21: -43.5633163, -35.0510635, -43.5491333, -35.0490570, -4.2717991, 4.2571526
22: -27.0083084, -19.5384941, -27.0138321, -19.5270290, -4.3432274, 4.3379402
23: -20.8707676, -12.5043163, -20.8576050, -12.5017071, -4.7942715, 4.7734432
24: -16.8787384, -7.6378627, -16.8650627, -7.6333241, -7.1828690, 7.1645126
25: -14.6409645, -6.9558024, -14.6420813, -6.9527082, -4.1983948, 4.1891994
26: -14.6221619, -7.8150148, -14.6209021, -7.7935557, -6.5723686, 6.5754623
27: -14.6416073, -9.5297642, -14.6340561, -9.5231352, -4.0736637, 4.0618095
28: -10.0368824, -1.4221604, -10.0247793, -1.4169959, -6.1731339, 6.1722527
29: -45.5960922, -36.8214569, -45.5942650, -36.8074532, -5.0189667, 5.0103168
30: -32.2182312, -23.0042248, -32.1865349, -22.9937763, -5.0246429, 4.9889584
31: -32.2542191, -23.5131187, -32.2444611, -23.5108261, -6.3241539, 6.3066483
32: 7.7133918, 13.6752224, 7.7004919, 13.6890335, -4.1738739, 4.1695957
33: 4.6069069, 16.3115788, 4.5872211, 16.3277111, -6.6974411, 6.6938477
34: 20.5231781, 30.9887352, 20.5256252, 31.0045853, -5.7463989, 5.7234707
35: 16.4910145, 26.8643875, 16.4886894, 26.8747864, -5.4418316, 5.4272633
36: 28.8081112, 35.1255417, 28.7956905, 35.1335068, -3.4340906, 3.4335556
37: 11.0157509, 20.1163387, 11.0045547, 20.1255970, -5.9676437, 5.9643059
38: 34.8522034, 43.7010612, 34.8405075, 43.6998749, -6.0471802, 6.0532417
39: 8.9883671, 18.5176048, 8.9638720, 18.5153084, -6.5309296, 6.5478020
40: 15.7818136, 25.1295052, 15.7707520, 25.1461296, -5.8236618, 5.8136787
41: 6.7241435, 13.2266836, 6.7153721, 13.2366104, -5.0314636, 5.0278130
42: -12.3867283, -3.4531779, -12.3985968, -3.4401624, -7.0612450, 7.0571899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=76, inp2_unstable=76, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1112
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1364

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 535

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5901413, upper bound: 3.6374929
time: 5.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5965987, upper bound: 3.6374932
time: 4.88 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 12.77 seconds
IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6339338
IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6381818
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6332456
IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6374970
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6339340
IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6381819
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6332459
IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5965989, upper bound: 3.6374973
IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6242553
IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6316467
IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6307889
IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5853167, upper bound: 3.6381817
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5827478, upper bound: 3.6374971
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5892050, upper bound: 3.6374971
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5901413, upper bound: 3.6374971
IS_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5965988, upper bound: 3.6374971
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5827478, upper bound: 3.6374974
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5892050, upper bound: 3.6374974
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5901413, upper bound: 3.6374974
IS_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5965988, upper bound: 3.6374974
IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6260930
IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6334849
IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6260929
IS_A2_B2_A2_B1_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6334851
IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5745822, upper bound: 3.6307845
IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5745822, upper bound: 3.6381770
IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5745822, upper bound: 3.6307843
IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5745792, upper bound: 3.6251582
IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6260929
IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6334852
IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6260931
IS_A2_B2_A2_B1_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5806332, upper bound: 3.6334854
IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5745822, upper bound: 3.6307843
IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5745822, upper bound: 3.6381771
IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5745822, upper bound: 3.6307845
IS_A2_B2_A2_B1_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5745792, upper bound: 3.6251583
IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6261447
IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6335370
IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6261449
IS_A2_B2_A2_B1_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6335369
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6308365
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6382292
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6308367
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6382294
IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6261450
IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6335371
IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6261452
IS_A2_B2_A2_B1_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5926484, upper bound: 3.6335373
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6308369
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6382295
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6308370
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5865979, upper bound: 3.6382297
IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5781204, upper bound: 3.6374407
IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5845772, upper bound: 3.6374407
IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5781204, upper bound: 3.6374406
IS_A2_B2_A2_B2_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5845772, upper bound: 3.6374409
IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5781204, upper bound: 3.6374407
IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5845772, upper bound: 3.6374410
IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5781204, upper bound: 3.6374409
IS_A2_B2_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5845772, upper bound: 3.6374412
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5901413, upper bound: 3.6374927
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5965987, upper bound: 3.6374927
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5901413, upper bound: 3.6374929
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5965987, upper bound: 3.6374929
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5901413, upper bound: 3.6374931
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5965987, upper bound: 3.6374928
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5901413, upper bound: 3.6374929
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 12.77
Output dim: 38, lower bound: -3.5965987, upper bound: 3.6374932

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5604401, -8.4840517, -21.5497856, -8.4852219, -10.4041824, 10.3647232
1: -21.4258575, -12.2339048, -21.4366894, -12.2486811, -5.2384739, 5.2735500
2: -12.3920135, -5.7795715, -12.3972998, -5.7838650, -4.2484989, 4.2591934
3: -12.0032148, -4.1774926, -11.9997730, -4.1877441, -5.3419037, 5.3562584
4: -10.2711401, -0.0071149, -10.2805767, -0.0181570, -6.0069962, 6.0514336
5: -13.5512571, -4.0496302, -13.5624466, -4.0617056, -6.1157188, 6.1475945
6: -8.3191814, 0.5289683, -8.3043442, 0.5338780, -6.4490395, 6.4163933
7: -32.1542702, -22.0547333, -32.1558876, -22.0963402, -5.7985229, 5.8528233
8: -18.7861576, -9.1424847, -18.7982864, -9.1235867, -5.1881104, 5.1958447
9: -5.3335476, 1.3855377, -5.3183298, 1.3881464, -4.0391598, 4.0265198
10: -36.1276169, -27.7589760, -36.1345177, -27.7839279, -5.2193184, 5.2631073
11: -55.1019592, -44.8018341, -55.1151085, -44.8622856, -4.8956947, 4.9826889
12: -11.5740070, -4.5942235, -11.5653887, -4.5988941, -6.2186279, 6.1977119
13: 0.8954894, 7.9918184, 0.9012778, 7.9903574, -5.2683067, 5.2728310
14: -71.0794525, -57.9507370, -71.1004639, -57.9731674, -8.1984215, 8.2621765
15: -8.8945312, 0.8947153, -8.9118433, 0.8907309, -4.8421001, 4.8943901
16: -33.5396576, -23.9891930, -33.5399780, -24.0129795, -6.4357300, 6.4331017
17: -88.6737976, -72.3971558, -88.6761627, -72.4893112, -8.1259842, 8.2432022
18: -4.1510458, 1.0607269, -4.1644273, 1.0401332, -3.3575878, 3.3931465
19: -30.5130386, -23.2090473, -30.5037231, -23.2301750, -4.6315632, 4.6448689
20: -11.1708679, -5.1575270, -11.1632576, -5.1662717, -4.9115181, 4.9188976
21: -43.5285950, -35.0656281, -43.5255890, -35.0936317, -4.2410736, 4.2588100
22: -26.9973049, -19.5301208, -26.9951992, -19.5678177, -4.3084736, 4.3534718
23: -20.8329430, -12.5248909, -20.8216019, -12.5378475, -4.7595234, 4.7482510
24: -16.8339329, -7.6573329, -16.8345814, -7.6664748, -7.1255569, 7.1328278
25: -14.6288881, -6.9626799, -14.6076899, -6.9822035, -4.1832714, 4.1734352
26: -14.6148205, -7.8057380, -14.6107941, -7.8233008, -6.5134850, 6.5327911
27: -14.6171494, -9.5363207, -14.6139545, -9.5693150, -4.0326271, 4.0604095
28: -10.0124159, -1.4414296, -10.0152740, -1.4302133, -6.1332779, 6.1267281
29: -45.5628738, -36.8227806, -45.5669098, -36.8734894, -4.9593868, 5.0233727
30: -32.1561050, -23.0310020, -32.1748543, -23.0522118, -4.9485092, 4.9965515
31: -32.2132797, -23.5260925, -32.2030373, -23.5537529, -6.2754440, 6.2818794
32: 7.7069449, 13.6697578, 7.7256427, 13.6792336, -4.1692657, 4.1298866
33: 4.6015291, 16.3046970, 4.6619358, 16.3128242, -6.7159977, 6.6223946
34: 20.5441132, 30.9753742, 20.5945053, 30.9686184, -5.7465363, 5.6918201
35: 16.5100479, 26.8485641, 16.5636406, 26.8457928, -5.4519596, 5.3851871
36: 28.8008232, 35.1208992, 28.8407173, 35.1193771, -3.4520264, 3.4045696
37: 11.0241318, 20.1059036, 11.0700607, 20.1089287, -5.9642525, 5.9000053
38: 34.8629341, 43.6746483, 34.9211426, 43.6712532, -6.0537643, 5.9902039
39: 8.9833775, 18.5016727, 9.0359783, 18.5124321, -6.5494308, 6.4753342
40: 15.7883492, 25.1163864, 15.8213854, 25.1351929, -5.8148308, 5.7436333
41: 6.7312536, 13.2141581, 6.7500315, 13.2234631, -5.0155640, 4.9791489
42: -12.3972492, -3.4591131, -12.3669062, -3.4569907, -7.0568466, 7.0161018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=75, inp2_unstable=76, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 535

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5845774, upper bound: 3.6267400
time: 6.15 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5845774, upper bound: 3.6331965
time: 5.22 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5773010, -8.4788942, -21.5497856, -8.4852219, -10.4254074, 10.3737793
1: -21.4293461, -12.2310991, -21.4366894, -12.2486811, -5.2416325, 5.2766285
2: -12.3952103, -5.7774730, -12.3972998, -5.7838650, -4.2489376, 4.2612095
3: -12.0096464, -4.1645212, -11.9997730, -4.1877441, -5.3486099, 5.3692589
4: -10.2863226, 0.0158755, -10.2805767, -0.0181570, -6.0239487, 6.0762291
5: -13.5581074, -4.0415535, -13.5624466, -4.0617056, -6.1173782, 6.1484871
6: -8.3383284, 0.5379789, -8.3043442, 0.5338780, -6.4681549, 6.4259949
7: -32.1559181, -22.0500679, -32.1558876, -22.0963402, -5.7978439, 5.8557377
8: -18.8048973, -9.1009121, -18.7982864, -9.1235867, -5.2077484, 5.2377739
9: -5.3416505, 1.3966296, -5.3183298, 1.3881464, -4.0479488, 4.0380611
10: -36.1340179, -27.7538795, -36.1345177, -27.7839279, -5.2243462, 5.2643661
11: -55.1314163, -44.7846413, -55.1151085, -44.8622856, -4.9260082, 5.0008640
12: -11.5808544, -4.5861893, -11.5653887, -4.5988941, -6.2243996, 6.2054253
13: 0.8849627, 8.0158405, 0.9012778, 7.9903574, -5.2787781, 5.2969589
14: -71.0827942, -57.9474564, -71.1004639, -57.9731674, -8.2015381, 8.2689743
15: -8.9098969, 0.9150310, -8.9118433, 0.8907309, -4.8593597, 4.9158707
16: -33.5663414, -23.9757042, -33.5399780, -24.0129795, -6.4610939, 6.4458427
17: -88.6756287, -72.3880997, -88.6761627, -72.4893112, -8.1258163, 8.2522278
18: -4.1773949, 1.0691304, -4.1644273, 1.0401332, -3.3842010, 3.4016895
19: -30.5247993, -23.2038002, -30.5037231, -23.2301750, -4.6420841, 4.6502132
20: -11.1719332, -5.1542811, -11.1632576, -5.1662717, -4.9142189, 4.9208927
21: -43.5459633, -35.0553970, -43.5255890, -35.0936317, -4.2594490, 4.2700768
22: -27.0044918, -19.5271816, -26.9951992, -19.5678177, -4.3153839, 4.3572006
23: -20.8559380, -12.5101871, -20.8216019, -12.5378475, -4.7830372, 4.7642422
24: -16.8621883, -7.6420512, -16.8345814, -7.6664748, -7.1536026, 7.1480904
25: -14.6401653, -6.9552832, -14.6076899, -6.9822035, -4.1952286, 4.1818161
26: -14.6169195, -7.8006525, -14.6107941, -7.8233008, -6.5195808, 6.5372963
27: -14.6304626, -9.5282488, -14.6139545, -9.5693150, -4.0459385, 4.0685272
28: -10.0229206, -1.4308958, -10.0152740, -1.4302133, -6.1321526, 6.1282272
29: -45.5831299, -36.8120918, -45.5669098, -36.8734894, -4.9790096, 5.0335922
30: -32.1859283, -23.0108204, -32.1748543, -23.0522118, -4.9801369, 5.0196571
31: -32.2406845, -23.5158367, -32.2030373, -23.5537529, -6.3017693, 6.2921944
32: 7.7010078, 13.6726227, 7.7256427, 13.6792336, -4.1753197, 4.1327763
33: 4.5883532, 16.3103065, 4.6619358, 16.3128242, -6.7251949, 6.6275978
34: 20.5271435, 30.9840717, 20.5945053, 30.9686184, -5.7635498, 5.7005463
35: 16.4907494, 26.8584881, 16.5636406, 26.8457928, -5.4712334, 5.3950844
36: 28.7974434, 35.1233978, 28.8407173, 35.1193771, -3.4537525, 3.4067898
37: 11.0068274, 20.1128845, 11.0700607, 20.1089287, -5.9822998, 5.9073830
38: 34.8496819, 43.6891098, 34.9211426, 43.6712532, -6.0676231, 6.0051003
39: 8.9734554, 18.5067253, 9.0359783, 18.5124321, -6.5585480, 6.4764023
40: 15.7737217, 25.1249886, 15.8213854, 25.1351929, -5.8307037, 5.7535362
41: 6.7168798, 13.2217522, 6.7500315, 13.2234631, -5.0297127, 4.9867859
42: -12.3982821, -3.4561133, -12.3669062, -3.4569907, -7.0569229, 7.0185890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=75, inp2_unstable=76, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 535

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5845774, upper bound: 3.6309870
time: 5.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5845774, upper bound: 3.6374447
time: 5.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5697269, -8.4812508, -21.5425224, -8.4853573, -10.4537354, 10.3534546
1: -21.4275875, -12.2315865, -21.4359550, -12.2514191, -5.2574921, 5.2736893
2: -12.3950119, -5.7767143, -12.3963223, -5.7868547, -4.2573471, 4.2595673
3: -12.0071983, -4.1686988, -11.9992208, -4.1956019, -5.3339157, 5.3644981
4: -10.2818432, 0.0138271, -10.2789803, -0.0388678, -5.9837799, 6.0713482
5: -13.5577564, -4.0423169, -13.5602779, -4.0681963, -6.1240540, 6.1466141
6: -8.3335285, 0.5342647, -8.2907400, 0.5338166, -6.4635429, 6.4011650
7: -32.1552620, -22.0561218, -32.1553078, -22.0963326, -5.7815247, 5.8544464
8: -18.7977581, -9.1067572, -18.7982502, -9.1588516, -5.1574173, 5.2312298
9: -5.3380842, 1.3951681, -5.3176231, 1.3790383, -4.0435085, 4.0352516
10: -36.1331520, -27.7525101, -36.1319923, -27.7898483, -5.2443352, 5.2617531
11: -55.1371803, -44.7883148, -55.0802917, -44.8630981, -4.9255676, 4.9608612
12: -11.5761127, -4.5866823, -11.5633354, -4.6021676, -6.2161026, 6.1999550
13: 0.8894489, 8.0111179, 0.9013723, 7.9729605, -5.2511749, 5.2922783
14: -71.0835190, -57.9438934, -71.0976791, -57.9784355, -8.2164612, 8.2632523
15: -8.9075012, 0.9163117, -8.9085445, 0.8695211, -4.8311520, 4.9127254
16: -33.5630760, -23.9808044, -33.5177307, -24.0135155, -6.4876060, 6.4179001
17: -88.6740952, -72.3941193, -88.6761856, -72.4919739, -8.0996971, 8.2497978
18: -4.1746044, 1.0656331, -4.1417656, 1.0403466, -3.3798428, 3.3754025
19: -30.5220299, -23.2061348, -30.4962692, -23.2306175, -4.6351852, 4.6396065
20: -11.1729689, -5.1563506, -11.1622257, -5.1662650, -4.9138832, 4.9267330
21: -43.5454025, -35.0588226, -43.5098877, -35.0947838, -4.2577667, 4.2497730
22: -27.0007477, -19.5294666, -26.9934959, -19.5689125, -4.3109016, 4.3518600
23: -20.8524475, -12.5158672, -20.8032722, -12.5394564, -4.7774734, 4.7391205
24: -16.8538017, -7.6498628, -16.8164062, -7.6660986, -7.1457520, 7.1217461
25: -14.6329212, -6.9600306, -14.6052351, -6.9831715, -4.1867104, 4.1718178
26: -14.6192226, -7.8023224, -14.6097927, -7.8249307, -6.5097084, 6.5545921
27: -14.6285229, -9.5310221, -14.6046715, -9.5702991, -4.0415287, 4.0571175
28: -10.0272083, -1.4328947, -10.0019512, -1.4317718, -6.1333504, 6.1382751
29: -45.5758514, -36.8185425, -45.5547943, -36.8737526, -4.9777031, 5.0140476
30: -32.1883926, -23.0152473, -32.1430855, -23.0560169, -4.9730186, 4.9806175
31: -32.2325668, -23.5217419, -32.1862183, -23.5542603, -6.2949562, 6.2692032
32: 7.7019877, 13.6719551, 7.7302217, 13.6790581, -4.1742020, 4.1194534
33: 4.5988364, 16.3059921, 4.6640296, 16.3121185, -6.7209034, 6.6076202
34: 20.5318527, 30.9803391, 20.6057014, 30.9685459, -5.7586555, 5.6722870
35: 16.4962254, 26.8536739, 16.5759163, 26.8461418, -5.4660912, 5.3710232
36: 28.7987423, 35.1225510, 28.8421669, 35.1188660, -3.4537163, 3.3997345
37: 11.0146379, 20.1094227, 11.0779409, 20.1082382, -5.9732552, 5.8854904
38: 34.8521233, 43.6885567, 34.9237823, 43.6579132, -6.0513535, 5.9860535
39: 8.9740620, 18.5134411, 9.0368147, 18.5009518, -6.5493927, 6.4833069
40: 15.7773342, 25.1212635, 15.8301744, 25.1310806, -5.8205070, 5.7452965
41: 6.7223530, 13.2185020, 6.7580085, 13.2230406, -5.0239296, 4.9698372
42: -12.3972425, -3.4564044, -12.3670769, -3.4589555, -7.0561256, 7.0144653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=75, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1767

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5886917, upper bound: 3.6323491
time: 5.08 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5956723, upper bound: 3.6323491
time: 5.89 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5865803, -8.4760675, -21.5425224, -8.4853573, -10.4749756, 10.3625031
1: -21.4310799, -12.2287951, -21.4359550, -12.2514191, -5.2606506, 5.2767372
2: -12.3982229, -5.7746058, -12.3963223, -5.7868547, -4.2577820, 4.2615585
3: -12.0136356, -4.1556745, -11.9992208, -4.1956019, -5.3406410, 5.3775063
4: -10.2970276, 0.0367746, -10.2789803, -0.0388678, -6.0007210, 6.0961494
5: -13.5645905, -4.0342393, -13.5602779, -4.0681963, -6.1257057, 6.1474991
6: -8.3526869, 0.5432765, -8.2907400, 0.5338166, -6.4826622, 6.4107437
7: -32.1569099, -22.0514717, -32.1553078, -22.0963326, -5.7808228, 5.8573418
8: -18.8165436, -9.0651417, -18.7982502, -9.1588516, -5.1770592, 5.2732220
9: -5.3461804, 1.4062407, -5.3176231, 1.3790383, -4.0522804, 4.0467873
10: -36.1395187, -27.7474136, -36.1319923, -27.7898483, -5.2493057, 5.2630024
11: -55.1666222, -44.7711105, -55.0802917, -44.8630981, -4.9558754, 4.9790382
12: -11.5829010, -4.5786724, -11.5633354, -4.6021676, -6.2218552, 6.2076530
13: 0.8789415, 8.0351534, 0.9013723, 7.9729605, -5.2616577, 5.3164597
14: -71.0869293, -57.9406128, -71.0976791, -57.9784355, -8.2195702, 8.2699890
15: -8.9228210, 0.9366274, -8.9085445, 0.8695211, -4.8484097, 4.9342022
16: -33.5897827, -23.9673023, -33.5177307, -24.0135155, -6.5129852, 6.4306335
17: -88.6759109, -72.3850708, -88.6761856, -72.4919739, -8.0995331, 8.2587891
18: -4.2010698, 1.0740366, -4.1417656, 1.0403466, -3.4065838, 3.3839474
19: -30.5338268, -23.2008743, -30.4962692, -23.2306175, -4.6457825, 4.6449509
20: -11.1740246, -5.1530972, -11.1622257, -5.1662650, -4.9165840, 4.9287434
21: -43.5628357, -35.0486145, -43.5098877, -35.0947838, -4.2761402, 4.2610283
22: -27.0079498, -19.5265617, -26.9934959, -19.5689125, -4.3178482, 4.3555851
23: -20.8754311, -12.5011444, -20.8032722, -12.5394564, -4.8009872, 4.7550945
24: -16.8820820, -7.6345654, -16.8164062, -7.6660986, -7.1737671, 7.1370087
25: -14.6442127, -6.9526176, -14.6052351, -6.9831715, -4.1986694, 4.1802025
26: -14.6213341, -7.7972536, -14.6097927, -7.8249307, -6.5158119, 6.5590858
27: -14.6418228, -9.5229607, -14.6046715, -9.5702991, -4.0548229, 4.0652275
28: -10.0376759, -1.4223733, -10.0019512, -1.4317718, -6.1322060, 6.1397095
29: -45.5960960, -36.8078651, -45.5547943, -36.8737526, -4.9973183, 5.0242558
30: -32.2182121, -22.9951286, -32.1430855, -23.0560169, -5.0046463, 5.0037079
31: -32.2600517, -23.5114670, -32.1862183, -23.5542603, -6.3213692, 6.2795296
32: 7.6960382, 13.6748219, 7.7302217, 13.6790581, -4.1802635, 4.1223488
33: 4.5856099, 16.3116112, 4.6640296, 16.3121185, -6.7301197, 6.6128044
34: 20.5149422, 30.9890194, 20.6057014, 30.9685459, -5.7756767, 5.6810150
35: 16.4769096, 26.8635864, 16.5759163, 26.8461418, -5.4853630, 5.3809052
36: 28.7953396, 35.1250458, 28.8421669, 35.1188660, -3.4554386, 3.4019403
37: 10.9973469, 20.1164131, 11.0779409, 20.1082382, -5.9913292, 5.8928642
38: 34.8389015, 43.7030258, 34.9237823, 43.6579132, -6.0651855, 6.0009346
39: 8.9641247, 18.5184422, 9.0368147, 18.5009518, -6.5585213, 6.4843788
40: 15.7628260, 25.1298752, 15.8301744, 25.1310806, -5.8362923, 5.7551937
41: 6.7080092, 13.2260914, 6.7580085, 13.2230406, -5.0380592, 4.9774742
42: -12.3982792, -3.4533842, -12.3670769, -3.4589555, -7.0561790, 7.0169449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=75, inp2_unstable=76, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1767

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5886917, upper bound: 3.6365510
time: 5.12 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5956723, upper bound: 3.6365510
time: 5.00 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5604401, -8.4840517, -21.5542889, -8.4844904, -10.4043427, 10.3684578
1: -21.4258575, -12.2339048, -21.4399872, -12.2451820, -5.2399807, 5.2749767
2: -12.3920135, -5.7795715, -12.4027195, -5.7818608, -4.2443371, 4.2585163
3: -12.0032148, -4.1774926, -12.0123081, -4.1783438, -5.3388367, 5.3570557
4: -10.2711401, -0.0071149, -10.2895489, -0.0150387, -6.0043411, 6.0532475
5: -13.5512571, -4.0496302, -13.5646915, -4.0585237, -6.1182594, 6.1494141
6: -8.3191814, 0.5289683, -8.3097410, 0.5432746, -6.4517136, 6.4154816
7: -32.1542702, -22.0547333, -32.1621475, -22.0881195, -5.8027496, 5.8562469
8: -18.7861576, -9.1424847, -18.8062229, -9.1180887, -5.1880341, 5.1971512
9: -5.3335476, 1.3855377, -5.3261600, 1.3905404, -4.0436611, 4.0377121
10: -36.1276169, -27.7589760, -36.1435089, -27.7701797, -5.2193565, 5.2596188
11: -55.1019592, -44.8018341, -55.1344910, -44.8267403, -4.8935280, 4.9636421
12: -11.5740070, -4.5942235, -11.5726290, -4.5896258, -6.2213478, 6.2000885
13: 0.8954894, 7.9918184, 0.8915400, 7.9949312, -5.2679176, 5.2763176
14: -71.0794525, -57.9507370, -71.1032257, -57.9704285, -8.2033806, 8.2667923
15: -8.8945312, 0.8947153, -8.9189930, 0.8931608, -4.8358383, 4.8902664
16: -33.5396576, -23.9891930, -33.5600052, -23.9903374, -6.4385490, 6.4324684
17: -88.6737976, -72.3971558, -88.6871490, -72.4707031, -8.1285553, 8.2404900
18: -4.1510458, 1.0607269, -4.1673861, 1.0452347, -3.3613186, 3.3951321
19: -30.5130386, -23.2090473, -30.5174904, -23.2114296, -4.6292362, 4.6377373
20: -11.1708679, -5.1575270, -11.1693745, -5.1565981, -4.9128418, 4.9171658
21: -43.5285950, -35.0656281, -43.5428925, -35.0668106, -4.2330399, 4.2421570
22: -26.9973049, -19.5301208, -27.0017948, -19.5575848, -4.3107643, 4.3519211
23: -20.8329430, -12.5248909, -20.8285637, -12.5238104, -4.7606888, 4.7433968
24: -16.8339329, -7.6573329, -16.8384247, -7.6565685, -7.1301956, 7.1326675
25: -14.6288881, -6.9626799, -14.6169662, -6.9658098, -4.1836891, 4.1666222
26: -14.6148205, -7.8057380, -14.6129608, -7.8196192, -6.5160599, 6.5337067
27: -14.6171494, -9.5363207, -14.6238871, -9.5537577, -4.0329380, 4.0565224
28: -10.0124159, -1.4414296, -10.0182056, -1.4273407, -6.1368294, 6.1298103
29: -45.5628738, -36.8227806, -45.5797729, -36.8503494, -4.9605312, 5.0137997
30: -32.1561050, -23.0310020, -32.1858444, -23.0306511, -4.9511871, 4.9875069
31: -32.2132797, -23.5260925, -32.2182655, -23.5319328, -6.2716560, 6.2732849
32: 7.7069449, 13.6697578, 7.7220559, 13.6798153, -4.1694870, 4.1332321
33: 4.6015291, 16.3046970, 4.6502810, 16.3160515, -6.7156410, 6.6340866
34: 20.5441132, 30.9753742, 20.5670834, 30.9871941, -5.7379799, 5.6915169
35: 16.5100479, 26.8485641, 16.5412064, 26.8584137, -5.4422035, 5.3861885
36: 28.8008232, 35.1208992, 28.8270187, 35.1279831, -3.4493694, 3.4048824
37: 11.0241318, 20.1059036, 11.0635157, 20.1127968, -5.9653702, 5.9049911
38: 34.8629341, 43.6746483, 34.8999596, 43.6865005, -6.0540466, 5.9953079
39: 8.9833775, 18.5016727, 9.0267048, 18.5132866, -6.5521507, 6.4860649
40: 15.7883492, 25.1163864, 15.8102894, 25.1369743, -5.8143845, 5.7536163
41: 6.7312536, 13.2141581, 6.7470183, 13.2263851, -5.0171165, 4.9811134
42: -12.3972492, -3.4591131, -12.3750868, -3.4441195, -7.0585899, 7.0137939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=75, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=4, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 535

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5845774, upper bound: 3.6267402
time: 5.93 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5845774, upper bound: 3.6331968
time: 5.86 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5773010, -8.4788942, -21.5542889, -8.4844904, -10.4255600, 10.3774986
1: -21.4293461, -12.2310991, -21.4399872, -12.2451820, -5.2431469, 5.2780552
2: -12.3952103, -5.7774730, -12.4027195, -5.7818608, -4.2447720, 4.2605228
3: -12.0096464, -4.1645212, -12.0123081, -4.1783438, -5.3455505, 5.3700485
4: -10.2863226, 0.0158755, -10.2895489, -0.0150387, -6.0212898, 6.0780430
5: -13.5581074, -4.0415535, -13.5646915, -4.0585237, -6.1199226, 6.1503067
6: -8.3383284, 0.5379789, -8.3097410, 0.5432746, -6.4708252, 6.4250717
7: -32.1559181, -22.0500679, -32.1621475, -22.0881195, -5.8020706, 5.8591881
8: -18.8048973, -9.1009121, -18.8062229, -9.1180887, -5.2076721, 5.2390461
9: -5.3416505, 1.3966296, -5.3261600, 1.3905404, -4.0524483, 4.0492554
10: -36.1340179, -27.7538795, -36.1435089, -27.7701797, -5.2243729, 5.2608757
11: -55.1314163, -44.7846413, -55.1344910, -44.8267403, -4.9238415, 4.9818192
12: -11.5808544, -4.5861893, -11.5726290, -4.5896258, -6.2271156, 6.2078018
13: 0.8849627, 8.0158405, 0.8915400, 7.9949312, -5.2783813, 5.3004417
14: -71.0827942, -57.9474564, -71.1032257, -57.9704285, -8.2065086, 8.2735977
15: -8.9098969, 0.9150310, -8.9189930, 0.8931608, -4.8531303, 4.9117508
16: -33.5663414, -23.9757042, -33.5600052, -23.9903374, -6.4639091, 6.4452057
17: -88.6756287, -72.3880997, -88.6871490, -72.4707031, -8.1283875, 8.2495384
18: -4.1773949, 1.0691304, -4.1673861, 1.0452347, -3.3879280, 3.4036789
19: -30.5247993, -23.2038002, -30.5174904, -23.2114296, -4.6397495, 4.6430817
20: -11.1719332, -5.1542811, -11.1693745, -5.1565981, -4.9155502, 4.9191570
21: -43.5459633, -35.0553970, -43.5428925, -35.0668106, -4.2514153, 4.2534294
22: -27.0044918, -19.5271816, -27.0017948, -19.5575848, -4.3176708, 4.3556423
23: -20.8559380, -12.5101871, -20.8285637, -12.5238104, -4.7842026, 4.7593994
24: -16.8621883, -7.6420512, -16.8384247, -7.6565685, -7.1582336, 7.1479225
25: -14.6401653, -6.9552832, -14.6169662, -6.9658098, -4.1956482, 4.1750050
26: -14.6169195, -7.8006525, -14.6129608, -7.8196192, -6.5221558, 6.5382118
27: -14.6304626, -9.5282488, -14.6238871, -9.5537577, -4.0462513, 4.0646439
28: -10.0229206, -1.4308958, -10.0182056, -1.4273407, -6.1357002, 6.1313095
29: -45.5831299, -36.8120918, -45.5797729, -36.8503494, -4.9801464, 5.0240192
30: -32.1859283, -23.0108204, -32.1858444, -23.0306511, -4.9828186, 5.0106373
31: -32.2406845, -23.5158367, -32.2182655, -23.5319328, -6.2979393, 6.2835922
32: 7.7010078, 13.6726227, 7.7220559, 13.6798153, -4.1755371, 4.1361237
33: 4.5883532, 16.3103065, 4.6502810, 16.3160515, -6.7248535, 6.6392860
34: 20.5271435, 30.9840717, 20.5670834, 30.9871941, -5.7550125, 5.7002411
35: 16.4907494, 26.8584881, 16.5412064, 26.8584137, -5.4614677, 5.3960838
36: 28.7974434, 35.1233978, 28.8270187, 35.1279831, -3.4510956, 3.4070930
37: 11.0068274, 20.1128845, 11.0635157, 20.1127968, -5.9834061, 5.9123688
38: 34.8496819, 43.6891098, 34.8999596, 43.6865005, -6.0679398, 6.0102005
39: 8.9734554, 18.5067253, 9.0267048, 18.5132866, -6.5612717, 6.4871407
40: 15.7737217, 25.1249886, 15.8102894, 25.1369743, -5.8302612, 5.7635193
41: 6.7168798, 13.2217522, 6.7470183, 13.2263851, -5.0312691, 4.9887505
42: -12.3982821, -3.4561133, -12.3750868, -3.4441195, -7.0586662, 7.0162811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=75, inp2_unstable=76, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 535

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5845774, upper bound: 3.6309871
time: 6.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 38, lower bound: -3.5845739, upper bound: 3.6244255
time: 5.73 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -21.5697269, -8.4812508, -21.5470181, -8.4845743, -10.4538879, 10.3572083
1: -21.4275875, -12.2315865, -21.4392567, -12.2478981, -5.2590065, 5.2751198
2: -12.3950119, -5.7767143, -12.4017029, -5.7848673, -4.2531891, 4.2588863
3: -12.0071983, -4.1686988, -12.0117426, -4.1861687, -5.3308563, 5.3652802
4: -10.2818432, 0.0138271, -10.2879629, -0.0357738, -5.9811134, 6.0731602
5: -13.5577564, -4.0423169, -13.5625381, -4.0650082, -6.1265907, 6.1484375
6: -8.3335285, 0.5342647, -8.2961578, 0.5431879, -6.4662132, 6.4002457
7: -32.1552620, -22.0561218, -32.1615677, -22.0881195, -5.7857361, 5.8579025
8: -18.7977581, -9.1067572, -18.8061657, -9.1533127, -5.1573353, 5.2325039
9: -5.3380842, 1.3951681, -5.3254790, 1.3814548, -4.0480042, 4.0464439
10: -36.1331520, -27.7525101, -36.1409836, -27.7760696, -5.2443657, 5.2582779
11: -55.1371803, -44.7883148, -55.0997162, -44.8275681, -4.9233837, 4.9418068
12: -11.5761127, -4.5866823, -11.5705681, -4.5928936, -6.2188187, 6.2023163
13: 0.8894489, 8.0111179, 0.8916259, 7.9775629, -5.2507858, 5.2957878
14: -71.0835190, -57.9438934, -71.1004715, -57.9756851, -8.2214241, 8.2678719
15: -8.9075012, 0.9163117, -8.9156971, 0.8719544, -4.8249149, 4.9086018
16: -33.5630760, -23.9808044, -33.5377579, -23.9908581, -6.4904099, 6.4172592
17: -88.6740952, -72.3941193, -88.6871719, -72.4733887, -8.1022682, 8.2471046
18: -4.1746044, 1.0656331, -4.1446924, 1.0454412, -3.3835678, 3.3773537
19: -30.5220299, -23.2061348, -30.5100517, -23.2118874, -4.6328487, 4.6324730
20: -11.1729689, -5.1563506, -11.1683760, -5.1565895, -4.9152222, 4.9250145
21: -43.5454025, -35.0588226, -43.5271683, -35.0679703, -4.2497292, 4.2331276
22: -27.0007477, -19.5294666, -27.0000725, -19.5586758, -4.3131790, 4.3502903
23: -20.8524475, -12.5158672, -20.8102341, -12.5254612, -4.7786484, 4.7342758
24: -16.8538017, -7.6498628, -16.8202114, -7.6561999, -7.1503906, 7.1215782
25: -14.6329212, -6.9600306, -14.6145391, -6.9667692, -4.1871243, 4.1650085
26: -14.6192226, -7.8023224, -14.6119814, -7.8212638, -6.5122910, 6.5554962
27: -14.6285229, -9.5310221, -14.6146107, -9.5547371, -4.0418377, 4.0532494
28: -10.0272083, -1.4328947, -10.0049353, -1.4288294, -6.1369209, 6.1413574
29: -45.5758514, -36.8185425, -45.5676727, -36.8505783, -4.9788437, 5.0044537
30: -32.1883926, -23.0152473, -32.1540794, -23.0345001, -4.9757137, 4.9715958
31: -32.2325668, -23.5217419, -32.2014732, -23.5324326, -6.2911339, 6.2606049
32: 7.7019877, 13.6719551, 7.7266674, 13.6796055, -4.1744137, 4.1228046
33: 4.5988364, 16.3059921, 4.6523628, 16.3153458, -6.7205544, 6.6193161
34: 20.5318527, 30.9803391, 20.5782528, 30.9871483, -5.7501202, 5.6719875
35: 16.4962254, 26.8536739, 16.5535049, 26.8587723, -5.4563427, 5.3720207
36: 28.7987423, 35.1225510, 28.8284512, 35.1274529, -3.4510670, 3.4000397
37: 11.0146379, 20.1094227, 11.0714273, 20.1121216, -5.9743690, 5.8904800
38: 34.8521233, 43.6885567, 34.9025879, 43.6731949, -6.0516663, 5.9911575
39: 8.9740620, 18.5134411, 9.0275612, 18.5017967, -6.5521240, 6.4940453
40: 15.7773342, 25.1212635, 15.8190880, 25.1328545, -5.8200684, 5.7552719
41: 6.7223530, 13.2185020, 6.7549877, 13.2259598, -5.0254898, 4.9718018
42: -12.3972425, -3.4564044, -12.3752909, -3.4460938, -7.0578308, 7.0121460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=75, inp2_unstable=76, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1767

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5886917, upper bound: 3.6323494
time: 5.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5956723, upper bound: 3.6323494
time: 6.10 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5865803, -8.4760675, -21.5470181, -8.4845743, -10.4751282, 10.3662491
1: -21.4310799, -12.2287951, -21.4392567, -12.2478981, -5.2621651, 5.2781639
2: -12.3982229, -5.7746058, -12.4017029, -5.7848673, -4.2536201, 4.2608719
3: -12.0136356, -4.1556745, -12.0117426, -4.1861687, -5.3375816, 5.3782883
4: -10.2970276, 0.0367746, -10.2879629, -0.0357738, -5.9980507, 6.0979576
5: -13.5645905, -4.0342393, -13.5625381, -4.0650082, -6.1282463, 6.1493225
6: -8.3526869, 0.5432765, -8.2961578, 0.5431879, -6.4853210, 6.4098320
7: -32.1569099, -22.0514717, -32.1615677, -22.0881195, -5.7850380, 5.8608170
8: -18.8165436, -9.0651417, -18.8061657, -9.1533127, -5.1769733, 5.2744675
9: -5.3461804, 1.4062407, -5.3254790, 1.3814548, -4.0567760, 4.0579796
10: -36.1395187, -27.7474136, -36.1409836, -27.7760696, -5.2493324, 5.2595272
11: -55.1666222, -44.7711105, -55.0997162, -44.8275681, -4.9536896, 4.9599838
12: -11.5829010, -4.5786724, -11.5705681, -4.5928936, -6.2245750, 6.2100182
13: 0.8789415, 8.0351534, 0.8916259, 7.9775629, -5.2612648, 5.3199615
14: -71.0869293, -57.9406128, -71.1004715, -57.9756851, -8.2245407, 8.2746239
15: -8.9228210, 0.9366274, -8.9156971, 0.8719544, -4.8422012, 4.9300823
16: -33.5897827, -23.9673023, -33.5377579, -23.9908581, -6.5157814, 6.4299965
17: -88.6759109, -72.3850708, -88.6871719, -72.4733887, -8.1021042, 8.2561035
18: -4.2010698, 1.0740366, -4.1446924, 1.0454412, -3.4103050, 3.3859043
19: -30.5338268, -23.2008743, -30.5100517, -23.2118874, -4.6434383, 4.6378193
20: -11.1740246, -5.1530972, -11.1683760, -5.1565895, -4.9179230, 4.9270248
21: -43.5628357, -35.0486145, -43.5271683, -35.0679703, -4.2681046, 4.2443848
22: -27.0079498, -19.5265617, -27.0000725, -19.5586758, -4.3201199, 4.3540134
23: -20.8754311, -12.5011444, -20.8102341, -12.5254612, -4.8021584, 4.7502594
24: -16.8820820, -7.6345654, -16.8202114, -7.6561999, -7.1783981, 7.1368370
25: -14.6442127, -6.9526176, -14.6145391, -6.9667692, -4.1990814, 4.1733913
26: -14.6213341, -7.7972536, -14.6119814, -7.8212638, -6.5183907, 6.5599899
27: -14.6418228, -9.5229607, -14.6146107, -9.5547371, -4.0551338, 4.0613594
28: -10.0376759, -1.4223733, -10.0049353, -1.4288294, -6.1357689, 6.1427956
29: -45.5960960, -36.8078651, -45.5676727, -36.8505783, -4.9984512, 5.0146580
30: -32.2182121, -22.9951286, -32.1540794, -23.0345001, -5.0073452, 4.9947052
31: -32.2600517, -23.5114670, -32.2014732, -23.5324326, -6.3175049, 6.2709312
32: 7.6960382, 13.6748219, 7.7266674, 13.6796055, -4.1804752, 4.1257000
33: 4.5856099, 16.3116112, 4.6523628, 16.3153458, -6.7297935, 6.6244965
34: 20.5149422, 30.9890194, 20.5782528, 30.9871483, -5.7671490, 5.6807156
35: 16.4769096, 26.8635864, 16.5535049, 26.8587723, -5.4756050, 5.3819008
36: 28.7953396, 35.1250458, 28.8284512, 35.1274529, -3.4527874, 3.4022350
37: 10.9973469, 20.1164131, 11.0714273, 20.1121216, -5.9924431, 5.8978539
38: 34.8389015, 43.7030258, 34.9025879, 43.6731949, -6.0655327, 6.0060349
39: 8.9641247, 18.5184422, 9.0275612, 18.5017967, -6.5612526, 6.4951134
40: 15.7628260, 25.1298752, 15.8190880, 25.1328545, -5.8358574, 5.7651691
41: 6.7080092, 13.2260914, 6.7549877, 13.2259598, -5.0396309, 4.9794312
42: -12.3982792, -3.4533842, -12.3752909, -3.4460938, -7.0578995, 7.0146217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=75, inp2_unstable=76, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=4, delta_unstable=43

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1112
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1364

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1767

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5886917, upper bound: 3.6365513
time: 5.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 38, lower bound: -3.5956723, upper bound: 3.6365513
time: 5.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -21.5773010, -8.4788942, -21.5666218, -8.4800406, -10.4645386, 10.4251022
1: -21.4293461, -12.2310991, -21.4401703, -12.2458916, -5.2592964, 5.2943974
2: -12.3952103, -5.7774730, -12.4005070, -5.7817798, -4.2665443, 4.2772751
3: -12.0096464, -4.1645212, -12.0062132, -4.1747456, -5.3551712, 5.3695450
4: -10.2863226, 0.0158755, -10.2957773, 0.0048300, -6.0247765, 6.0691719
5: -13.5581074, -4.0415535, -13.5692844, -4.0536427, -6.1286888, 6.1605453
6: -8.3383284, 0.5379789, -8.3234959, 0.5428777, -6.4735031, 6.4408722
7: -32.1559181, -22.0500679, -32.1575050, -22.0916862, -5.8082161, 5.8625069
8: -18.8048973, -9.1009121, -18.8170166, -9.0819998, -5.2208138, 5.2285500
9: -5.3416505, 1.3966296, -5.3263884, 1.3992236, -4.0498028, 4.0371323
10: -36.1340179, -27.7538795, -36.1409035, -27.7788410, -5.2296944, 5.2734375
11: -55.1314163, -44.7846413, -55.1445694, -44.8450928, -4.9171162, 5.0041122
12: -11.5808544, -4.5861893, -11.5721998, -4.5908766, -6.2263947, 6.2054825
13: 0.8849627, 8.0158405, 0.8907709, 8.0143890, -5.2957458, 5.3002396
14: -71.0827942, -57.9474564, -71.1038666, -57.9698563, -8.2246895, 8.2884598
15: -8.9098969, 0.9150310, -8.9271755, 0.9110293, -4.8621941, 4.9145031
16: -33.5663414, -23.9757042, -33.5667038, -23.9994774, -6.4640350, 6.4614258
17: -88.6756287, -72.3880997, -88.6780014, -72.4802399, -8.1380730, 8.2552834
18: -4.1773949, 1.0691304, -4.1908832, 1.0485342, -3.3792763, 3.4149399
19: -30.5247993, -23.2038002, -30.5155277, -23.2249012, -4.6398449, 4.6531696
20: -11.1719332, -5.1542811, -11.1642981, -5.1630192, -4.9265137, 4.9338703
21: -43.5459633, -35.0553970, -43.5429916, -35.0834427, -4.2542934, 4.2720451
22: -27.0044918, -19.5271816, -27.0023842, -19.5649300, -4.3142719, 4.3592625
23: -20.8559380, -12.5101871, -20.8446274, -12.5231304, -4.7823257, 4.7710648
24: -16.8621883, -7.6420512, -16.8628483, -7.6511889, -7.1584320, 7.1657028
25: -14.6401653, -6.9552832, -14.6189671, -6.9748197, -4.1958847, 4.1860504
26: -14.6169195, -7.8006525, -14.6128874, -7.8182354, -6.5465775, 6.5658913
27: -14.6304626, -9.5282488, -14.6272631, -9.5612526, -4.0481949, 4.0759792
28: -10.0229206, -1.4308958, -10.0257559, -1.4196960, -6.1518784, 6.1453362
29: -45.5831299, -36.8120918, -45.5871429, -36.8628616, -4.9795685, 5.0435619
30: -32.1859283, -23.0108204, -32.2046967, -23.0320816, -4.9748249, 5.0229034
31: -32.2406845, -23.5158367, -32.2304840, -23.5434990, -6.3079605, 6.3143921
32: 7.7010078, 13.6726227, 7.7196856, 13.6821251, -4.1809807, 4.1416111
33: 4.5883532, 16.3103065, 4.6487775, 16.3184166, -6.7267857, 6.6331635
34: 20.5271435, 30.9840717, 20.5775700, 30.9773369, -5.7600574, 5.7053566
35: 16.4907494, 26.8584881, 16.5443287, 26.8556900, -5.4671688, 5.4004173
36: 28.7974434, 35.1233978, 28.8373108, 35.1218796, -3.4562578, 3.4088078
37: 11.0068274, 20.1128845, 11.0527039, 20.1159096, -5.9816246, 5.9174194
38: 34.8496819, 43.6891098, 34.9079285, 43.6857147, -6.0710411, 6.0074692
39: 8.9734554, 18.5067253, 9.0260201, 18.5174522, -6.5595856, 6.4855194
40: 15.7737217, 25.1249886, 15.8068352, 25.1437874, -5.8377743, 5.7665100
41: 6.7168798, 13.2217522, 6.7356853, 13.2310562, -5.0329514, 4.9965515
42: -12.3982821, -3.4561133, -12.3679466, -3.4539590, -7.0606956, 7.0199089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=75, inp2_unstable=76, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=76, inp2_unstable=76, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=6, inp2_unstable=6, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=5, inp2_unstable=5, delta_unstable=43

Time for backsubstitution: 2.10 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 18.95 + 1781.90 = 1800.85 seconds
