## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 1800 seconds
Split limit: 100


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7104340, 14.7104340)
1: (1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5414047, 9.5414085)
2: (1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0458145, 10.0458145)
3: (-7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7421951, 9.7421951)
4: (2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8557358, 10.8557358)
5: (-4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4473877, 10.4473877)
6: (-29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0734482, 12.0734482)
7: (-3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.1195145, 10.1195145)
8: (-9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5771904, 12.5771866)
9: (0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9932938, 13.9932938)
10: (-11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5265808, 16.5265808)
11: (-11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3363876, 10.3363876)
12: (-24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8874283, 12.8874283)
13: (-13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8037796, 19.8037796)
14: (-30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8365479, 20.8365479)
15: (-6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4528122, 11.4528084)
16: (-10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1598053, 14.1598053)
17: (-32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2701340, 16.2701340)
18: (-9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3259888, 18.3259964)
19: (-4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810)
20: (-6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8263474, 12.8263474)
21: (-5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408)
22: (-6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1884232, 13.1884232)
23: (-7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3579941, 13.3579941)
24: (-5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9462967, 13.9462967)
25: (-6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7357941, 13.7357941)
26: (-12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.9073334, 19.9073334)
27: (-8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9180984, 15.9180984)
28: (-6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4516220, 14.4516220)
29: (-9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6801910, 11.6801910)
30: (-14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4101791, 11.4101715)
31: (-7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800)
32: (-20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9830246, 10.9830170)
33: (-36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8694611, 17.8694611)
34: (-40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2516861, 12.2516861)
35: (-27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.1248703, 13.1248703)
36: (-23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.7439423, 13.7439423)
37: (-44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7356262, 17.7356262)
38: (-28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.3029785, 19.3029709)
39: (-32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.1012955, 17.1012955)
40: (-42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0566711, 15.0566711)
41: (-24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2872772, 13.2872772)
42: (-25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3308830, 9.3308830)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.81 + 29.79 = 32.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -6.0971536, upper bound: 6.0971536

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1658
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1658

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0891810, upper bound: 6.0966543
time: 19.50 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0966543, upper bound: 6.0891810
time: 16.08 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 35.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 35.70
Output dim: 4, lower bound: -6.0891810, upper bound: 6.0966543
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 35.70
Output dim: 4, lower bound: -6.0966543, upper bound: 6.0891810

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7042313, 14.7065201
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5401917, 9.5404015
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0458107, 10.0457573
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7358513, 9.7390289
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8558655, 10.8553123
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4439240, 10.4449692
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0709076, 12.0733910
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.1193237, 10.1192741
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5765114, 12.5754700
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9845886, 13.9872894
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5238266, 16.5244522
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3249741, 10.3241806
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8844604, 12.8836670
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8033447, 19.8034363
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8207703, 20.8118668
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4495926, 11.4494438
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1588211, 14.1587677
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2545624, 16.2458801
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3267365, 18.3225861
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8240128, 12.8237762
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1884232, 13.1884155
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3579865, 13.3579865
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9440918, 13.9427795
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7350540, 13.7350464
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.9002075, 19.8953094
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9160156, 15.9148636
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4487076, 14.4483490
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6774597, 11.6763382
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4024506, 11.4015427
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9772339, 10.9820251
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8557892, 17.8608780
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2413101, 12.2448463
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.1246643, 13.1246872
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.7433701, 13.7433624
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7355270, 17.7344818
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.3013000, 19.3019409
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0977936, 17.0984344
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0540924, 15.0547714
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2814484, 13.2844467
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3269386, 9.3333702

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0679192, upper bound: 6.0958978
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0884237, upper bound: 6.0754266
time: 16.97 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7065201, 14.7042313
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5404053, 9.5401878
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0457573, 10.0458107
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7390327, 9.7358475
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8553085, 10.8558655
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4449692, 10.4439240
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0733948, 12.0709076
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.1192780, 10.1193275
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5754738, 12.5765152
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9872894, 13.9845886
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5244522, 16.5238266
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3241806, 10.3249779
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8836670, 12.8844566
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8034363, 19.8033447
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8118744, 20.8207703
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4494400, 11.4495888
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1587677, 14.1588211
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2458801, 16.2545624
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3225861, 18.3267365
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8237762, 12.8240128
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1884155, 13.1884232
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3579865, 13.3579865
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9427795, 13.9440842
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7350464, 13.7350540
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8953094, 19.9002075
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9148636, 15.9160156
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4483566, 14.4487076
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6763382, 11.6774597
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4015350, 11.4024582
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9820251, 10.9772339
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8608856, 17.8557968
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2448502, 12.2413101
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.1246872, 13.1246643
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.7433624, 13.7433701
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7344818, 17.7355270
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.3019409, 19.3013077
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0984344, 17.0977936
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0547714, 15.0540924
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2844467, 13.2814407
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3333664, 9.3269386

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0754266, upper bound: 6.0884237
time: 9.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0958978, upper bound: 6.0679192
time: 11.28 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.80 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.80
Output dim: 4, lower bound: -6.0679192, upper bound: 6.0958978
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.80
Output dim: 4, lower bound: -6.0884237, upper bound: 6.0754266
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.80
Output dim: 4, lower bound: -6.0754266, upper bound: 6.0884237
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.80
Output dim: 4, lower bound: -6.0958978, upper bound: 6.0679192

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7070885, 14.7089539
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5325241, 9.5339127
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0310364, 10.0333900
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7175598, 9.7236710
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8393097, 10.8421707
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4226151, 10.4271202
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0755005, 12.0787926
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0828667, 10.0885735
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5432663, 12.5487785
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9770203, 13.9805756
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5104980, 16.5132980
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3405380, 10.3426666
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8902321, 12.8840294
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8326035, 19.8271332
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8110199, 20.8034439
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4501495, 11.4478912
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1288071, 14.1336288
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2624969, 16.2517090
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3269958, 18.3228302
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8324738, 12.8336411
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1735535, 13.1707230
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3589630, 13.3588867
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9471512, 13.9460373
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7305756, 13.7300644
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8971710, 19.8921051
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9329071, 15.9349365
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4422760, 14.4410477
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6643982, 11.6607285
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4096146, 11.4095345
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9747467, 10.9795036
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8357468, 17.8371048
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2121429, 12.2099190
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0921326, 13.0860329
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.7114944, 13.7052002
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7058487, 17.6996155
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2679749, 19.2617264
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0908813, 17.0903168
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0592194, 15.0611801
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2726059, 13.2741394
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3285866, 9.3351974

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1762

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0601349, upper bound: 6.0956789
time: 15.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0676972, upper bound: 6.0881434
time: 18.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7066689, 14.7093735
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5336990, 9.5327377
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0334473, 10.0309792
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7204895, 9.7207413
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8427277, 10.8387566
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4260788, 10.4236565
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0763092, 12.0779839
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0886192, 10.0828171
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5498199, 12.5422249
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9778748, 13.9797287
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5126724, 16.5111237
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3434677, 10.3397484
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8848152, 12.8894386
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8270340, 19.8326950
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8123474, 20.8021240
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4480362, 11.4500046
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1336899, 14.1287537
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2603989, 16.2538071
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3269806, 18.3228378
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8338776, 12.8322372
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1707230, 13.1735535
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3588867, 13.3589630
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9473419, 13.9458466
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7300720, 13.7305679
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8969879, 19.8922729
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9360962, 15.9317551
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4414062, 14.4419174
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6618500, 11.6632767
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4104538, 11.4087029
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9747086, 10.9795418
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8320236, 17.8408356
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2063828, 12.2156868
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0860138, 13.0921593
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.7052078, 13.7114868
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7006607, 17.7047958
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2610931, 19.2686081
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0896759, 17.0915222
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0605011, 15.0598984
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2711411, 13.2756119
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3287659, 9.3350143

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1762

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0806464, upper bound: 6.0752061
time: 16.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0882037, upper bound: 6.0676597
time: 14.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7093773, 14.7066650
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5327377, 9.5336990
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0309830, 10.0334435
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7207413, 9.7204895
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8387604, 10.8427277
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4236603, 10.4260788
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0779800, 12.0763092
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0828209, 10.0886230
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5422211, 12.5498199
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9797287, 13.9778748
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5111237, 16.5126724
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3397446, 10.3434639
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8894386, 12.8848190
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8326950, 19.8270340
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8021240, 20.8123474
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4500046, 11.4480400
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1287537, 14.1336899
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2538071, 16.2603989
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3228455, 18.3269806
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8322372, 12.8338776
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1735535, 13.1707230
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3589630, 13.3588867
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9458389, 13.9473419
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7305679, 13.7300720
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8922729, 19.8969879
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9317474, 15.9360962
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4419174, 14.4413986
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6632767, 11.6618500
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4086990, 11.4104500
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9795456, 10.9747124
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8408432, 17.8320236
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2156830, 12.2063828
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0921631, 13.0860100
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.7114868, 13.7052078
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7047958, 17.7006683
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2686005, 19.2610931
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0915222, 17.0896759
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0598984, 15.0605011
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2756042, 13.2711411
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3350143, 9.3287659

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1762

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0676597, upper bound: 6.0882036
time: 17.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0752062, upper bound: 6.0806464
time: 20.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7089500, 14.7070847
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5339127, 9.5325241
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0333939, 10.0310326
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7236710, 9.7175598
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8421783, 10.8393135
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4271240, 10.4226151
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0787888, 12.0755005
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0885735, 10.0828705
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5487747, 12.5432663
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9805756, 13.9770203
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5132980, 16.5105057
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3426743, 10.3405457
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8840294, 12.8902283
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8271255, 19.8325958
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8034515, 20.8110199
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4478912, 11.4501534
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1336288, 14.1288071
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2517090, 16.2624969
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3228302, 18.3269882
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8336411, 12.8324738
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1707230, 13.1735535
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3588867, 13.3589630
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9460373, 13.9471512
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7300644, 13.7305756
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8921051, 19.8971558
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9349365, 15.9329071
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4410477, 14.4422760
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6607285, 11.6643982
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4095383, 11.4096184
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9795074, 10.9747505
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8371048, 17.8357544
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2099152, 12.2121468
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0860291, 13.0921364
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.7052002, 13.7114944
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6996155, 17.7058487
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2617340, 19.2679749
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0903168, 17.0908813
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0611801, 15.0592194
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2741394, 13.2726059
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3351974, 9.3285866

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1762

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0881434, upper bound: 6.0676972
time: 21.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0956789, upper bound: 6.0601349
time: 15.34 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 39.03 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 39.03
Output dim: 4, lower bound: -6.0601349, upper bound: 6.0956789
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 39.03
Output dim: 4, lower bound: -6.0676972, upper bound: 6.0881434
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 39.03
Output dim: 4, lower bound: -6.0806464, upper bound: 6.0752061
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 39.03
Output dim: 4, lower bound: -6.0882037, upper bound: 6.0676597
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 39.03
Output dim: 4, lower bound: -6.0676597, upper bound: 6.0882036
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 39.03
Output dim: 4, lower bound: -6.0752062, upper bound: 6.0806464
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 39.03
Output dim: 4, lower bound: -6.0881434, upper bound: 6.0676972
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 39.03
Output dim: 4, lower bound: -6.0956789, upper bound: 6.0601349

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7061043, 14.7083740
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5302849, 9.5326538
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0297546, 10.0326767
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7171707, 9.7232704
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8371124, 10.8408699
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4211426, 10.4262886
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0753860, 12.0791588
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0787392, 10.0862427
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5358810, 12.5446968
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9754028, 13.9795609
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5052643, 16.5103683
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3402939, 10.3431511
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8907318, 12.8836479
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8359222, 19.8263702
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8076935, 20.8014145
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4502068, 11.4477882
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1228867, 14.1302261
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2625580, 16.2516022
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3266602, 18.3224106
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8324203, 12.8335876
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1719208, 13.1678085
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3589249, 13.3588257
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9469299, 13.9457397
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7301407, 13.7292786
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8960876, 19.8904877
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9326706, 15.9358521
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4420471, 14.4407654
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6642456, 11.6605377
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4095688, 11.4095078
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9735641, 10.9778976
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8320389, 17.8305435
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2077866, 12.2021942
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0870209, 13.0769691
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.7056808, 13.6949005
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7043304, 17.6969070
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2620239, 19.2512054
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0872269, 17.0838547
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0585785, 15.0618439
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2722321, 13.2736740
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3284149, 9.3358612

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1741

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0395036, upper bound: 6.0950561
time: 25.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0595071, upper bound: 6.0752989
time: 23.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7065086, 14.7079697
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5312614, 9.5316734
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0303192, 10.0321121
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7171555, 9.7232819
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8380127, 10.8399773
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4217834, 10.4256516
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0758667, 12.0786781
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0805397, 10.0844421
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5391846, 12.5413895
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9760056, 13.9789658
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5075760, 16.5080566
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3410263, 10.3424187
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8898468, 12.8845329
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8318329, 19.8304596
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8089752, 20.8001175
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4500542, 11.4479408
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1254120, 14.1277008
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2623901, 16.2517700
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3265686, 18.3224945
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8324203, 12.8335876
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1706467, 13.1690826
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3588943, 13.3588486
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9468613, 13.9458160
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7297897, 13.7296295
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8955383, 19.8910217
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9338150, 15.9347000
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4419937, 14.4408188
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6641998, 11.6605835
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4095840, 11.4094963
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9731445, 10.9783211
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8291855, 17.8333893
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2044296, 12.2055550
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0830688, 13.0809135
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.7012024, 13.6993790
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7031403, 17.6981049
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2574463, 19.2557831
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0844116, 17.0866699
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0598831, 15.0605392
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2721329, 13.2737656
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3292503, 9.3350258

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1741

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0470740, upper bound: 6.0875213
time: 14.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0670695, upper bound: 6.0677503
time: 17.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7056847, 14.7088013
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5314598, 9.5314789
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0321655, 10.0302658
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7201004, 9.7203407
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8405304, 10.8374557
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4246063, 10.4228287
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0761948, 12.0783501
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0844917, 10.0804901
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5424347, 12.5381432
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9762573, 13.9787064
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5074387, 16.5081940
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3432159, 10.3402290
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8853226, 12.8890572
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8303680, 19.8319321
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8090057, 20.8000870
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4480858, 11.4499016
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1277618, 14.1253510
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2604599, 16.2537003
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3266449, 18.3224258
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8338242, 12.8321838
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1690903, 13.1706390
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3588486, 13.3588943
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9471283, 13.9455490
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7296371, 13.7297821
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8959045, 19.8906555
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9358597, 15.9326630
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4411774, 14.4416351
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6616974, 11.6630859
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4104080, 11.4086723
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9735260, 10.9779358
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8283005, 17.8342743
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2020187, 12.2079582
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0808868, 13.0830956
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6993866, 13.7011948
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6991501, 17.7020950
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2551575, 19.2580872
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0860214, 17.0850525
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0598602, 15.0605621
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2707672, 13.2751389
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3285980, 9.3356781

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1741

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0602323, upper bound: 6.0745783
time: 16.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0800236, upper bound: 6.0546073
time: 16.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7060890, 14.7083893
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5324440, 9.5304947
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0327301, 10.0297012
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7200928, 9.7203484
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8414307, 10.8365631
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4252472, 10.4221878
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0766754, 12.0778694
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0862923, 10.0786896
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5457382, 12.5348358
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9768524, 13.9781113
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5097427, 16.5058899
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3439484, 10.3395004
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8844376, 12.8899422
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8262787, 19.8360291
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8103180, 20.7987900
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4479332, 11.4500542
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1302872, 14.1228256
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2602921, 16.2538681
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3265686, 18.3225021
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8338242, 12.8321838
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1678162, 13.1719208
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3588181, 13.3589249
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9470520, 13.9456253
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7292862, 13.7301331
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8953705, 19.8911896
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9370041, 15.9315186
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4411163, 14.4416885
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6616592, 11.6631241
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4104233, 11.4086609
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9731064, 10.9783592
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8254623, 17.8371201
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1986618, 12.2113190
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0769501, 13.0870399
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6949081, 13.7056732
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6979599, 17.7032852
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2505798, 19.2626648
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0832138, 17.0878677
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0611572, 15.0592651
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2706680, 13.2752304
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3294296, 9.3348427

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1741

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0677991, upper bound: 6.0670324
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0875805, upper bound: 6.0470493
time: 14.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7083931, 14.7060852
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5304985, 9.5324402
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0297012, 10.0327301
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7203445, 9.7200928
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8365631, 10.8414268
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4221878, 10.4252472
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0778732, 12.0766754
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0786858, 10.0862923
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5348358, 12.5457382
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9781113, 13.9768524
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5058899, 16.5097427
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3395004, 10.3439484
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8899460, 12.8844376
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8360291, 19.8262711
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7987823, 20.8103104
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4500542, 11.4479332
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1228256, 14.1302872
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2538681, 16.2602921
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3225098, 18.3265686
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8321838, 12.8338242
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1719208, 13.1678162
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3589249, 13.3588181
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9456253, 13.9470520
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7301331, 13.7292862
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8911896, 19.8953705
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9315186, 15.9370041
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4416885, 14.4411163
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6631241, 11.6616592
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4086533, 11.4104233
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9783630, 10.9731064
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8371201, 17.8254547
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2113190, 12.1986580
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0870361, 13.0769463
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.7056732, 13.6949081
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7032852, 17.6979599
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2626648, 19.2505722
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0878677, 17.0832138
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0592651, 15.0611572
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2752304, 13.2706680
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3348427, 9.3294296

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1741

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0470493, upper bound: 6.0875805
time: 14.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0670324, upper bound: 6.0677991
time: 11.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7087975, 14.7056808
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5314751, 9.5314598
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0302658, 10.0321655
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7203369, 9.7201004
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8374634, 10.8405304
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4228287, 10.4246063
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0783463, 12.0761948
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0804863, 10.0844917
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5381393, 12.5424309
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9787064, 13.9762573
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5081940, 16.5074387
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3402328, 10.3432159
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8890610, 12.8853226
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8319244, 19.8303680
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8000946, 20.8090210
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4499016, 11.4480896
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1253510, 14.1277618
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2537003, 16.2604599
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3224182, 18.3266449
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8321838, 12.8338242
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1706390, 13.1690903
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3588943, 13.3588486
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9455490, 13.9471283
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7297821, 13.7296371
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8906555, 19.8959045
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9326630, 15.9358597
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4416351, 14.4411774
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6630859, 11.6616974
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4086685, 11.4104080
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9779358, 10.9735298
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8342819, 17.8283081
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2079620, 12.2020149
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0830994, 13.0808907
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.7011948, 13.6993866
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7020874, 17.6991501
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2580872, 19.2551498
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0850525, 17.0860291
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0605621, 15.0598602
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2751389, 13.2707672
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3356781, 9.3285980

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1741

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0546073, upper bound: 6.0800236
time: 15.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0745783, upper bound: 6.0602323
time: 17.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7079659, 14.7065125
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5316734, 9.5312653
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0321121, 10.0303192
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7232819, 9.7171593
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8399811, 10.8380127
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4256516, 10.4217834
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0786819, 12.0758667
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0844383, 10.0805397
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5413895, 12.5391846
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9789658, 13.9760056
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5080566, 16.5075760
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3424225, 10.3410263
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8845291, 12.8898468
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8304596, 19.8318329
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8001251, 20.8089905
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4479485, 11.4500465
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1277084, 14.1254120
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2517700, 16.2623901
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3224792, 18.3265762
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8335876, 12.8324203
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1690826, 13.1706467
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3588486, 13.3588943
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9458160, 13.9468613
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7296295, 13.7297897
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8910217, 19.8955383
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9347000, 15.9338150
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4408188, 14.4419937
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6605835, 11.6641998
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4094925, 11.4095879
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9783173, 10.9731407
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8333969, 17.8291855
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2055588, 12.2044220
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0809174, 13.0830727
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6993790, 13.7012024
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6981049, 17.7031403
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2557831, 19.2574539
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0866623, 17.0844116
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0605392, 15.0598831
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2737656, 13.2721329
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3350258, 9.3292503

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1741

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0677503, upper bound: 6.0670695
time: 16.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0875213, upper bound: 6.0470739
time: 20.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7083778, 14.7061005
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5326500, 9.5302849
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0326767, 10.0297546
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7232742, 9.7171669
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8408661, 10.8371162
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4262924, 10.4211426
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0791550, 12.0753860
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0862389, 10.0787392
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5447006, 12.5358772
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9795609, 13.9754028
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5103683, 16.5052643
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3431473, 10.3402939
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8836441, 12.8907318
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8263702, 19.8359299
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8014069, 20.8076935
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4477959, 11.4502029
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1302261, 14.1228867
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2516022, 16.2625580
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3224182, 18.3266525
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8335876, 12.8324203
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1678085, 13.1719208
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3588181, 13.3589249
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9457397, 13.9469376
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7292786, 13.7301407
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8904877, 19.8960724
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9358521, 15.9326706
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4407654, 14.4420471
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6605377, 11.6642456
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4095078, 11.4095764
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9778976, 10.9735641
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8305435, 17.8320389
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2021942, 12.2077827
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0769653, 13.0870171
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6949005, 13.7056808
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6969070, 17.7043304
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2512054, 19.2620316
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0838547, 17.0872269
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0618439, 15.0585785
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2736664, 13.2722321
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3358612, 9.3284149

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1741

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0752989, upper bound: 6.0595071
time: 19.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0950561, upper bound: 6.0395036
time: 6.95 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 29.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0395036, upper bound: 6.0950561
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0595071, upper bound: 6.0752989
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0470740, upper bound: 6.0875213
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0670695, upper bound: 6.0677503
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0602323, upper bound: 6.0745783
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0800236, upper bound: 6.0546073
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0677991, upper bound: 6.0670324
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0875805, upper bound: 6.0470493
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0470493, upper bound: 6.0875805
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0670324, upper bound: 6.0677991
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0546073, upper bound: 6.0800236
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0745783, upper bound: 6.0602323
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0677503, upper bound: 6.0670695
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0875213, upper bound: 6.0470739
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0752989, upper bound: 6.0595071
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 4, lower bound: -6.0950561, upper bound: 6.0395036

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7139893, 14.7149506
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5231209, 9.5261650
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0153198, 10.0203514
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6942902, 9.7040329
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8166351, 10.8238831
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4044876, 10.4122200
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0754471, 12.0810204
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0509109, 10.0624237
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5068665, 12.5199928
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9811935, 13.9844055
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5045547, 16.5096817
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3452911, 10.3490334
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8957291, 12.8879089
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8427429, 19.8328552
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8076324, 20.8013687
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4498062, 11.4463272
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1206894, 14.1281128
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2608109, 16.2499008
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3206787, 18.3155518
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8326645, 12.8338394
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1588745, 13.1524734
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3481598, 13.3460617
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9422836, 13.9403152
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7175827, 13.7145844
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8868713, 19.8796692
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9369583, 15.9404221
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4279480, 14.4242630
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6496887, 11.6431274
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4053879, 11.4050255
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9675598, 10.9739799
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8309326, 17.8292770
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2035751, 12.1979179
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0735168, 13.0623856
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6946030, 13.6831779
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7070541, 17.6964569
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2547607, 19.2440643
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0835800, 17.0809021
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0491562, 15.0541229
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2721939, 13.2736588
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3303757, 9.3384476

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0363645, upper bound: 6.0948736
time: 18.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0388430, upper bound: 6.0899208
time: 15.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7126694, 14.7162704
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5237999, 9.5254860
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0174332, 10.0182381
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6979294, 9.7003937
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8201294, 10.8203888
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4070740, 10.4096336
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0772476, 12.0792274
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0549240, 10.0584145
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5111771, 12.5156898
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9802551, 13.9853516
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5045776, 16.5096588
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3461761, 10.3481483
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8949966, 12.8886490
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8424072, 19.8332367
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8076477, 20.8013535
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4487381, 11.4474411
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1207809, 14.1280365
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2608566, 16.2498856
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3197937, 18.3164444
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8326721, 12.8338394
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1565857, 13.1547623
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3461685, 13.3480530
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9415054, 13.9410858
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7154465, 13.7167206
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8852692, 19.8812714
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9372406, 15.9401398
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4255447, 14.4266663
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6468353, 11.6459808
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4050903, 11.4053230
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9696503, 10.9718895
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8307800, 17.8294296
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2035065, 12.1980133
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0724335, 13.0634995
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6939621, 13.6838875
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7038803, 17.6996307
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2548981, 19.2439728
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0843124, 17.0802078
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0508575, 15.0524216
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2722168, 13.2736282
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3310013, 9.3378181

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0563714, upper bound: 6.0751162
time: 16.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0588467, upper bound: 6.0701528
time: 16.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7144012, 14.7145386
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5240974, 9.5251846
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0158768, 10.0197868
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6942825, 9.7040443
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8175278, 10.8229904
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4051285, 10.4115791
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0759354, 12.0805397
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0527115, 10.0606270
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5101776, 12.5166855
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9817886, 13.9838104
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5068588, 16.5073700
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3460236, 10.3483047
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8948441, 12.8887939
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8386536, 19.8369446
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8089294, 20.8000717
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4496536, 11.4464798
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1232147, 14.1255875
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2606354, 16.2500763
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3206177, 18.3156281
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8326645, 12.8338394
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1576004, 13.1537476
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3481293, 13.3460922
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9422073, 13.9403839
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7172318, 13.7149353
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8863373, 19.8802032
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9381104, 15.9392700
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4278946, 14.4243164
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6496429, 11.6431732
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4054031, 11.4050140
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9671326, 10.9744034
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8280792, 17.8321228
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2002182, 12.2012787
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0695724, 13.0663300
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6901169, 13.6876564
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7058563, 17.6976471
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2501831, 19.2486420
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0807648, 17.0837173
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0504608, 15.0528183
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2720947, 13.2737503
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3312111, 9.3376122

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0419083, upper bound: 6.0868635
time: 14.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0468904, upper bound: 6.0843980
time: 15.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7130814, 14.7158585
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5247765, 9.5245056
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0179901, 10.0176735
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6979218, 9.7004013
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8210297, 10.8194923
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4077148, 10.4089966
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0777206, 12.0787468
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0567245, 10.0566177
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5144806, 12.5123825
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9808502, 13.9847488
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5068817, 16.5073471
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3469086, 10.3474159
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8941116, 12.8895340
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8383179, 19.8373337
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8089447, 20.8000565
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4485855, 11.4475975
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1233063, 14.1255112
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2606812, 16.2500610
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3197174, 18.3165207
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8326721, 12.8338394
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1553116, 13.1560364
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3461380, 13.3480835
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9414291, 13.9411621
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7150955, 13.7170715
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8847351, 19.8818054
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9383926, 15.9389954
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4254913, 14.4267197
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6467972, 11.6460266
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4051056, 11.4053078
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9692230, 10.9723129
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8279266, 17.8322754
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2001419, 12.2013741
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0684891, 13.0674438
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6894760, 13.6883659
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7026825, 17.7008209
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2503204, 19.2485504
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0814972, 17.0830154
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0521622, 15.0511169
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2721176, 13.2737274
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3318367, 9.3369865

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0619114, upper bound: 6.0670912
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0668862, upper bound: 6.0646200
time: 13.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7135696, 14.7153702
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5242958, 9.5249901
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0177307, 10.0179405
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6972198, 9.7011032
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8200455, 10.8204727
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4079514, 10.4087563
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0762634, 12.0802078
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0566635, 10.0566711
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5134201, 12.5134392
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9820480, 13.9835587
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5067291, 16.5075073
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3482132, 10.3461151
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8903198, 12.8933182
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8372345, 19.8384171
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8089447, 20.8000412
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4477463, 11.4484406
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1255722, 14.1232452
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2587433, 16.2519989
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3206787, 18.3155670
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8340759, 12.8324356
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1560440, 13.1553040
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3480835, 13.3461380
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9424744, 13.9401169
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7170792, 13.7150879
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8866882, 19.8798523
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9401474, 15.9372330
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4270782, 14.4251328
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6471405, 11.6456757
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4062195, 11.4041901
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9675217, 10.9740181
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8271942, 17.8330078
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1978378, 12.2036819
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0674210, 13.0685120
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6883774, 13.6894722
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7018738, 17.7016373
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2479248, 19.2509460
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0823822, 17.0821381
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0504379, 15.0528412
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2707214, 13.2751236
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3305588, 9.3382645

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0570971, upper bound: 6.0743955
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0595724, upper bound: 6.0694329
time: 16.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7122498, 14.7166901
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5249748, 9.5243111
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0198364, 10.0158272
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7008591, 9.6974640
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8235474, 10.8169746
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4105377, 10.4061699
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0780563, 12.0784187
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0606766, 10.0526619
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5177307, 12.5091324
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9811020, 13.9844971
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5067520, 16.5074844
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3490982, 10.3452301
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8895798, 12.8940582
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8368530, 19.8387451
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8089600, 20.8000259
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4466248, 11.4495087
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1256485, 14.1231613
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2587585, 16.2519455
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3197784, 18.3164597
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8340759, 12.8324356
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1537552, 13.1575928
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3460922, 13.3481293
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9416962, 13.9408951
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7149429, 13.7172241
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8851013, 19.8814545
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9404297, 15.9369583
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4246750, 14.4275360
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6442947, 11.6485291
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4059296, 11.4044876
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9696121, 10.9719276
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8270416, 17.8331604
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1977386, 12.2037544
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0663071, 13.0695953
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6876602, 13.6901131
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6987000, 17.7048111
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2480164, 19.2508087
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0830765, 17.0814056
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0521393, 15.0511398
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2707520, 13.2751007
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3311844, 9.3376389

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0768921, upper bound: 6.0544242
time: 15.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0793639, upper bound: 6.0494531
time: 12.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7139816, 14.7149582
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5252724, 9.5240097
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0182877, 10.0173759
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6972122, 9.7011108
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8209457, 10.8195763
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4085922, 10.4081192
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0767441, 12.0797272
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0584641, 10.0548744
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5167313, 12.5101318
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9826431, 13.9829559
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5090332, 16.5052032
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3489456, 10.3453827
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8894348, 12.8942032
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8331451, 19.8425140
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8102417, 20.7987442
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4475861, 11.4485931
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1280975, 14.1207199
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2585754, 16.2521667
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3205872, 18.3156433
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8340759, 12.8324356
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1547699, 13.1565857
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3480530, 13.3461685
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9423981, 13.9401932
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7167282, 13.7154388
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8861694, 19.8803864
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9412994, 15.9360886
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4270248, 14.4251938
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6470947, 11.6457214
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4062347, 11.4041786
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9671021, 10.9744377
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8243408, 17.8358536
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1944809, 12.2070427
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0634766, 13.0724564
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6838913, 13.6939507
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7006760, 17.7028275
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2433472, 19.2555237
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0795670, 17.0849533
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0517349, 15.0515442
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2706299, 13.2752151
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3313904, 9.3374290

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0626374, upper bound: 6.0663735
time: 20.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0676159, upper bound: 6.0639027
time: 6.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7126617, 14.7162781
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5259514, 9.5233307
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0204010, 10.0152626
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7008514, 9.6974716
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8244400, 10.8160782
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4111786, 10.4055328
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0785294, 12.0779381
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0624771, 10.0508652
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5210342, 12.5058289
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9817047, 13.9839020
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5090561, 16.5051804
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3498306, 10.3444977
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8886948, 12.8949432
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8327637, 19.8428421
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8102570, 20.7987289
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4464722, 11.4496613
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1281738, 14.1206360
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2585907, 16.2521210
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3197021, 18.3165359
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8340759, 12.8324356
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1524811, 13.1588669
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3460617, 13.3481598
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9416199, 13.9409714
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7145920, 13.7175751
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8845520, 19.8819885
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9415741, 15.9358063
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4246216, 14.4275970
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6442490, 11.6485672
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4059372, 11.4044762
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9691849, 10.9723511
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8241882, 17.8360062
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1943817, 12.2071152
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0623627, 13.0735397
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6831894, 13.6945915
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6975021, 17.7060089
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2434387, 19.2553864
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0802612, 17.0842209
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0534363, 15.0498352
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2706528, 13.2751923
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3320198, 9.3368034

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0824278, upper bound: 6.0463897
time: 9.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0873976, upper bound: 6.0439153
time: 14.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7162781, 14.7126617
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5233269, 9.5259514
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0152664, 10.0204048
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6974716, 9.7008553
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8160782, 10.8244400
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4055328, 10.4111748
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0779343, 12.0785332
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0508652, 10.0624771
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5058289, 12.5210342
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9839020, 13.9817047
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5051804, 16.5090561
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3444977, 10.3498306
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8949432, 12.8886986
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8428345, 19.8327560
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7987213, 20.8102646
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4496613, 11.4464722
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1206360, 14.1281738
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2521210, 16.2585907
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3165283, 18.3197021
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8324356, 12.8340759
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1588669, 13.1524811
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3481598, 13.3460617
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9409714, 13.9416199
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7175751, 13.7145920
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8819885, 19.8845673
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9358063, 15.9415741
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4275970, 14.4246140
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6485672, 11.6442490
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4044724, 11.4059410
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9723511, 10.9691849
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8360138, 17.8241882
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2071152, 12.1943817
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0735397, 13.0623627
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6945877, 13.6831856
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7060089, 17.6975021
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2553864, 19.2434311
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0842209, 17.0802689
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0498428, 15.0534363
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2751923, 13.2706528
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3368034, 9.3320198

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0439153, upper bound: 6.0873976
time: 16.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0463897, upper bound: 6.0824278
time: 7.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7149582, 14.7139816
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5240135, 9.5252724
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0173721, 10.0182915
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7011108, 9.6972122
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8195801, 10.8209419
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4081192, 10.4085922
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0797348, 12.0767441
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0548706, 10.0584679
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5101318, 12.5167313
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9829559, 13.9826431
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5052032, 16.5090332
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3453827, 10.3489456
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8942032, 12.8894348
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8425140, 19.8331451
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7987366, 20.8102493
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4485931, 11.4475899
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1207275, 14.1280975
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2521667, 16.2585754
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3156433, 18.3205948
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8324356, 12.8340759
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1565857, 13.1547623
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3461685, 13.3480530
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9401932, 13.9423981
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7154388, 13.7167282
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8803711, 19.8861694
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9360886, 15.9412994
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4251862, 14.4270248
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6457214, 11.6470947
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4041748, 11.4062347
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9744415, 10.9670982
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8358612, 17.8243484
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2070389, 12.1944771
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0724564, 13.0634766
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6939468, 13.6838951
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7028275, 17.7006760
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2555237, 19.2433395
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0849457, 17.0795670
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0515442, 15.0517349
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2752151, 13.2706299
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3374329, 9.3313904

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0639027, upper bound: 6.0676159
time: 6.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0663735, upper bound: 6.0626374
time: 41.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7166901, 14.7122498
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5243111, 9.5249710
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0158310, 10.0198402
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6974640, 9.7008629
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8169708, 10.8235474
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4061737, 10.4105377
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0784225, 12.0780563
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0526581, 10.0606766
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5091324, 12.5177307
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9844971, 13.9811020
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5074844, 16.5067520
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3452301, 10.3490982
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8940582, 12.8895836
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8387451, 19.8368530
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8000183, 20.8089676
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4495087, 11.4466248
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1231613, 14.1256485
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2519455, 16.2587585
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3164673, 18.3197784
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8324356, 12.8340759
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1575928, 13.1537552
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3481293, 13.3460922
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9408951, 13.9416962
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7172241, 13.7149429
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8814545, 19.8850861
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9369583, 15.9404297
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4275360, 14.4246750
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6485214, 11.6442947
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4044876, 11.4059296
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9719238, 10.9696083
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8331604, 17.8270416
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2037582, 12.1977386
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0695953, 13.0662994
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6901169, 13.6876640
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7048111, 17.6987000
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2508087, 19.2480164
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0814056, 17.0830765
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0511398, 15.0521393
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2751007, 13.2707520
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3376389, 9.3311844

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0494531, upper bound: 6.0793639
time: 6.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0544242, upper bound: 6.0768921
time: 33.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7153702, 14.7135696
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5249901, 9.5242920
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0179367, 10.0177269
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7011032, 9.6972237
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8204727, 10.8200493
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4087601, 10.4079514
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0802078, 12.0762634
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0566711, 10.0566673
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5134430, 12.5134239
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9835587, 13.9820480
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5075073, 16.5067291
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3461151, 10.3482132
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8933182, 12.8903198
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8384094, 19.8372345
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8000336, 20.8089523
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4484406, 11.4477425
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1232452, 14.1255722
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2519989, 16.2587433
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3155518, 18.3206711
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8324356, 12.8340759
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1553040, 13.1560440
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3461380, 13.3480835
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9401169, 13.9424744
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7150879, 13.7170792
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8798523, 19.8867035
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9372330, 15.9401474
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4251328, 14.4270782
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6456757, 11.6471405
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4041901, 11.4062233
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9740143, 10.9675217
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8330078, 17.8271942
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2036819, 12.1978378
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0685120, 13.0674133
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6894760, 13.6883736
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7016373, 17.7018738
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2509460, 19.2479172
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0821381, 17.0823746
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0528412, 15.0504379
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2751236, 13.2707214
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3382645, 9.3305550

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0694329, upper bound: 6.0595724
time: 16.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0743955, upper bound: 6.0570971
time: 17.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7158585, 14.7130814
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5245094, 9.5247765
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0176773, 10.0179939
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7004013, 9.6979218
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8194962, 10.8210297
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4089966, 10.4077148
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0787430, 12.0777245
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0566177, 10.0567245
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5123825, 12.5144806
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9847488, 13.9808502
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5073471, 16.5068817
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3474197, 10.3469124
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8895340, 12.8941078
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8373413, 19.8383179
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8000488, 20.8089371
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4476013, 11.4485855
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1255112, 14.1233063
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2500610, 16.2606812
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3165283, 18.3197174
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8338394, 12.8326721
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1560364, 13.1553116
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3480835, 13.3461380
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9411621, 13.9414291
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7170715, 13.7150955
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8818054, 19.8847351
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9389954, 15.9383850
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4267273, 14.4254913
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6460190, 11.6467972
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4053116, 11.4051056
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9723129, 10.9692230
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8322754, 17.8279190
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2013779, 12.2001457
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0674362, 13.0684891
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6883621, 13.6894798
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7008209, 17.7026825
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2485504, 19.2503204
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0830154, 17.0814972
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0511169, 15.0521622
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2737274, 13.2721176
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3369865, 9.3318367

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0646200, upper bound: 6.0668862
time: 17.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0670912, upper bound: 6.0619114
time: 15.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7145386, 14.7144012
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5251884, 9.5240974
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0197830, 10.0158806
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7040405, 9.6942825
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8229904, 10.8175278
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4115829, 10.4051285
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0805435, 12.0759315
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0606232, 10.0527115
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5166855, 12.5101776
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9838104, 13.9817886
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5073700, 16.5068588
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3483047, 10.3460274
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8887939, 12.8948441
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8369446, 19.8386536
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8000641, 20.8089218
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4464798, 11.4496536
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1255875, 14.1232147
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2500687, 16.2606354
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3156281, 18.3206100
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8338394, 12.8326645
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1537476, 13.1576004
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3460922, 13.3481293
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9403839, 13.9422073
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7149353, 13.7172318
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8802032, 19.8863373
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9392776, 15.9381104
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4243164, 14.4278946
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6431732, 11.6496429
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4050140, 11.4054031
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9744034, 10.9671326
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8321228, 17.8280716
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.2012787, 12.2002182
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0663300, 13.0695724
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6876602, 13.6901207
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6976471, 17.7058563
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2486420, 19.2501755
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0837173, 17.0807648
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0528183, 15.0504608
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2737503, 13.2720947
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3376122, 9.3312111

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0843980, upper bound: 6.0468904
time: 19.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0868635, upper bound: 6.0419083
time: 16.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7162704, 14.7126694
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5254860, 9.5237961
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0182419, 10.0174294
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7003937, 9.6979294
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8203888, 10.8201332
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4096375, 10.4070740
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0792313, 12.0772438
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0584106, 10.0549240
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5156860, 12.5111732
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9853516, 13.9802551
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5096588, 16.5045776
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3481522, 10.3461800
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8886490, 12.8949928
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8332367, 19.8424149
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8013458, 20.8076401
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4474411, 11.4487381
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1280365, 14.1207809
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2498856, 16.2608566
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3164368, 18.3197937
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8338394, 12.8326721
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1547623, 13.1565857
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3480530, 13.3461685
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9410858, 13.9415054
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7167206, 13.7154465
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8812866, 19.8852692
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9401398, 15.9372406
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4266663, 14.4255447
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6459808, 11.6468353
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4053192, 11.4050941
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9718933, 10.9696465
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8294373, 17.8307724
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1980133, 12.2035065
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0634995, 13.0724335
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6838913, 13.6939583
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6996307, 17.7038803
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2439728, 19.2548981
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0802002, 17.0843124
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0524216, 15.0508575
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2736282, 13.2722168
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3378220, 9.3310013

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0701528, upper bound: 6.0588467
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0751162, upper bound: 6.0563713
time: 12.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7149506, 14.7139893
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5261650, 9.5231171
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0203476, 10.0153160
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7040329, 9.6942902
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8238831, 10.8166351
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4122162, 10.4044876
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0810165, 12.0754547
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0624237, 10.0509148
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5199966, 12.5068703
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9844055, 13.9811935
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.5096817, 16.5045547
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3490372, 10.3452950
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8879089, 12.8957291
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8328552, 19.8427429
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8013611, 20.8076248
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4463272, 11.4498062
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1281128, 14.1206894
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2499008, 16.2608109
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3155518, 18.3206863
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8338394, 12.8326645
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1524734, 13.1588745
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3460617, 13.3481598
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9403152, 13.9422836
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7145844, 13.7175827
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8796692, 19.8868713
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9404221, 15.9369583
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4242630, 14.4279556
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6431274, 11.6496887
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4050293, 11.4053917
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9739838, 10.9675560
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8292847, 17.8309250
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1979141, 12.2035751
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0623856, 13.0735168
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6831894, 13.6945992
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6964569, 17.7070541
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2440643, 19.2547607
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0809021, 17.0835800
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0541229, 15.0491562
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2736588, 13.2721939
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3384476, 9.3303757

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1778

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0899208, upper bound: 6.0388430
time: 16.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0948736, upper bound: 6.0363645
time: 15.81 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 34.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0363645, upper bound: 6.0948736
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0388430, upper bound: 6.0899208
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0563714, upper bound: 6.0751162
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0588467, upper bound: 6.0701528
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0419083, upper bound: 6.0868635
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0468904, upper bound: 6.0843980
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0619114, upper bound: 6.0670912
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0668862, upper bound: 6.0646200
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0570971, upper bound: 6.0743955
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0595724, upper bound: 6.0694329
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0768921, upper bound: 6.0544242
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0793639, upper bound: 6.0494531
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0626374, upper bound: 6.0663735
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0676159, upper bound: 6.0639027
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0824278, upper bound: 6.0463897
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0873976, upper bound: 6.0439153
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0439153, upper bound: 6.0873976
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0463897, upper bound: 6.0824278
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0639027, upper bound: 6.0676159
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0663735, upper bound: 6.0626374
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0494531, upper bound: 6.0793639
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0544242, upper bound: 6.0768921
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0694329, upper bound: 6.0595724
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0743955, upper bound: 6.0570971
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0646200, upper bound: 6.0668862
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0670912, upper bound: 6.0619114
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0843980, upper bound: 6.0468904
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0868635, upper bound: 6.0419083
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0701528, upper bound: 6.0588467
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0751162, upper bound: 6.0563713
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0899208, upper bound: 6.0388430
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 34.77
Output dim: 4, lower bound: -6.0948736, upper bound: 6.0363645

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7105637, 14.7120285
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5160637, 9.5202484
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0112877, 10.0169754
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6946526, 9.7041893
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8112030, 10.8195229
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4002151, 10.4086266
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0779037, 12.0841370
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0378952, 10.0515099
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4834518, 12.5003853
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9762192, 13.9802399
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4884491, 16.4961777
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3503799, 10.3556252
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.9006996, 12.8917465
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8589020, 19.8441620
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7971649, 20.7925949
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4522095, 11.4481621
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1012115, 14.1117096
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2631912, 16.2520065
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3200378, 18.3148727
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8338699, 12.8353500
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1510773, 13.1431732
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3490372, 13.3470230
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9420395, 13.9400635
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7156601, 13.7123413
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8829041, 19.8749771
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9419327, 15.9468460
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4276886, 14.4239807
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6491394, 11.6425629
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4075089, 11.4076691
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9628754, 10.9683990
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8137131, 17.8087540
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1832886, 12.1737137
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0496063, 13.0338631
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6675110, 13.6508789
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.7001877, 17.6883163
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2269516, 19.2108994
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0663452, 17.0603561
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0535202, 15.0600357
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2709274, 13.2721825
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3337059, 9.3427505

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0361207, upper bound: 6.0895338
time: 16.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0309814, upper bound: 6.0946299
time: 39.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7110596, 14.7115097
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5172005, 9.5191116
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0119438, 10.0163231
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6944466, 9.7043991
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8122635, 10.8184509
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4008942, 10.4079437
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0785522, 12.0834656
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0400009, 10.0494041
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4872665, 12.4965706
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9770279, 13.9794388
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4910507, 16.4935684
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3518829, 10.3541183
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8995705, 12.8928757
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8540497, 19.8490219
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7988586, 20.7909012
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4516449, 11.4487343
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1042938, 14.1086273
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2629089, 16.2522507
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3200073, 18.3149033
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8341675, 12.8350372
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1495743, 13.1446762
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3491211, 13.3469467
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9420319, 13.9400711
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7153473, 13.7126617
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8821716, 19.8756485
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9433899, 15.9453964
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4276657, 14.4239960
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6491241, 11.6425705
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4080124, 11.4071503
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9619751, 10.9692993
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8104019, 17.8120651
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1793747, 12.1776237
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0449982, 13.0384712
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6622925, 13.6560898
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6989136, 17.6895905
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2215958, 19.2162552
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0630264, 17.0636673
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0550766, 15.0584869
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2707138, 13.2723808
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3346786, 9.3417816

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0385991, upper bound: 6.0845744
time: 13.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0334619, upper bound: 6.0896771
time: 14.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7092361, 14.7133560
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5167427, 9.5195694
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0134010, 10.0148621
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6982994, 9.7005501
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8146973, 10.8160248
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4027939, 10.4060402
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0796890, 12.0823441
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0419006, 10.0475006
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4877548, 12.4960823
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9752808, 13.9811783
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4884720, 16.4961548
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3512650, 10.3547401
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8999596, 12.8924866
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8585815, 19.8445511
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7971802, 20.7925797
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4511414, 11.4492760
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1012955, 14.1116333
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2632446, 16.2519913
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3191528, 18.3157654
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8338699, 12.8353424
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1487885, 13.1454620
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3470535, 13.3490143
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9412613, 13.9408417
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7135239, 13.7144775
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8812866, 19.8765793
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9422150, 15.9465714
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4252777, 14.4263840
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6462936, 11.6454163
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4072189, 11.4079666
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9649658, 10.9663086
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8135605, 17.8089066
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1832123, 12.1738129
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0485229, 13.0349770
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6668701, 13.6515808
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6970062, 17.6914978
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2270889, 19.2108002
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0670776, 17.0596542
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0552216, 15.0583344
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2709427, 13.2721519
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3343353, 9.3421211

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0561274, upper bound: 6.0697689
time: 17.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0509956, upper bound: 6.0748726
time: 15.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7097397, 14.7128372
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5178795, 9.5184326
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0140572, 10.0142097
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6980858, 9.7007599
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8157578, 10.8149529
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4034805, 10.4053574
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0803452, 12.0816727
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0440063, 10.0453949
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4915695, 12.4922638
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9760818, 13.9803772
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4910736, 16.4935455
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3527679, 10.3532333
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8988304, 12.8936157
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8537292, 19.8494034
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7988739, 20.7908859
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4505768, 11.4498482
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1043777, 14.1085510
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2629623, 16.2522354
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3191071, 18.3157959
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8341751, 12.8350372
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1472855, 13.1469650
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3471298, 13.3489380
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9412537, 13.9408493
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7132111, 13.7147980
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8805695, 19.8772507
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9436646, 15.9451141
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4252625, 14.4263992
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6462708, 11.6454239
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4077225, 11.4074440
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9640656, 10.9672089
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8102493, 17.8122177
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1793060, 12.1777229
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0439148, 13.0395851
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6616516, 13.6567993
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6957397, 17.6927643
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2217331, 19.2161560
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0637589, 17.0629730
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0567780, 15.0567856
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2707443, 13.2723503
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3353043, 9.3411522

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0586027, upper bound: 6.0647984
time: 18.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0534718, upper bound: 6.0699093
time: 15.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7109680, 14.7116089
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5170403, 9.5192680
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0118523, 10.0164108
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6946449, 9.7041969
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8120956, 10.8186150
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4008484, 10.4079857
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0783768, 12.0836411
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0396957, 10.0497131
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4867477, 12.4970779
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9768219, 13.9796448
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4907532, 16.4938660
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3511124, 10.3548927
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8998146, 12.8926315
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8548126, 19.8482590
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7984619, 20.7912979
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4520569, 11.4483147
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1037292, 14.1091843
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2629852, 16.2521744
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3199463, 18.3149490
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8338699, 12.8353424
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1498032, 13.1444473
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3490143, 13.3470535
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9419632, 13.9401398
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7153091, 13.7126923
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8823090, 19.8755112
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9430847, 15.9457016
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4276276, 14.4240341
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6490860, 11.6426086
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4075241, 11.4076385
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9624557, 10.9688187
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8108749, 17.8115997
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1799240, 12.1770744
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0456543, 13.0378075
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6630325, 13.6553574
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6989899, 17.6895142
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2223740, 19.2154770
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0635300, 17.0631714
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0548248, 15.0587387
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2708206, 13.2722816
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3345413, 9.3419151

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0416643, upper bound: 6.0815156
time: 15.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0365292, upper bound: 6.0866199
time: 12.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7114868, 14.7111053
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5181847, 9.5181313
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0125084, 10.0157585
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6944389, 9.7044106
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8131638, 10.8175583
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4015350, 10.4073029
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0790482, 12.0829849
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0417938, 10.0476074
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4905624, 12.4932632
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9776230, 13.9788361
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4933624, 16.4912643
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3526154, 10.3533859
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8986855, 12.8937607
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8499603, 19.8531113
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8001556, 20.7896042
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4514923, 11.4488869
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1068115, 14.1061020
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2627411, 16.2524567
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3199158, 18.3149796
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8341751, 12.8350372
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1483002, 13.1459503
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3490906, 13.3469772
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9419556, 13.9401474
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7149963, 13.7130127
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8816528, 19.8762283
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9445343, 15.9442444
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4276123, 14.4240494
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6490784, 11.6426239
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4080505, 11.4071350
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9615555, 10.9697227
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8075485, 17.8149109
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1760101, 12.1809845
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0410461, 13.0424156
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6578140, 13.6605682
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6977234, 17.6907806
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2170029, 19.2208328
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0602188, 17.0664825
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0563736, 15.0571823
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2706223, 13.2724800
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3355141, 9.3409462

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0466460, upper bound: 6.0790455
time: 17.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0415137, upper bound: 6.0841545
time: 18.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7096481, 14.7129288
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5177193, 9.5185890
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0139656, 10.0142975
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6982841, 9.7005577
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8155899, 10.8151169
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4034348, 10.4053993
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0801773, 12.0818481
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0437012, 10.0457001
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4910507, 12.4927750
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9758759, 13.9805832
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4907761, 16.4938431
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3519974, 10.3540077
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8990746, 12.8933716
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8544769, 19.8486404
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7984772, 20.7912827
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4509888, 11.4494324
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1038208, 14.1091080
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2630386, 16.2521591
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3190613, 18.3158417
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8338699, 12.8353348
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1475143, 13.1467361
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3470230, 13.3490448
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9411926, 13.9409180
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7131729, 13.7148285
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8807068, 19.8771133
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9433594, 15.9454193
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4252243, 14.4264374
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6462326, 11.6454544
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4072266, 11.4079361
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9645462, 10.9667320
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8107224, 17.8117523
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1798553, 12.1771736
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0445709, 13.0389214
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6623917, 13.6560593
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6958160, 17.6926880
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2224960, 19.2153854
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0642624, 17.0624695
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0565262, 15.0570374
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2708511, 13.2722511
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3351707, 9.3412895

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0616674, upper bound: 6.0617341
time: 15.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0565378, upper bound: 6.0668477
time: 6.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7101669, 14.7124252
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5188637, 9.5174522
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0146217, 10.0136452
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6980782, 9.7007675
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8166656, 10.8140602
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4041214, 10.4047165
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0808487, 12.0811920
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0458069, 10.0435944
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4948807, 12.4889565
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9766846, 13.9797821
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4933853, 16.4912415
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3535004, 10.3525009
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8979454, 12.8945007
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8496246, 19.8535004
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8001709, 20.7895889
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4504242, 11.4500046
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1069031, 14.1060257
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2627869, 16.2524414
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3190308, 18.3158722
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8341751, 12.8350372
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1460114, 13.1482391
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3470993, 13.3489685
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9411774, 13.9409256
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7128601, 13.7151489
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8800354, 19.8778305
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9448166, 15.9439697
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4252090, 14.4264603
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6462326, 11.6454773
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4077530, 11.4074326
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9636459, 10.9676323
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8073959, 17.8150711
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1759491, 12.1810799
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0399628, 13.0435295
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6571732, 13.6612778
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6945496, 17.6939545
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2171555, 19.2207413
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0609436, 17.0657806
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0580750, 15.0554810
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2706528, 13.2724495
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3361397, 9.3403168

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0666421, upper bound: 6.0592624
time: 12.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0615182, upper bound: 6.0643766
time: 18.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7101364, 14.7124557
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5172386, 9.5190735
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0136986, 10.0145645
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6975899, 9.7012558
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8146133, 10.8161087
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4036713, 10.4051628
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0787125, 12.0833282
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0436478, 10.0457573
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4899979, 12.4938316
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9770737, 13.9793854
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4906158, 16.4940033
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3532944, 10.3527031
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8952904, 12.8971596
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8534088, 19.8497238
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7984924, 20.7912674
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4501495, 11.4502754
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1060867, 14.1068497
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2611313, 16.2540970
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3200226, 18.3148804
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8352737, 12.8339386
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1482468, 13.1460037
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3489685, 13.3470993
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9422302, 13.9398727
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7151566, 13.7128525
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8827209, 19.8751450
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9451218, 15.9436569
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4268112, 14.4248505
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6465912, 11.6451111
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4083481, 11.4068375
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9628372, 10.9684334
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8099899, 17.8124847
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1775436, 12.1794815
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0435028, 13.0399895
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6612854, 13.6571655
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6950073, 17.6935043
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2201157, 19.2177811
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0651398, 17.0615845
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0548019, 15.0587616
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2694473, 13.2736473
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3338890, 9.3425674

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0568534, upper bound: 6.0690429
time: 18.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0517283, upper bound: 6.0741516
time: 15.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7106400, 14.7119370
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5183754, 9.5179329
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0143547, 10.0139122
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6973763, 9.7014694
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8156738, 10.8150368
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4043579, 10.4044800
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0793610, 12.0826569
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0457535, 10.0436516
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4938126, 12.4900169
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9778748, 13.9785843
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4932251, 16.4914017
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3548050, 10.3511963
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8941612, 12.8982849
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8485413, 19.8545837
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8001862, 20.7895737
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4495697, 11.4508438
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1091690, 14.1037598
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2608490, 16.2543488
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3199921, 18.3149109
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8355713, 12.8336334
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1467438, 13.1475067
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3490448, 13.3470230
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9422226, 13.9398804
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7148361, 13.7131729
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8820038, 19.8758163
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9465790, 15.9422073
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4267960, 14.4248657
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6465759, 11.6451187
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4088516, 11.4063148
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9619370, 10.9693336
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8066635, 17.8157959
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1736374, 12.1833916
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0388947, 13.0445976
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6560669, 13.6623840
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6937332, 17.6947708
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2147446, 19.2231369
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0618286, 17.0649033
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0563583, 15.0572052
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2692490, 13.2738457
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3348579, 9.3415985

Time for backsubstitution: 2.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0593288, upper bound: 6.0640731
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0542048, upper bound: 6.0691890
time: 16.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7088165, 14.7137680
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5179176, 9.5183945
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0158119, 10.0124512
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7012291, 9.6976166
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8181152, 10.8126106
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4062576, 10.4025764
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0804977, 12.0815353
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0476532, 10.0417480
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4943008, 12.4895287
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9761353, 13.9803314
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4906387, 16.4939804
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3541794, 10.3518181
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8945503, 12.8978958
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8530121, 19.8500595
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7985077, 20.7912521
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4490356, 11.4513435
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1061630, 14.1067581
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2611465, 16.2540512
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3191223, 18.3157730
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8352737, 12.8339386
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1459579, 13.1482925
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3469772, 13.3490906
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9414520, 13.9406433
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7130203, 13.7149887
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8811188, 19.8767624
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9454041, 15.9433823
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4244080, 14.4272537
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6437454, 11.6479568
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4080505, 11.4071350
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9649277, 10.9663429
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8098373, 17.8126373
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1774445, 12.1795540
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0423889, 13.0410728
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6605759, 13.6578064
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6918259, 17.6966782
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2202072, 19.2176437
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0658417, 17.0608521
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0565033, 15.0570602
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2694778, 13.2736244
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3345184, 9.3419418

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0766484, upper bound: 6.0490625
time: 18.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0715299, upper bound: 6.0541802
time: 6.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7093201, 14.7132568
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5190544, 9.5172539
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0164680, 10.0117989
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7010155, 9.6978264
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8191681, 10.8115387
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4069443, 10.4018936
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0811539, 12.0808640
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0497589, 10.0396423
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4981155, 12.4857101
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9769363, 13.9795227
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4932480, 16.4913788
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3556900, 10.3503113
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8934212, 12.8990250
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8481598, 19.8549118
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8002014, 20.7895584
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4484558, 11.4519157
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1092453, 14.1036758
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2608643, 16.2543030
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3190918, 18.3158035
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8355789, 12.8336334
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1444550, 13.1497955
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3470535, 13.3490143
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9414444, 13.9406586
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7126999, 13.7153091
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8804016, 19.8774185
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9468536, 15.9419250
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4243927, 14.4272690
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6437302, 11.6479645
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4085541, 11.4066124
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9640274, 10.9672470
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8065109, 17.8159485
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1735382, 12.1834602
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0377808, 13.0456810
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6553574, 13.6630249
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6905594, 17.6979446
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2148361, 19.2229996
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0625305, 17.0641708
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0580521, 15.0555038
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2692795, 13.2738228
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3354874, 9.3409729

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0791202, upper bound: 6.0440849
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0740028, upper bound: 6.0492092
time: 15.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7105484, 14.7120285
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5182152, 9.5180931
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0142632, 10.0139999
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6975822, 9.7012672
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8155060, 10.8152008
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4043121, 10.4045219
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0791855, 12.0828285
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0454483, 10.0439568
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4933090, 12.4905243
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9776688, 13.9787903
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4929276, 16.4916992
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3540268, 10.3519707
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8944054, 12.8980408
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8493042, 19.8538208
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7997894, 20.7899704
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4499969, 11.4504280
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1086121, 14.1043243
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2609253, 16.2542725
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3199463, 18.3149567
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8352737, 12.8339386
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1469650, 13.1472855
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3489380, 13.3471298
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9421539, 13.9399414
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7148056, 13.7132034
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8821411, 19.8756790
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9462738, 15.9425125
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4267578, 14.4249039
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6465378, 11.6451569
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4083557, 11.4068031
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9624100, 10.9688568
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8071365, 17.8153305
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1741791, 12.1828423
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0395660, 13.0439339
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6568069, 13.6616440
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6938095, 17.6946945
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2155380, 19.2223587
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0623245, 17.0643997
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0560989, 15.0574570
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2693558, 13.2737465
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3347244, 9.3417320

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0623938, upper bound: 6.0610104
time: 14.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0572732, upper bound: 6.0661297
time: 16.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7110596, 14.7115250
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5193596, 9.5169525
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0149193, 10.0133476
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6973686, 9.7014771
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8165817, 10.8141441
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4049988, 10.4038391
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0798569, 12.0821762
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0475540, 10.0418549
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4971237, 12.4867096
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9784775, 13.9779892
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4955292, 16.4890900
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3555374, 10.3504639
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8932762, 12.8991699
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8444519, 19.8586731
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8014679, 20.7882767
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4494171, 11.4510002
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1116943, 14.1012344
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2606812, 16.2545547
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3199158, 18.3149948
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8355789, 12.8336334
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1454697, 13.1487808
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3490143, 13.3470535
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9421463, 13.9399567
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7144852, 13.7135239
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8814697, 19.8763962
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9477234, 15.9410629
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4267426, 14.4249268
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6465302, 11.6451721
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4088821, 11.4063034
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9615097, 10.9697571
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8038254, 17.8186493
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1702728, 12.1867485
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0349579, 13.0485420
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6515884, 13.6668625
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6925430, 17.6959610
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2101669, 19.2277145
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0590134, 17.0677185
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0576553, 15.0559006
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2691574, 13.2739449
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3356934, 9.3407631

Time for backsubstitution: 2.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0673722, upper bound: 6.0585386
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0622565, upper bound: 6.0636589
time: 16.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7092285, 14.7133484
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5189018, 9.5174141
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0163765, 10.0118866
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7012215, 9.6976242
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8190079, 10.8117027
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4068985, 10.4019394
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0809860, 12.0810394
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0494537, 10.0399475
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4976120, 12.4862213
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9767303, 13.9797363
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4929504, 16.4916763
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3549118, 10.3510857
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8936653, 12.8987808
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8489227, 19.8541489
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7998047, 20.7899551
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4488831, 11.4514961
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1086884, 14.1042328
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2609406, 16.2542191
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3190613, 18.3158493
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8352737, 12.8339310
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1446838, 13.1495743
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3469467, 13.3491211
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9413834, 13.9407196
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7126694, 13.7153397
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8805237, 19.8772812
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9465485, 15.9422379
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4243546, 14.4273148
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6436920, 11.6480026
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4080658, 11.4071007
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9645004, 10.9667664
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8069839, 17.8154831
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1740875, 12.1829109
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0384521, 13.0450172
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6560974, 13.6622849
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6906357, 17.6978683
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2156296, 19.2222214
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0630264, 17.0636673
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0578003, 15.0557556
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2693863, 13.2737160
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3353500, 9.3411064

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0821840, upper bound: 6.0410181
time: 17.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0770672, upper bound: 6.0461458
time: 15.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7097397, 14.7128525
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5200386, 9.5162735
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0170326, 10.0112343
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7010078, 9.6978378
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8200760, 10.8106461
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4075851, 10.4012566
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0816574, 12.0803833
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0515594, 10.0378418
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.5014267, 12.4824028
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9775314, 13.9789276
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4955521, 16.4890747
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3564224, 10.3495789
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8925362, 12.8999100
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8440704, 19.8590088
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.8014984, 20.7882690
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4483032, 11.4520683
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1117706, 14.1011505
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2606888, 16.2545013
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3190308, 18.3158875
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8355865, 12.8336334
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1431808, 13.1510696
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3470230, 13.3490448
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9413757, 13.9407272
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7123489, 13.7156601
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8798676, 19.8780136
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9480057, 15.9407806
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4243317, 14.4273300
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6436844, 11.6480179
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4085846, 11.4065971
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9636002, 10.9676666
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8036728, 17.8188019
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1701813, 12.1868210
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0338440, 13.0496254
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6508789, 13.6675034
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6893692, 17.6991348
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2102585, 19.2275772
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0597153, 17.0669861
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0593567, 15.0541992
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2691879, 13.2739220
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3363228, 9.3401375

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0871537, upper bound: 6.0385411
time: 15.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0820424, upper bound: 6.0436714
time: 18.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7128448, 14.7097397
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5162773, 9.5200348
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0112343, 10.0170288
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6978340, 9.7010078
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8106461, 10.8200760
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4012527, 10.4075813
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0803833, 12.0816536
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0378418, 10.0515594
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4823990, 12.5014305
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9789276, 13.9775314
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4890671, 16.4955521
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3495789, 10.3564224
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8999062, 12.8925362
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8590088, 19.8440628
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7882690, 20.8014908
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4520721, 11.4483070
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1011505, 14.1117706
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2545013, 16.2606888
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3158875, 18.3190231
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8336334, 12.8355865
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1510696, 13.1431808
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3490448, 13.3470230
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9407272, 13.9413757
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7156601, 13.7123489
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8780060, 19.8798599
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9407806, 15.9480057
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4273300, 14.4243317
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6480179, 11.6436844
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4066010, 11.4085846
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9676743, 10.9636040
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8188095, 17.8036652
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1868210, 12.1701775
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0496216, 13.0338402
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6675034, 13.6508865
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6991425, 17.6893692
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2275772, 19.2102661
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0669861, 17.0597153
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0542068, 15.0593567
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2739182, 13.2691841
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3401375, 9.3363190

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0436714, upper bound: 6.0820424
time: 15.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0385411, upper bound: 6.0871537
time: 17.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7133484, 14.7092285
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5174141, 9.5188980
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0118904, 10.0163765
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6976280, 9.7012215
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8117065, 10.8190079
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4019394, 10.4068985
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0810394, 12.0809822
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0399475, 10.0494537
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4862137, 12.4976120
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9797287, 13.9767303
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4916763, 16.4929504
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3510818, 10.3549156
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8987846, 12.8936653
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8541565, 19.8489227
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7899628, 20.7997971
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4514923, 11.4488792
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1042328, 14.1086884
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2542267, 16.2609406
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3158569, 18.3190536
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8339310, 12.8352737
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1495743, 13.1446838
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3491211, 13.3469467
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9407196, 13.9413834
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7153397, 13.7126694
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8772888, 19.8805313
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9422302, 15.9465485
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4273148, 14.4243546
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6480026, 11.6436920
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4070969, 11.4080620
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9667664, 10.9645042
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8154831, 17.8069763
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1829071, 12.1740875
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0450134, 13.0384483
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6622849, 13.6560974
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6978683, 17.6906357
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2222214, 19.2156219
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0636673, 17.0630264
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0557556, 15.0578003
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2737198, 13.2693825
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3411064, 9.3353500

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0461459, upper bound: 6.0770671
time: 12.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0410181, upper bound: 6.0821840
time: 13.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7115250, 14.7110672
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5169563, 9.5193558
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0133476, 10.0149155
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7014732, 9.6973686
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8141403, 10.8165817
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4038391, 10.4049988
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0821762, 12.0798607
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0418549, 10.0475502
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4867020, 12.4971237
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9779892, 13.9784775
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4890976, 16.4955292
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3504639, 10.3555336
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8991737, 12.8932762
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8586731, 19.8444519
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7882843, 20.8014755
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4510040, 11.4494247
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1012421, 14.1116943
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2545547, 16.2606735
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3150024, 18.3199158
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8336334, 12.8355789
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1487808, 13.1454697
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3470535, 13.3490143
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9399567, 13.9421463
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7135239, 13.7144852
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8764038, 19.8814774
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9410553, 15.9477234
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4249268, 14.4267426
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6451721, 11.6465302
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4063034, 11.4088821
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9697571, 10.9615135
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8186569, 17.8038177
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1867523, 12.1702766
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0485382, 13.0349541
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6668625, 13.6515884
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6959610, 17.6925430
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2277145, 19.2101746
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0677109, 17.0590134
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0559006, 15.0576553
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2739487, 13.2691536
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3407631, 9.3356934

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0636589, upper bound: 6.0622565
time: 7.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0585386, upper bound: 6.0673722
time: 16.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7120285, 14.7105484
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5180931, 9.5182190
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0140038, 10.0142632
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7012672, 9.6975784
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8152008, 10.8155098
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4045258, 10.4043121
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0828323, 12.0791893
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0439606, 10.0454445
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4905319, 12.4933052
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9787903, 13.9776688
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4916992, 16.4929276
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3519745, 10.3540306
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8980446, 12.8944016
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8538208, 19.8493118
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7899628, 20.7997818
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4504242, 11.4499969
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1043243, 14.1086121
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2542725, 16.2609253
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3149567, 18.3199463
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8339386, 12.8352737
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1472855, 13.1469650
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3471298, 13.3489380
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9399414, 13.9421539
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7132034, 13.7148056
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8756866, 19.8821335
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9425125, 15.9462738
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4249039, 14.4267578
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6451569, 11.6465378
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4068069, 11.4083595
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9688568, 10.9624176
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8153305, 17.8071365
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1828461, 12.1741829
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0439301, 13.0395622
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6616440, 13.6568069
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6946945, 17.6938095
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2223587, 19.2155304
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0643997, 17.0623322
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0574570, 15.0560989
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2737503, 13.2693520
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3417320, 9.3347244

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0661297, upper bound: 6.0572732
time: 7.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0610104, upper bound: 6.0623938
time: 14.45 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7132568, 14.7093201
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5172539, 9.5190582
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0117989, 10.0164642
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6978264, 9.7010155
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8115387, 10.8191719
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4018936, 10.4069443
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0808640, 12.0811539
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0396423, 10.0497627
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4857101, 12.4981194
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9795227, 13.9769363
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4913788, 16.4932480
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3503113, 10.3556900
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8990211, 12.8934212
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8549042, 19.8481598
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7895660, 20.8001938
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4519196, 11.4484596
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1036758, 14.1092453
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2543030, 16.2608643
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3157959, 18.3190994
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8336334, 12.8355789
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1497955, 13.1444550
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3490143, 13.3470535
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9406586, 13.9414444
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7153091, 13.7126999
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8774109, 19.8803940
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9419250, 15.9468536
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4272690, 14.4243927
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6479645, 11.6437302
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4066086, 11.4085541
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9672470, 10.9640274
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8159409, 17.8065186
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1834564, 12.1735382
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0456848, 13.0377846
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6630249, 13.6553574
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6979446, 17.6905594
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2229996, 19.2148438
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0641708, 17.0625305
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0555038, 15.0580597
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2738266, 13.2692757
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3409691, 9.3354874

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0492092, upper bound: 6.0740028
time: 8.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0440849, upper bound: 6.0791202
time: 14.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7137756, 14.7088165
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5183983, 9.5179176
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0124550, 10.0158119
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.6976204, 9.7012291
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8126068, 10.8181114
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4025803, 10.4062576
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0815353, 12.0805016
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0417480, 10.0476570
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4895248, 12.4943047
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9803314, 13.9761353
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4939804, 16.4906387
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3518143, 10.3541832
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8978996, 12.8945503
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8500519, 19.8530197
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7912445, 20.7985001
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4513397, 11.4490318
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1067581, 14.1061630
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2540512, 16.2611465
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3157654, 18.3191299
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8339386, 12.8352737
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1482925, 13.1459579
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3490906, 13.3469772
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9406433, 13.9414520
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7149887, 13.7130203
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8767548, 19.8811264
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9433823, 15.9454041
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4272537, 14.4244080
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6479568, 11.6437454
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4071350, 11.4080505
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9663467, 10.9649277
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8126450, 17.8098297
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1795502, 12.1774483
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0410767, 13.0423851
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6578064, 13.6605759
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6966782, 17.6918259
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2176437, 19.2201996
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0608597, 17.0658417
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0570602, 15.0565033
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2736282, 13.2694740
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3419418, 9.3345184

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0541802, upper bound: 6.0715299
time: 18.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0490625, upper bound: 6.0766484
time: 17.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.2843571, 7.6583152, -8.2843571, 7.6583152, -14.7119370, 14.7106400
1: 1.0464747, 12.2789888, 1.0464747, 12.2789888, -9.5179329, 9.5183792
2: 1.5309100, 13.8900375, 1.5309100, 13.8900375, -10.0139122, 10.0143509
3: -7.5208917, 5.4945164, -7.5208917, 5.4945164, -9.7014656, 9.6973763
4: 2.1457684, 15.8209162, 2.1457684, 15.8209162, -10.8150406, 10.8156738
5: -4.3287992, 8.9530363, -4.3287992, 8.9530363, -10.4044800, 10.4043579
6: -29.4849968, -13.8607407, -29.4849968, -13.8607407, -12.0826569, 12.0793610
7: -3.2364430, 10.9765720, -3.2364430, 10.9765720, -10.0436478, 10.0457535
8: -9.7619944, 6.4818788, -9.7619944, 6.4818788, -12.4900131, 12.4938164
9: 0.1287656, 15.4720554, 0.1287656, 15.4720554, -13.9785843, 13.9778748
10: -11.6403475, 8.8810911, -11.6403475, 8.8810911, -16.4914017, 16.4932251
11: -11.3521109, 1.1453779, -11.3521109, 1.1453779, -10.3511963, 10.3548050
12: -24.6050873, -7.6571088, -24.6050873, -7.6571088, -12.8982887, 12.8941612
13: -13.6753120, 9.5110378, -13.6753120, 9.5110378, -19.8545837, 19.8485489
14: -30.3303909, -5.9055753, -30.3303909, -5.9055753, -20.7895813, 20.8001785
15: -6.7971234, 6.7155504, -6.7971234, 6.7155504, -11.4508514, 11.4495773
16: -10.1628942, 6.2712493, -10.1628942, 6.2712493, -14.1037598, 14.1091690
17: -32.1595192, -11.8610830, -32.1595192, -11.8610830, -16.2543488, 16.2608490
18: -9.6763000, 9.6430712, -9.6763000, 9.6430712, -18.3149109, 18.3199921
19: -4.4435468, 8.5375338, -4.4435468, 8.5375338, -12.9810810, 12.9810810
20: -6.0582066, 7.5867257, -6.0582066, 7.5867257, -12.8336334, 12.8355713
21: -5.9738121, 7.9137292, -5.9738121, 7.9137292, -13.8875408, 13.8875408
22: -6.3876781, 8.6714373, -6.3876781, 8.6714373, -13.1475067, 13.1467438
23: -7.2980547, 7.2273664, -7.2980547, 7.2273664, -13.3470230, 13.3490448
24: -5.3812890, 10.4683084, -5.3812890, 10.4683084, -13.9398804, 13.9422226
25: -6.8029170, 8.6521540, -6.8029170, 8.6521540, -13.7131729, 13.7148361
26: -12.6675320, 9.3369436, -12.6675320, 9.3369436, -19.8758240, 19.8820114
27: -8.9801788, 7.8022480, -8.9801788, 7.8022480, -15.9422073, 15.9465790
28: -6.8675041, 9.7838058, -6.8675041, 9.7838058, -14.4248657, 14.4267960
29: -9.5417509, 4.1442003, -9.5417509, 4.1442003, -11.6451187, 11.6465759
30: -14.7814779, 0.9732071, -14.7814779, 0.9732071, -11.4063110, 11.4088478
31: -7.2083097, 7.6287708, -7.2083097, 7.6287708, -14.8370800, 14.8370800
32: -20.4426403, -5.5111451, -20.4426403, -5.5111451, -10.9693375, 10.9619370
33: -36.7996712, -13.3733616, -36.7996712, -13.3733616, -17.8157883, 17.8066711
34: -40.4705505, -21.1543388, -40.4705505, -21.1543388, -12.1833954, 12.1736336
35: -27.7586746, -8.2660675, -27.7586746, -8.2660675, -13.0446014, 13.0388985
36: -23.8175621, -2.8019857, -23.8175621, -2.8019857, -13.6623840, 13.6560669
37: -44.8246765, -20.0948143, -44.8246765, -20.0948143, -17.6947708, 17.6937332
38: -28.4762802, -3.1663480, -28.4762802, -3.1663480, -19.2231369, 19.2147522
39: -32.1460724, -8.2421141, -32.1460724, -8.2421141, -17.0649033, 17.0618286
40: -42.2234612, -24.5833740, -42.2234612, -24.5833740, -15.0572052, 15.0563507
41: -24.0012341, -5.8546500, -24.0012341, -5.8546500, -13.2738419, 13.2692528
42: -25.3831577, -12.3419886, -25.3831577, -12.3419886, -9.3415985, 9.3348618

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1675
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 785
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1653
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1339
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1337
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1660
type: RSZ, layer: 1, pos: 1022
type: RSZ, layer: 1, pos: 772
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1623
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1310
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1473
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1409
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 969
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 795
type: RSZ, layer: 1, pos: 1625
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 769
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 880
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1786

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0691890, upper bound: 6.0542048
time: 15.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -6.0640731, upper bound: 6.0593288
time: 18.82 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 37.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0361207, upper bound: 6.0895338
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0309814, upper bound: 6.0946299
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0385991, upper bound: 6.0845744
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0334619, upper bound: 6.0896771
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0561274, upper bound: 6.0697689
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0509956, upper bound: 6.0748726
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0586027, upper bound: 6.0647984
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0534718, upper bound: 6.0699093
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0416643, upper bound: 6.0815156
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0365292, upper bound: 6.0866199
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0466460, upper bound: 6.0790455
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0415137, upper bound: 6.0841545
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0616674, upper bound: 6.0617341
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0565378, upper bound: 6.0668477
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0666421, upper bound: 6.0592624
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0615182, upper bound: 6.0643766
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0568534, upper bound: 6.0690429
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0517283, upper bound: 6.0741516
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0593288, upper bound: 6.0640731
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0542048, upper bound: 6.0691890
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0766484, upper bound: 6.0490625
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0715299, upper bound: 6.0541802
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0791202, upper bound: 6.0440849
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0740028, upper bound: 6.0492092
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0623938, upper bound: 6.0610104
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0572732, upper bound: 6.0661297
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0673722, upper bound: 6.0585386
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0622565, upper bound: 6.0636589
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0821840, upper bound: 6.0410181
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0770672, upper bound: 6.0461458
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0871537, upper bound: 6.0385411
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0820424, upper bound: 6.0436714
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0436714, upper bound: 6.0820424
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0385411, upper bound: 6.0871537
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0461459, upper bound: 6.0770671
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0410181, upper bound: 6.0821840
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0636589, upper bound: 6.0622565
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0585386, upper bound: 6.0673722
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0661297, upper bound: 6.0572732
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0610104, upper bound: 6.0623938
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0492092, upper bound: 6.0740028
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0440849, upper bound: 6.0791202
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0541802, upper bound: 6.0715299
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0490625, upper bound: 6.0766484
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0691890, upper bound: 6.0542048
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 37.02
Output dim: 4, lower bound: -6.0640731, upper bound: 6.0593288
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.02
Output dim: 4, lower bound: -6.0743955, upper bound: 6.0570971
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.02
Output dim: 4, lower bound: -6.0646200, upper bound: 6.0668862
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.02
Output dim: 4, lower bound: -6.0670912, upper bound: 6.0619114
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.02
Output dim: 4, lower bound: -6.0843980, upper bound: 6.0468904
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.02
Output dim: 4, lower bound: -6.0868635, upper bound: 6.0419083
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.02
Output dim: 4, lower bound: -6.0701528, upper bound: 6.0588467
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.02
Output dim: 4, lower bound: -6.0751162, upper bound: 6.0563713
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.02
Output dim: 4, lower bound: -6.0899208, upper bound: 6.0388430
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.02
Output dim: 4, lower bound: -6.0948736, upper bound: 6.0363645

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 32.61 + 1776.20 = 1808.81 seconds
