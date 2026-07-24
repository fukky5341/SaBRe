## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 3600 seconds
Split limit: 100
Threshold: 18.0421680717


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2290497, 63.2290497)
1: (-21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1254883, 42.1254883)
2: (-15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5613937, 37.5613937)
3: (-28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717)
4: (-26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792)
5: (-24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062)
6: (-53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7803345, 41.7803383)
7: (-34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2268066, 48.2268066)
8: (-40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2519226, 57.2519226)
9: (-23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606)
10: (-36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148)
11: (-27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099)
12: (-45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0751190, 62.0751190)
13: (-37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453)
14: (-52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0949097, 68.0949249)
15: (-21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095)
16: (-35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5010834, 55.5010834)
17: (-48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583)
18: (-22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815)
19: (-17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714)
20: (-24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767)
21: (-22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478)
22: (-13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2794189, 37.2794189)
23: (-16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198)
24: (-16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253)
25: (-16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3220367, 43.3220367)
26: (-27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738)
27: (-22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061)
28: (-17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1172943, 42.1172981)
29: (-9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7173233, 30.7173233)
30: (-29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949)
31: (-23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085)
32: (-47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8463669, 40.8463669)
33: (-70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2104034, 67.2104034)
34: (-63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3345490, 46.3345490)
35: (-41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8395233, 53.8395157)
36: (-42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0696106, 58.0696182)
37: (-71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5504761, 66.5504761)
38: (-59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630)
39: (-74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3985901, 74.3985748)
40: (-70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8707275, 41.8707352)
41: (-47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5570602, 46.5570602)
42: (-46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1460800, 39.1460800)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.74 + 48.07 = 50.80 seconds
status: Status.UNKNOWN
relational distance
Output dim: 29, lower bound: -18.0602283, upper bound: 18.0602283

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 897

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1320

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0589244, upper bound: 18.0524274
time: 36.31 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0524275, upper bound: 18.0589244
time: 36.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 72.85 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 72.85
Output dim: 29, lower bound: -18.0589244, upper bound: 18.0524274
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 72.85
Output dim: 29, lower bound: -18.0524275, upper bound: 18.0589244

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2295380, 63.2293549
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1228943, 42.1231384
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5600281, 37.5601273
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7780228, 41.7775116
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2238159, 48.2241592
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2519073, 57.2519379
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0753326, 62.0753937
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0967102, 68.0973511
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4970703, 55.4975586
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2772751, 37.2770844
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3207092, 43.3205109
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1173172, 42.1173096
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7160110, 30.7158432
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8475342, 40.8472366
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2078400, 67.2069092
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3299103, 46.3294067
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8387299, 53.8387833
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0699615, 58.0698700
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5439301, 66.5433960
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3995056, 74.3992615
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8558350, 41.8552170
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5553360, 46.5551224
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1391525, 39.1386642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1630

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1307

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0514686, upper bound: 18.0449715
time: 42.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0514686, upper bound: 18.0449715
time: 47.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2293549, 63.2295380
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1231384, 42.1228943
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5601196, 37.5600281
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7775116, 41.7780228
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2241516, 48.2238083
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2519379, 57.2519073
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0753937, 62.0753326
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0973511, 68.0967102
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4975586, 55.4970703
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2770920, 37.2772751
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3205109, 43.3207092
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1173172, 42.1173096
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7158432, 30.7160149
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8472443, 40.8475418
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2069244, 67.2078400
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3294067, 46.3299103
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8387756, 53.8387299
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0698700, 58.0699615
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5433960, 66.5439301
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3992615, 74.3995209
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8552246, 41.8558426
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5551224, 46.5553360
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1386566, 39.1391525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1558

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1788

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0520052, upper bound: 18.0553111
time: 44.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0488147, upper bound: 18.0585021
time: 47.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 94.33 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 94.33
Output dim: 29, lower bound: -18.0514686, upper bound: 18.0449715
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 94.33
Output dim: 29, lower bound: -18.0514686, upper bound: 18.0449715
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 94.33
Output dim: 29, lower bound: -18.0520052, upper bound: 18.0553111
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 94.33
Output dim: 29, lower bound: -18.0488147, upper bound: 18.0585021

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2302551, 63.2293396
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1228943, 42.1231613
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5600510, 37.5601196
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7789154, 41.7774582
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2238159, 48.2241898
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2520599, 57.2519150
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0753326, 62.0754547
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0966949, 68.0978546
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4970856, 55.4975739
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2772675, 37.2772675
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3206558, 43.3205032
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1173019, 42.1176186
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7160110, 30.7159576
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8481903, 40.8472137
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2085266, 67.2069092
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3301392, 46.3293839
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8387146, 53.8390503
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0702515, 58.0698547
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5446472, 66.5433502
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3997345, 74.3992462
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8571167, 41.8551636
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5562439, 46.5550766
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1411743, 39.1385803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1006

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1570

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0260900, upper bound: 18.0197143
time: 42.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0262120, upper bound: 18.0195922
time: 42.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2295227, 63.2293549
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1228943, 42.1231232
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5600204, 37.5601273
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7779694, 41.7775116
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2238159, 48.2241592
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2518768, 57.2519379
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0753326, 62.0753937
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0967102, 68.0973358
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4970703, 55.4975739
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2772751, 37.2770844
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3207016, 43.3205109
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1173172, 42.1172981
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7160110, 30.7158356
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8475189, 40.8472366
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2078400, 67.2069092
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3298950, 46.3294067
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8387299, 53.8387680
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0699463, 58.0698700
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5438843, 66.5433960
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3995209, 74.3992615
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8557892, 41.8552170
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5552979, 46.5551224
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1390686, 39.1386642

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1458

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0514081, upper bound: 18.0442677
time: 33.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0507652, upper bound: 18.0449110
time: 49.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2215424, 63.2224426
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1185074, 42.1188126
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5537872, 37.5544815
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7716980, 41.7717552
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2168732, 48.2172775
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2461395, 57.2468643
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0773010, 62.0768585
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0910645, 68.0912018
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4932022, 55.4936371
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2687302, 37.2676697
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3165131, 43.3164444
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1148911, 42.1145706
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7107697, 30.7100716
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8439026, 40.8439484
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1909790, 67.1897736
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3127747, 46.3104172
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8305817, 53.8295517
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0696564, 58.0697327
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5422668, 66.5427399
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3958740, 74.3957214
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8471756, 41.8476257
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5554199, 46.5556412
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1394577, 39.1399803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 926

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1388

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0467825, upper bound: 18.0548888
time: 39.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0515829, upper bound: 18.0500885
time: 45.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2222748, 63.2217255
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1190567, 42.1182709
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5545807, 37.5536804
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7712402, 41.7722130
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2176208, 48.2165298
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2469025, 57.2461014
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0769196, 62.0772400
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0918427, 68.0904236
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4941177, 55.4927216
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2674866, 37.2689133
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3162384, 43.3167114
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1145706, 42.1148834
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7098999, 30.7109489
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8436432, 40.8441925
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1888428, 67.1918945
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3099213, 46.3132858
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8296051, 53.8305206
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0696411, 58.0697479
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5422211, 66.5427856
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3954468, 74.3961487
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8470078, 41.8477936
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5554352, 46.5556335
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1394882, 39.1399460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 929

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 578

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0452497, upper bound: 18.0549370
time: 31.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0452497, upper bound: 18.0549370
time: 31.72 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 65.05 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 65.05
Output dim: 29, lower bound: -18.0260900, upper bound: 18.0197143
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 65.05
Output dim: 29, lower bound: -18.0262120, upper bound: 18.0195922
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 65.05
Output dim: 29, lower bound: -18.0514081, upper bound: 18.0442677
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 65.05
Output dim: 29, lower bound: -18.0507652, upper bound: 18.0449110
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 65.05
Output dim: 29, lower bound: -18.0467825, upper bound: 18.0548888
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 65.05
Output dim: 29, lower bound: -18.0515829, upper bound: 18.0500885
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 65.05
Output dim: 29, lower bound: -18.0452497, upper bound: 18.0549370
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 65.05
Output dim: 29, lower bound: -18.0452497, upper bound: 18.0549370

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2262421, 63.2261963
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1232758, 42.1234818
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5552063, 37.5558243
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7722778, 41.7715149
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2208252, 48.2216263
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2494049, 57.2497711
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0686035, 62.0677338
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1015472, 68.1016846
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4981689, 55.4984207
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2748795, 37.2744904
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3184662, 43.3179855
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1186600, 42.1187897
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7154579, 30.7151794
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8452148, 40.8449326
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2065887, 67.2053680
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3234329, 46.3223190
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8350372, 53.8339233
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0694580, 58.0692825
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5440369, 66.5441895
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.4006348, 74.4005585
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8533173, 41.8535614
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5564423, 46.5566101
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1373444, 39.1369476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 899

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1007

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0480266, upper bound: 18.0431553
time: 39.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0502965, upper bound: 18.0408852
time: 42.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2263336, 63.2261200
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1232452, 42.1235046
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5556946, 37.5553284
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7719803, 41.7718124
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2213135, 48.2211533
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2497253, 57.2494583
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0676727, 62.0686493
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1010742, 68.1021729
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4979248, 55.4986572
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2746964, 37.2746658
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3181763, 43.3182831
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1188126, 42.1186523
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7153587, 30.7152786
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8452301, 40.8449249
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2062531, 67.2057190
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3227921, 46.3229523
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8338928, 53.8350677
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0693970, 58.0693436
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5446472, 66.5435638
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.4007874, 74.4003754
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8541412, 41.8527374
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5567780, 46.5562820
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1373520, 39.1369400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1284

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 881

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0502750, upper bound: 18.0444658
time: 42.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0503200, upper bound: 18.0444207
time: 34.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2178802, 63.2183228
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1155396, 42.1154480
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5496521, 37.5498352
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7624588, 41.7636032
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2132263, 48.2129898
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2437134, 57.2440720
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0716705, 62.0718689
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0915527, 68.0910797
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4896011, 55.4895096
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2648010, 37.2642746
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3160477, 43.3159714
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1150055, 42.1146851
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7081146, 30.7077904
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8359375, 40.8368149
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1759033, 67.1763763
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2933350, 46.2932892
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8235626, 53.8234177
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0686340, 58.0688324
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5370331, 66.5381927
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3916931, 74.3920898
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8366547, 41.8383408
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5506897, 46.5514374
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1318665, 39.1331863

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 519

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1465

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0111394, upper bound: 18.0192438
time: 35.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0111394, upper bound: 18.0192438
time: 35.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2174377, 63.2187805
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1151581, 42.1158295
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5491180, 37.5503540
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7635498, 41.7625122
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2125854, 48.2136230
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2433472, 57.2444382
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0722961, 62.0712433
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0909424, 68.0916901
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4890823, 55.4900360
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2653351, 37.2637405
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3160477, 43.3159790
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1149902, 42.1147003
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7084961, 30.7074165
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8367615, 40.8359909
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1775818, 67.1746979
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2956390, 46.2909698
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8244476, 53.8225403
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0687561, 58.0687103
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5377197, 66.5375061
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3922424, 74.3915558
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8378906, 41.8370895
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5512238, 46.5509033
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1326675, 39.1323929

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 930

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 883

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0510577, upper bound: 18.0467188
time: 35.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0482140, upper bound: 18.0495631
time: 36.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2222748, 63.2216644
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1191101, 42.1181717
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5546341, 37.5534515
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7700119, 41.7729492
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2175446, 48.2164612
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2468872, 57.2461700
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0764313, 62.0773773
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0920258, 68.0901794
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4941025, 55.4927292
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2676239, 37.2688065
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3162842, 43.3166580
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1146240, 42.1146965
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7099457, 30.7109299
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8428802, 40.8445587
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1880493, 67.1921539
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3091736, 46.3140259
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8294373, 53.8305511
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0694885, 58.0698166
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5419617, 66.5428467
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3953247, 74.3961334
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8462143, 41.8484573
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5548248, 46.5558777
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1383286, 39.1406059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1307

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 848

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0412169, upper bound: 18.0527659
time: 45.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0430788, upper bound: 18.0509040
time: 47.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2221985, 63.2217255
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1189575, 42.1182709
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5543442, 37.5536804
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7712402, 41.7709808
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2175598, 48.2165298
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2469025, 57.2460938
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0769196, 62.0767517
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0916138, 68.0904236
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4941177, 55.4927063
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2673721, 37.2689133
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3161926, 43.3167114
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1143799, 42.1148834
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7098846, 30.7109489
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8436432, 40.8434219
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1888428, 67.1911011
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3099213, 46.3125458
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8296051, 53.8303680
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0696411, 58.0695877
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5422211, 66.5425415
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3954468, 74.3960266
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8470078, 41.8469925
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5554352, 46.5550385
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1394882, 39.1387825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 783

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 892

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0344798, upper bound: 18.0357638
time: 36.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0260797, upper bound: 18.0441633
time: 39.51 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 78.46 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 78.46
Output dim: 29, lower bound: -18.0480266, upper bound: 18.0431553
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 78.46
Output dim: 29, lower bound: -18.0502965, upper bound: 18.0408852
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 78.46
Output dim: 29, lower bound: -18.0502750, upper bound: 18.0444658
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 78.46
Output dim: 29, lower bound: -18.0503200, upper bound: 18.0444207
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 78.46
Output dim: 29, lower bound: -18.0111394, upper bound: 18.0192438
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 78.46
Output dim: 29, lower bound: -18.0111394, upper bound: 18.0192438
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 78.46
Output dim: 29, lower bound: -18.0510577, upper bound: 18.0467188
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 78.46
Output dim: 29, lower bound: -18.0482140, upper bound: 18.0495631
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 78.46
Output dim: 29, lower bound: -18.0412169, upper bound: 18.0527659
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 78.46
Output dim: 29, lower bound: -18.0430788, upper bound: 18.0509040
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 78.46
Output dim: 29, lower bound: -18.0344798, upper bound: 18.0357638
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 78.46
Output dim: 29, lower bound: -18.0260797, upper bound: 18.0441633

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2209778, 63.2201691
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1222458, 42.1223145
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5513458, 37.5513916
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7654877, 41.7653046
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2188873, 48.2194061
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2476196, 57.2477112
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0604095, 62.0605621
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1021271, 68.1023865
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4986115, 55.4988556
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2740021, 37.2737274
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3188934, 43.3184128
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1190414, 42.1190872
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7146149, 30.7144623
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8402557, 40.8404694
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1998596, 67.1994476
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3089600, 46.3096313
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8258514, 53.8258743
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0691071, 58.0690002
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5447388, 66.5448151
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3999329, 74.3999176
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8511810, 41.8514633
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5555573, 46.5557022
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1353302, 39.1348343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 807

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1551

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0458049, upper bound: 18.0411116
time: 32.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0459854, upper bound: 18.0409316
time: 35.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2202301, 63.2209167
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1221085, 42.1224594
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5507965, 37.5519409
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7660751, 41.7647171
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2186127, 48.2196808
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2473602, 57.2479706
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0614166, 62.0595398
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1022644, 68.1022644
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4985809, 55.4988861
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2741165, 37.2736130
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3188934, 43.3183975
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1189651, 42.1191635
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7147446, 30.7143402
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8407135, 40.8400040
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2006989, 67.1986084
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3107605, 46.3078461
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8269806, 53.8247375
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0691528, 58.0689545
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5447083, 66.5448608
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3999939, 74.3998413
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8512115, 41.8514252
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5555573, 46.5557022
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1352386, 39.1349258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1619

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0497370, upper bound: 18.0403818
time: 54.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0497971, upper bound: 18.0403011
time: 36.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2265625, 63.2259827
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1223297, 42.1227875
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5557251, 37.5553207
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7718430, 41.7716751
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2199707, 48.2201080
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2488403, 57.2488098
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0671387, 62.0688324
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1004486, 68.1020660
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4957123, 55.4971542
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2724075, 37.2713470
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3167191, 43.3161240
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1180878, 42.1175766
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7148056, 30.7141914
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8451385, 40.8449020
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2040100, 67.2029114
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3226776, 46.3228378
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8336487, 53.8353577
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0693665, 58.0693436
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5438538, 66.5425262
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.4000549, 74.3994446
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8527603, 41.8512268
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5564423, 46.5559464
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1372757, 39.1367950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 866

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 964

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0309990, upper bound: 18.0392517
time: 35.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0450608, upper bound: 18.0251897
time: 61.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2262115, 63.2263336
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1225128, 42.1226044
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5557098, 37.5553436
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7718430, 41.7716751
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2202454, 48.2198257
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2490845, 57.2485809
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0678406, 62.0681152
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1009521, 68.1015472
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4964142, 55.4964600
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2713852, 37.2723770
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3160477, 43.3167953
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1177368, 42.1179352
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7142715, 30.7147217
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8451996, 40.8448410
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2034760, 67.2034607
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3226776, 46.3228302
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8341675, 53.8348465
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0693665, 58.0693436
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5436096, 66.5427704
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3998718, 74.3996277
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8526077, 41.8513870
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5564423, 46.5559311
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1372070, 39.1368637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1332

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1546

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0498828, upper bound: 18.0440527
time: 39.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0499520, upper bound: 18.0439836
time: 45.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2160339, 63.2185822
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1144867, 42.1156921
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5477753, 37.5500259
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7622604, 41.7561798
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2122498, 48.2132416
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2431488, 57.2442169
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0715485, 62.0694427
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0891266, 68.0913086
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4882889, 55.4897537
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2650757, 37.2634888
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3160095, 43.3159561
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1149902, 42.1147003
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7080116, 30.7068939
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8364258, 40.8328705
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1768036, 67.1709290
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2947769, 46.2857208
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8235931, 53.8220367
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0686646, 58.0682526
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5376587, 66.5372925
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3922272, 74.3913422
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8372040, 41.8325577
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5510025, 46.5490952
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1318665, 39.1281891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 822

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1407

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0471852, upper bound: 18.0461244
time: 56.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0504632, upper bound: 18.0428456
time: 57.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2172241, 63.2173767
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1150360, 42.1151428
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5487823, 37.5489960
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7572174, 41.7612228
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2122192, 48.2132797
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2431335, 57.2442322
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0705109, 62.0704956
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0905609, 68.0898895
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4887924, 55.4892578
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2650909, 37.2634811
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3160095, 43.3159485
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1149902, 42.1146927
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7079735, 30.7069359
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8336487, 40.8356400
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1738129, 67.1739197
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2903976, 46.2901077
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8239441, 53.8216858
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0682831, 58.0686188
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5375061, 66.5374603
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3920441, 74.3915253
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8333588, 41.8364105
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5494156, 46.5506897
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1284637, 39.1315842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1762

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0408361, upper bound: 18.0484550
time: 33.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0471059, upper bound: 18.0421857
time: 35.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2223053, 63.2215271
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1191101, 42.1183090
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5546494, 37.5534592
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7695618, 41.7730446
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2175598, 48.2167816
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2469025, 57.2463303
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0763092, 62.0773926
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0920105, 68.0902100
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4941177, 55.4930496
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2678833, 37.2687607
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3162537, 43.3165894
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1146545, 42.1146736
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7100601, 30.7109146
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8427277, 40.8445663
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1884918, 67.1921234
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3090820, 46.3139114
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8297424, 53.8305206
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0694885, 58.0698166
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5425568, 66.5428314
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3956604, 74.3960876
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8461838, 41.8484573
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5548019, 46.5558777
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1380692, 39.1405945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 999

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1604

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0349623, upper bound: 18.0526282
time: 39.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0410791, upper bound: 18.0465140
time: 93.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2221527, 63.2216644
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1191101, 42.1181564
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5546341, 37.5534592
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7700119, 41.7724991
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2175446, 48.2164612
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2468872, 57.2461700
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0764313, 62.0772552
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0920258, 68.0901794
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4941025, 55.4927368
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2675934, 37.2688065
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3162079, 43.3166580
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1146088, 42.1146965
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7099304, 30.7109299
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8428802, 40.8444061
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1880341, 67.1921539
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3090668, 46.3140259
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8294220, 53.8305511
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0694885, 58.0698166
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5419617, 66.5428467
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3952942, 74.3961334
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8461990, 41.8484573
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5548325, 46.5558777
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1383286, 39.1403503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1325

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 851

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0417937, upper bound: 18.0445933
time: 37.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0367681, upper bound: 18.0496188
time: 39.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2219086, 63.2207489
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1184845, 42.1164856
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5536880, 37.5515213
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7664948, 41.7693405
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2167053, 48.2139130
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2465363, 57.2449875
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0741882, 62.0758209
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0924530, 68.0891266
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4935379, 55.4908829
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2662201, 37.2685394
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3162231, 43.3166809
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1146393, 42.1148682
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7090912, 30.7107086
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8394241, 40.8421097
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1803894, 67.1888580
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2986679, 46.3095398
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8253174, 53.8292542
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0688477, 58.0693283
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5367737, 66.5406647
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3922272, 74.3950500
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8397675, 41.8448181
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5512466, 46.5535965
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1339645, 39.1368942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2044
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1351

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 975

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0154640, upper bound: 18.0440830
time: 40.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0259994, upper bound: 18.0335494
time: 36.02 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 79.05 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0458049, upper bound: 18.0411116
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0459854, upper bound: 18.0409316
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0497370, upper bound: 18.0403818
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0497971, upper bound: 18.0403011
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0309990, upper bound: 18.0392517
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0450608, upper bound: 18.0251897
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0498828, upper bound: 18.0440527
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0499520, upper bound: 18.0439836
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0471852, upper bound: 18.0461244
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0504632, upper bound: 18.0428456
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0408361, upper bound: 18.0484550
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0471059, upper bound: 18.0421857
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0349623, upper bound: 18.0526282
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0410791, upper bound: 18.0465140
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0417937, upper bound: 18.0445933
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0367681, upper bound: 18.0496188
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0154640, upper bound: 18.0440830
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 79.05
Output dim: 29, lower bound: -18.0259994, upper bound: 18.0335494

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2209625, 63.2199707
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1222534, 42.1222839
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5513000, 37.5512390
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7654343, 41.7652969
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2188873, 48.2193451
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2476654, 57.2476654
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0601959, 62.0604401
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1021271, 68.1024017
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4985962, 55.4988098
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2737961, 37.2735977
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3187561, 43.3182602
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1190109, 42.1190948
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7144470, 30.7143860
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8402557, 40.8404846
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1998444, 67.1995544
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3089142, 46.3097153
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8258362, 53.8259888
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0690918, 58.0689545
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5444336, 66.5445099
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3998718, 74.3998566
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8510742, 41.8514099
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5552673, 46.5554276
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1351776, 39.1346817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 999

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 807

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0393433, upper bound: 18.0405020
time: 38.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0452873, upper bound: 18.0366813
time: 40.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2208099, 63.2201233
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1222229, 42.1223221
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5511932, 37.5513458
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7654800, 41.7652588
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2188263, 48.2193985
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2475891, 57.2477570
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0602722, 62.0603638
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1021271, 68.1024017
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4985657, 55.4988403
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2738724, 37.2735214
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3187408, 43.3182831
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1190414, 42.1190643
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7145386, 30.7142944
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8402557, 40.8404694
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1999664, 67.1994324
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3090515, 46.3095856
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8259583, 53.8258591
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0690918, 58.0689697
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5444031, 66.5445251
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3998718, 74.3998566
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8511047, 41.8513794
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5552673, 46.5554276
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1351700, 39.1346855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 941

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0417927, upper bound: 18.0404335
time: 44.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0454861, upper bound: 18.0367290
time: 41.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2180786, 63.2189941
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1203384, 42.1209106
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5467377, 37.5484467
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7630081, 41.7612915
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2154083, 48.2168427
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2445526, 57.2453308
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0540161, 62.0511627
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1024933, 68.1024933
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4985657, 55.4988098
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2750092, 37.2746048
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3181076, 43.3176575
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1170578, 42.1174774
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7147179, 30.7143173
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8366776, 40.8354034
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1957703, 67.1923218
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2941895, 46.2888489
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8159943, 53.8120499
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0682526, 58.0679626
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5421906, 66.5422516
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3965607, 74.3961334
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8496857, 41.8496933
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5526810, 46.5525970
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1337433, 39.1330872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 997

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 796

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0486675, upper bound: 18.0343696
time: 43.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0437251, upper bound: 18.0393118
time: 33.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2183228, 63.2187500
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1205521, 42.1207047
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5473022, 37.5478668
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7626495, 41.7616501
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2157745, 48.2164688
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2447205, 57.2451706
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0530548, 62.0521088
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1024780, 68.1025085
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4985352, 55.4988480
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2751007, 37.2745132
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3181686, 43.3175964
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1172867, 42.1172371
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7147331, 30.7142982
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8361282, 40.8359528
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1943970, 67.1936951
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2917633, 46.2912750
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8143005, 53.8137436
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0681458, 58.0680695
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5420532, 66.5424042
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3962555, 74.3964081
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8494720, 41.8499069
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5524368, 46.5528412
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1333923, 39.1334457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 873

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 535

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0045654, upper bound: 17.9954802
time: 12.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0045654, upper bound: 17.9954802
time: 12.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2265625, 63.2259674
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1222229, 42.1227875
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5552979, 37.5553207
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7712250, 41.7716751
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2188568, 48.2201080
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2479401, 57.2488098
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0671387, 62.0688019
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1004486, 68.1018219
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4948883, 55.4971542
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2724075, 37.2690201
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3167191, 43.3147888
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1180878, 42.1171799
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7148056, 30.7129745
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8439941, 40.8449020
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2040100, 67.2028503
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3226547, 46.3228378
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8336487, 53.8351517
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0693054, 58.0693436
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5430450, 66.5425262
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3999329, 74.3994446
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8503799, 41.8512268
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5550079, 46.5559464
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1363297, 39.1367950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 794

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1533

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0450175, upper bound: 18.0228725
time: 34.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0427440, upper bound: 18.0251463
time: 33.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2273865, 63.2263184
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1224976, 42.1228714
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5557098, 37.5553360
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7725754, 41.7715759
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2201843, 48.2202606
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2492676, 57.2485809
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0678711, 62.0681610
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1008759, 68.1023102
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4963989, 55.4965363
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2713470, 37.2727203
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3160477, 43.3168106
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1177216, 42.1184044
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7142563, 30.7150269
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8459625, 40.8447571
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2051392, 67.2032471
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3240585, 46.3226624
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8341370, 53.8348007
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0697021, 58.0693283
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5455627, 66.5425262
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.4007263, 74.3995514
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8543854, 41.8511810
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5578308, 46.5557556
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1394958, 39.1366119

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 527

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 998

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0480355, upper bound: 18.0422045
time: 39.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0480355, upper bound: 18.0422045
time: 42.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2262115, 63.2263336
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1225128, 42.1225891
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5557098, 37.5553436
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7717438, 41.7716751
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2202454, 48.2197647
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2490845, 57.2485809
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0678406, 62.0681305
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1009521, 68.1014862
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4964142, 55.4964371
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2713852, 37.2723389
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3160477, 43.3167953
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1177368, 42.1179352
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7142715, 30.7147064
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8450928, 40.8448410
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2032623, 67.2034607
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3225174, 46.3228302
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8341370, 53.8348465
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0693054, 58.0693436
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5433960, 66.5427704
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3998108, 74.3996277
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8524017, 41.8513870
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5562744, 46.5559311
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1369553, 39.1368637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1305

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0493928, upper bound: 18.0434797
time: 40.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0494526, upper bound: 18.0434025
time: 48.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2157135, 63.2182312
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1132355, 42.1141739
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5472107, 37.5493851
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7686234, 41.7620468
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2094879, 48.2101440
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2411194, 57.2418747
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0725555, 62.0703430
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0905914, 68.0924988
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4869003, 55.4878387
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2607269, 37.2597275
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3149414, 43.3150024
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1132889, 42.1132126
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7054367, 30.7046967
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8379974, 40.8343430
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1737213, 67.1687622
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2899780, 46.2817307
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8233032, 53.8222733
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0690155, 58.0685577
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5316620, 66.5317535
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3895569, 74.3890839
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8371811, 41.8325424
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5516815, 46.5496674
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1379547, 39.1337051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1009

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1387

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0471327, upper bound: 18.0460979
time: 40.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0471613, upper bound: 18.0460945
time: 47.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2156830, 63.2182617
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1129608, 42.1144485
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5471344, 37.5494766
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7681198, 41.7625504
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2091522, 48.2104874
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2407990, 57.2422028
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0724487, 62.0704498
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0903320, 68.0927734
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4863815, 55.4883652
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2613144, 37.2591400
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3150635, 43.3149033
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1135025, 42.1130066
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7058105, 30.7043228
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8378906, 40.8344498
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1746521, 67.1678314
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2907867, 46.2809219
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8238220, 53.8217545
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0689697, 58.0686035
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5321198, 66.5312958
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3899841, 74.3886719
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8371811, 41.8325424
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5515747, 46.5497665
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1373749, 39.1342850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 776

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 964

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0311861, upper bound: 18.0376316
time: 37.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0452490, upper bound: 18.0235655
time: 41.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1967316, 63.1942902
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1123886, 42.1122894
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5395737, 37.5388031
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7174530, 41.7260742
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2143402, 48.2157364
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2398682, 57.2409821
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0621338, 62.0628052
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0848846, 68.0838318
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4833298, 55.4835663
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2593994, 37.2579117
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3154449, 43.3153763
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1113510, 42.1112061
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7010651, 30.7005196
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8163147, 40.8202667
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1690521, 67.1697845
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2672729, 46.2691040
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8244019, 53.8220901
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0683441, 58.0686951
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5444641, 66.5437775
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3883514, 74.3874817
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8186493, 41.8233566
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5489197, 46.5502014
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1209183, 39.1245232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1576

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1007

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0374524, upper bound: 18.0473425
time: 40.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0397221, upper bound: 18.0450730
time: 40.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1941376, 63.1968842
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1121750, 42.1125031
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5385971, 37.5397797
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7220688, 41.7214661
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2146759, 48.2154160
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2398834, 57.2409668
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0628204, 62.0621338
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0845032, 68.0842133
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4831009, 55.4837952
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2595291, 37.2577820
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3154449, 43.3153839
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1115036, 42.1110611
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7015533, 30.7000275
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8182678, 40.8183136
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1696625, 67.1691589
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2693939, 46.2669983
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8243408, 53.8221436
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0683594, 58.0686798
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5438232, 66.5444183
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3879852, 74.3878326
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8202972, 41.8217239
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5489197, 46.5501862
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1214066, 39.1240349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1542

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 992

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0423306, upper bound: 18.0374285
time: 46.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0423306, upper bound: 18.0374285
time: 46.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2182159, 63.2168884
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1185226, 42.1176834
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5506744, 37.5489807
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7603455, 41.7647743
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2147827, 48.2138748
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2446289, 57.2438431
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0749969, 62.0768890
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0943298, 68.0926208
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4948654, 55.4938126
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2652740, 37.2663956
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3153152, 43.3156967
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1154327, 42.1154404
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7082672, 30.7093124
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8393555, 40.8414688
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1915588, 67.1955261
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2998352, 46.3053894
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8329773, 53.8342361
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0693665, 58.0697327
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5423279, 66.5422516
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3947906, 74.3951416
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8478775, 41.8501129
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5564728, 46.5574875
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1362457, 39.1388168

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1309

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 796

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0337879, upper bound: 18.0463608
time: 38.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0286905, upper bound: 18.0514580
time: 36.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2176666, 63.2174377
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1184769, 42.1177292
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5501709, 37.5494995
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7612915, 41.7638359
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2146606, 48.2139969
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2444153, 57.2440720
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0758057, 62.0760803
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0944366, 68.0925293
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4948807, 55.4937897
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2655106, 37.2661591
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3153610, 43.3156509
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1154327, 42.1154594
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7084579, 30.7091141
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8396301, 40.8411942
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1918945, 67.1951904
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3005676, 46.3046646
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8334656, 53.8337555
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0693970, 58.0697021
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5419769, 66.5426025
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3947296, 74.3952026
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8478317, 41.8501587
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5564117, 46.5575485
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1362915, 39.1387711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 836

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0199779, upper bound: 18.0270895
time: 36.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0184712, upper bound: 18.0273288
time: 37.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2221527, 63.2216492
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1187286, 42.1178818
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5545273, 37.5533524
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7693253, 41.7714233
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2174377, 48.2163010
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2469025, 57.2461700
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0761108, 62.0769501
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0907135, 68.0892334
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4932327, 55.4919434
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2672577, 37.2684479
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3160553, 43.3165131
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1144562, 42.1145401
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7097130, 30.7106705
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8426056, 40.8438950
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1875153, 67.1912231
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3077545, 46.3126068
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8290710, 53.8302841
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0694427, 58.0697708
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5413208, 66.5421448
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3952026, 74.3960876
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8450012, 41.8464050
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5544128, 46.5551682
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1372070, 39.1389847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 812

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 933

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0413120, upper bound: 18.0443675
time: 39.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0415680, upper bound: 18.0441115
time: 43.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2221527, 63.2216492
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1188507, 42.1177597
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5545273, 37.5533447
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7689362, 41.7718048
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2174072, 48.2163315
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2469025, 57.2461700
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0761261, 62.0769348
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0910645, 68.0888367
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4933243, 55.4918594
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2672424, 37.2684708
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3160553, 43.3165131
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1144562, 42.1145477
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7096672, 30.7107086
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8423615, 40.8441391
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1871033, 67.1916046
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3076324, 46.3127289
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8291626, 53.8302002
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0694427, 58.0697861
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5412598, 66.5421906
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3952332, 74.3960571
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8441467, 41.8472519
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5541382, 46.5554352
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1369553, 39.1392136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 943

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 810

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0349873, upper bound: 18.0493021
time: 41.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0364514, upper bound: 18.0478376
time: 36.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2218933, 63.2208862
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1183167, 42.1162415
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5533371, 37.5507660
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7665405, 41.7693291
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2159424, 48.2122345
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2460327, 57.2438812
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0735321, 62.0755463
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0924377, 68.0893555
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4931335, 55.4901428
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2636719, 37.2673340
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3144226, 43.3158646
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1145020, 42.1147766
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7078781, 30.7101364
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8395996, 40.8419418
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1794434, 67.1886139
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2982101, 46.3092117
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8242493, 53.8287964
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0687256, 58.0692596
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5372467, 66.5406342
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3922119, 74.3950195
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8399506, 41.8437195
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5521088, 46.5535583
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1344833, 39.1368599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2043
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1391

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 926

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0153368, upper bound: 18.0392661
time: 39.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0106391, upper bound: 18.0439679
time: 46.42 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 88.15 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0393433, upper bound: 18.0405020
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0452873, upper bound: 18.0366813
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0417927, upper bound: 18.0404335
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0454861, upper bound: 18.0367290
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0486675, upper bound: 18.0343696
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0437251, upper bound: 18.0393118
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0045654, upper bound: 17.9954802
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0045654, upper bound: 17.9954802
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0450175, upper bound: 18.0228725
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0427440, upper bound: 18.0251463
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0480355, upper bound: 18.0422045
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0480355, upper bound: 18.0422045
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0493928, upper bound: 18.0434797
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0494526, upper bound: 18.0434025
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0471327, upper bound: 18.0460979
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0471613, upper bound: 18.0460945
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0311861, upper bound: 18.0376316
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0452490, upper bound: 18.0235655
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0374524, upper bound: 18.0473425
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0397221, upper bound: 18.0450730
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0423306, upper bound: 18.0374285
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0423306, upper bound: 18.0374285
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0337879, upper bound: 18.0463608
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0286905, upper bound: 18.0514580
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0199779, upper bound: 18.0270895
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0184712, upper bound: 18.0273288
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0413120, upper bound: 18.0443675
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0415680, upper bound: 18.0441115
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0349873, upper bound: 18.0493021
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0364514, upper bound: 18.0478376
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0153368, upper bound: 18.0392661
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 88.15
Output dim: 29, lower bound: -18.0106391, upper bound: 18.0439679

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2213745, 63.2200470
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1205597, 42.1211014
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5492401, 37.5492630
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7671967, 41.7664871
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2151184, 48.2159882
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2480316, 57.2480621
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0604401, 62.0606995
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1077881, 68.1094666
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4964142, 55.4974136
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2722778, 37.2719345
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3175354, 43.3169098
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1197815, 42.1198196
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7132874, 30.7130814
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8461456, 40.8457947
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2019043, 67.1996307
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3136139, 46.3131409
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8262177, 53.8264084
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0713806, 58.0709991
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5339508, 66.5325623
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.4055786, 74.4048157
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8338470, 41.8321457
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5578461, 46.5571594
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1260986, 39.1242294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1549

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 941

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0410940, upper bound: 18.0361848
time: 67.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0447878, upper bound: 18.0324683
time: 44.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2220306, 63.2204590
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1213989, 42.1207504
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5516510, 37.5509415
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7441330, 41.7468948
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2186584, 48.2189941
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2475281, 57.2477112
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0506744, 62.0520782
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1080933, 68.1069641
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4978485, 55.4979324
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2718658, 37.2711334
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3141403, 43.3128510
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1168518, 42.1165924
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7139740, 30.7136841
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8193665, 40.8221741
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1714783, 67.1745453
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3015213, 46.3076477
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8128662, 53.8143921
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0661469, 58.0664215
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5256805, 66.5281830
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3903809, 74.3917084
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8164444, 41.8208466
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5363617, 46.5388489
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1326218, 39.1352501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 909

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1478

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0142853, upper bound: 18.0054063
time: 37.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0142503, upper bound: 18.0054411
time: 36.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2163391, 63.2177277
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1202316, 42.1208420
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5461426, 37.5480347
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7627487, 41.7610283
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2153931, 48.2168350
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2443390, 57.2452316
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0535583, 62.0505524
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1024628, 68.1024017
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4982147, 55.4985809
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2745667, 37.2739487
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3180771, 43.3176422
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1169052, 42.1172562
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7142639, 30.7136917
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8365479, 40.8353043
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1958466, 67.1923218
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2945404, 46.2887650
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8161316, 53.8120499
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0680389, 58.0677795
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5411835, 66.5414581
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3963165, 74.3959198
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8495255, 41.8496323
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5522842, 46.5523071
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1328354, 39.1323547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 829

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 813

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0171284, upper bound: 18.0010081
time: 76.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0147716, upper bound: 18.0033583
time: 41.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2168121, 63.2172546
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1202927, 42.1207962
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5463257, 37.5478592
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7627487, 41.7610321
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2153931, 48.2168350
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2444611, 57.2451019
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0533905, 62.0507050
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.1024323, 68.1024475
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4983215, 55.4984589
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2743683, 37.2741547
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3180771, 43.3176422
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1168289, 42.1173401
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7140961, 30.7138596
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8365784, 40.8352737
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1957550, 67.1923981
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2940979, 46.2892227
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8159943, 53.8121948
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0680695, 58.0677490
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5413971, 66.5412445
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3963165, 74.3959045
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8496323, 41.8495255
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5523911, 46.5522079
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1330109, 39.1321793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2042
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 815

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1604

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0374657, upper bound: 18.0391741
time: 32.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0435873, upper bound: 18.0330535
time: 38.51 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 72.74 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 72.74
Output dim: 29, lower bound: -18.0410940, upper bound: 18.0361848
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 72.74
Output dim: 29, lower bound: -18.0447878, upper bound: 18.0324683
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 72.74
Output dim: 29, lower bound: -18.0142853, upper bound: 18.0054063
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 72.74
Output dim: 29, lower bound: -18.0142503, upper bound: 18.0054411
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 72.74
Output dim: 29, lower bound: -18.0171284, upper bound: 18.0010081
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 72.74
Output dim: 29, lower bound: -18.0147716, upper bound: 18.0033583
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 72.74
Output dim: 29, lower bound: -18.0374657, upper bound: 18.0391741
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 72.74
Output dim: 29, lower bound: -18.0435873, upper bound: 18.0330535
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0450175, upper bound: 18.0228725
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0427440, upper bound: 18.0251463
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0480355, upper bound: 18.0422045
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0480355, upper bound: 18.0422045
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0493928, upper bound: 18.0434797
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0494526, upper bound: 18.0434025
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0471327, upper bound: 18.0460979
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0471613, upper bound: 18.0460945
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0452490, upper bound: 18.0235655
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0374524, upper bound: 18.0473425
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0397221, upper bound: 18.0450730
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0423306, upper bound: 18.0374285
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0423306, upper bound: 18.0374285
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0337879, upper bound: 18.0463608
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0286905, upper bound: 18.0514580
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0413120, upper bound: 18.0443675
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0415680, upper bound: 18.0441115
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0349873, upper bound: 18.0493021
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0364514, upper bound: 18.0478376
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 72.74
Output dim: 29, lower bound: -18.0106391, upper bound: 18.0439679

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 50.80 + 3562.33 = 3613.14 seconds
