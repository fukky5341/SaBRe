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
execution time: IAR + RelationalAnalysis = 2.72 + 47.20 = 49.92 seconds
status: Status.UNKNOWN
relational distance
Output dim: 29, lower bound: -18.0602283, upper bound: 18.0602283

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1382

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1768

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0486932, upper bound: 18.0598346
time: 30.46 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0598346, upper bound: 18.0486932
time: 40.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 71.33 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 71.33
Output dim: 29, lower bound: -18.0486932, upper bound: 18.0598346
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 71.33
Output dim: 29, lower bound: -18.0598346, upper bound: 18.0486932

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2183075, 63.2174683
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1239166, 42.1237030
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5560303, 37.5552902
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7624054, 41.7623177
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2278442, 48.2277756
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2519226, 57.2519226
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0628052, 62.0643616
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0931702, 68.0931549
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5032196, 55.5033188
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2845993, 37.2843094
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3284988, 43.3278656
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1191864, 42.1191101
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7185745, 30.7185249
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8327789, 40.8338013
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2024689, 67.2032166
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3089294, 46.3121490
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8298340, 53.8310699
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0686035, 58.0686188
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5499115, 66.5499573
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3993225, 74.3994293
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8661652, 41.8673859
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5486603, 46.5490952
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1334915, 39.1336899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1382

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1639

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0315306, upper bound: 18.0248515
time: 39.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0136674, upper bound: 18.0427281
time: 40.33 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2174683, 63.2183075
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1237030, 42.1239166
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5552826, 37.5560303
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7623138, 41.7624054
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2277832, 48.2278519
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2519226, 57.2519226
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0643616, 62.0628052
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0931396, 68.0931702
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5033264, 55.5032196
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2843094, 37.2845993
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3278580, 43.3285065
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1191101, 42.1191826
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7185287, 30.7185745
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8338013, 40.8327789
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.2032166, 67.2024689
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3121490, 46.3089371
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8310852, 53.8298416
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0686188, 58.0686035
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5499573, 66.5499115
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3994446, 74.3993378
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8673706, 41.8661652
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5490875, 46.5486679
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1336899, 39.1334915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1382

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1639

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0427281, upper bound: 18.0136674
time: 39.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0248515, upper bound: 18.0315306
time: 48.14 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 90.09 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 90.09
Output dim: 29, lower bound: -18.0315306, upper bound: 18.0248515
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 90.09
Output dim: 29, lower bound: -18.0136674, upper bound: 18.0427281
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 90.09
Output dim: 29, lower bound: -18.0427281, upper bound: 18.0136674
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 90.09
Output dim: 29, lower bound: -18.0248515, upper bound: 18.0315306

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2173462, 63.2166595
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1235886, 42.1235275
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5554810, 37.5550156
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7605515, 41.7552528
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2277985, 48.2277069
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2518463, 57.2514801
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0621796, 62.0633698
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0895233, 68.0905304
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5030365, 55.5031586
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2834320, 37.2837982
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3283768, 43.3278351
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1176529, 42.1181717
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7179337, 30.7181091
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8289948, 40.8272705
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1989899, 67.1974487
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3056946, 46.3074036
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8292084, 53.8308792
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0678101, 58.0671082
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5484924, 66.5479736
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3990173, 74.3990784
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8613586, 41.8587418
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5460052, 46.5439148
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1347198, 39.1293259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1382

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0015067, upper bound: 18.0420253
time: 42.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0129031, upper bound: 18.0300787
time: 38.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2166595, 63.2173462
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1235275, 42.1235809
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5550232, 37.5554733
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7552490, 41.7605515
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2277069, 48.2277908
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2514801, 57.2518463
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0633698, 62.0621796
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0905304, 68.0895386
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5031586, 55.5030441
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2837982, 37.2834320
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3278275, 43.3283768
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1181717, 42.1176605
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7181091, 30.7179337
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8272705, 40.8289948
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1974487, 67.1989899
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3074036, 46.3056946
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8308868, 53.8292084
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0671082, 58.0677948
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5479736, 66.5484924
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3990784, 74.3990326
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8587341, 41.8613510
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5439301, 46.5459900
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1293259, 39.1347198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=257, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1341
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1512
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 906
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1335
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1513
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 1307
type: RSZ, layer: 1, pos: 1511
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1332
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 927
type: RSZ, layer: 1, pos: 1351
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1309
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 812
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1683
type: RSZ, layer: 1, pos: 1373
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 978
type: RSZ, layer: 1, pos: 788
type: RSZ, layer: 1, pos: 1325
type: RSZ, layer: 1, pos: 926
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1375
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1348
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 787
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1622
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1331
type: RSZ, layer: 1, pos: 955
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1458
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 1490
type: RSZ, layer: 1, pos: 1652
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 792
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 987
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1380
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 796
type: RSZ, layer: 1, pos: 849
type: RSZ, layer: 1, pos: 1407
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 1734
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 862
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1668
type: RSZ, layer: 1, pos: 1474
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1023
type: RSZ, layer: 1, pos: 1006
type: RSZ, layer: 1, pos: 959
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 1306
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 1381
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1355
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 793
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 897
type: RSZ, layer: 1, pos: 1510
type: RSZ, layer: 1, pos: 1316
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 930
type: RSZ, layer: 1, pos: 797
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1366
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1445
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 831
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1379
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1478
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 894
type: RSZ, layer: 1, pos: 1338
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1388
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1486
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 813
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1300
type: RSZ, layer: 1, pos: 1320
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1281
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 1363
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 960
type: RSZ, layer: 1, pos: 942
type: RSZ, layer: 1, pos: 844
type: RSZ, layer: 1, pos: 944
type: RSZ, layer: 1, pos: 1477
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1354
type: RSZ, layer: 1, pos: 804
type: RSZ, layer: 1, pos: 1372
type: RSZ, layer: 1, pos: 794
type: RSZ, layer: 1, pos: 913
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1284
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1441
type: RSZ, layer: 1, pos: 910
type: RSZ, layer: 1, pos: 991
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 815
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 783
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 976
type: RSZ, layer: 1, pos: 1412
type: RSZ, layer: 1, pos: 1007
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 943
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1009
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 814
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 975
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1368
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 992
type: RSZ, layer: 1, pos: 1356
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 1443
type: RSZ, layer: 1, pos: 911
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 830
type: RSZ, layer: 1, pos: 775
type: RSZ, layer: 1, pos: 929
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 810
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 816
type: RSZ, layer: 1, pos: 770
type: RSZ, layer: 1, pos: 945
type: RSZ, layer: 1, pos: 784
type: RSZ, layer: 1, pos: 1382

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1767

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0300787, upper bound: 18.0129031
time: 39.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0420254, upper bound: 18.0015066
time: 37.86 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 80.12 seconds
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 80.12
Output dim: 29, lower bound: -18.0015067, upper bound: 18.0420253
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 80.12
Output dim: 29, lower bound: -18.0129031, upper bound: 18.0300787
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 80.12
Output dim: 29, lower bound: -18.0300787, upper bound: 18.0129031
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 80.12
Output dim: 29, lower bound: -18.0420254, upper bound: 18.0015066

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 49.92 + 406.57 = 456.49 seconds
