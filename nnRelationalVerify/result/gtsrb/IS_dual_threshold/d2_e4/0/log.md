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
execution time: IAR + RelationalAnalysis = 2.71 + 47.25 = 49.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 29, lower bound: -18.0602283, upper bound: 18.0602283

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1639
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1587
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1595
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1622
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 999
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1580
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1744
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: A, layer: 1, pos: 1788
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 991
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1445
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 1412
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1512
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1372
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 914
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1443
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1636
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1490
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 543
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1652
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 1559
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 1348
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 527
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 1599
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1474
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1008
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1465
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1007
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1523
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1307
type: B, layer: 1, pos: 1307
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1766
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 895
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 1507
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 831
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 1391
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1338
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 1546
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 775
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 783
type: A, layer: 1, pos: 783

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1639

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0252450, upper bound: 18.0431210
time: 47.69 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0431210, upper bound: 18.0431210
time: 34.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 82.56 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 82.56
Output dim: 29, lower bound: -18.0252450, upper bound: 18.0431210
IS_B2, status: Status.UNKNOWN, split count: 1, time: 82.56
Output dim: 29, lower bound: -18.0431210, upper bound: 18.0431210

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -38.5447807, 25.2916927, -38.5385742, 25.2874317, -63.2159882, 63.2138672
1: -21.3280926, 21.3317108, -21.3247147, 21.3272018, -42.1135406, 42.1143799
2: -15.3767195, 23.6348953, -15.3748703, 23.6311932, -37.5520401, 37.5531311
3: -28.4834423, 18.8713226, -28.4793301, 18.8667240, -47.3501663, 47.3506546
4: -26.2765656, 22.7223015, -26.2683697, 22.7180367, -48.9946022, 48.9906693
5: -24.1814823, 24.2698212, -24.1805725, 24.2645149, -48.4459991, 48.4503937
6: -53.7430420, -5.1697340, -53.7379837, -5.1716099, -41.7677231, 41.7644920
7: -34.4518623, 15.3195152, -34.4496880, 15.3152866, -48.2167053, 48.2187805
8: -40.0387306, 17.2291374, -40.0367813, 17.2212582, -57.2340240, 57.2403030
9: -23.2131386, 25.8762131, -23.1917877, 25.8734798, -49.0866165, 49.0680008
10: -36.0527191, 24.9815216, -36.0468597, 24.9720745, -61.0247955, 61.0283813
11: -27.1844368, 14.0870647, -27.1775646, 14.0612011, -41.2456360, 41.2646294
12: -45.4622421, 18.8305683, -45.4562149, 18.8261929, -62.0614777, 62.0605469
13: -37.8202667, 26.0692253, -37.7976494, 26.0633278, -63.8835945, 63.8668747
14: -52.1743660, 16.0671139, -52.1660690, 16.0544891, -68.0632629, 68.0682373
15: -21.6603889, 29.1446953, -21.6437187, 29.1398621, -50.8002510, 50.7884140
16: -35.7266884, 20.8122559, -35.7171173, 20.8074474, -55.4824982, 55.4781570
17: -48.7013206, 12.4585686, -48.6928940, 12.4333820, -61.1347046, 61.1514626
18: -22.0003052, 30.1141338, -21.9935226, 30.1074486, -52.1077538, 52.1076584
19: -17.5800991, 19.9539795, -17.5743179, 19.9332848, -37.5133820, 37.5282974
20: -24.8809433, 14.0673199, -24.8760929, 14.0605106, -38.9414520, 38.9434128
21: -22.7557468, 21.9324818, -22.7472515, 21.9115543, -44.6673012, 44.6797333
22: -13.9377193, 27.4237328, -13.9322071, 27.4214077, -37.2680740, 37.2656555
23: -16.3229446, 22.9161873, -16.3181458, 22.8979950, -39.2209396, 39.2343330
24: -16.5007668, 23.8734035, -16.4955196, 23.8607407, -40.3615074, 40.3689232
25: -16.7571850, 28.0228157, -16.7511101, 28.0072250, -43.2852478, 43.2957001
26: -27.1782207, 35.4419479, -27.1705723, 35.4376793, -62.6158981, 62.6125183
27: -22.0459766, 19.6013050, -22.0408192, 19.5972404, -41.6432190, 41.6421242
28: -17.2188244, 25.1202946, -17.2144947, 25.1102257, -42.0948029, 42.1004181
29: -9.9287300, 22.7277031, -9.9229107, 22.7161407, -30.6905746, 30.6962433
30: -29.0302258, 15.6267366, -29.0236111, 15.6072922, -44.6375198, 44.6503487
31: -23.9144917, 24.2228298, -23.9069920, 24.1990376, -48.1135292, 48.1298218
32: -47.7593765, -1.8805642, -47.7409821, -1.8860030, -40.8201981, 40.8082962
33: -70.6923828, 5.5372896, -70.6685333, 5.5348988, -67.1850433, 67.1641846
34: -63.6674232, -5.6042328, -63.6541100, -5.6090522, -46.3156891, 46.3091125
35: -41.9678116, 14.3443766, -41.9569893, 14.3424425, -53.8250885, 53.8158722
36: -42.4368515, 16.1359444, -42.4185524, 16.1321507, -58.0457764, 58.0315704
37: -71.0478973, -1.2159595, -71.0387878, -1.2176418, -66.5388489, 66.5318451
38: -59.6299934, 8.7890072, -59.6169891, 8.7835312, -68.4135284, 68.4059982
39: -74.2676239, 1.1085887, -74.2395859, 1.1066222, -74.3694763, 74.3433990
40: -70.9704742, -18.7177086, -70.9537888, -18.7197552, -41.8522568, 41.8384781
41: -47.8736496, 3.7640963, -47.8647003, 3.7607098, -46.5429077, 46.5376129
42: -46.9655151, -3.8211365, -46.9621964, -3.8241773, -39.1272888, 39.1296082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1639
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: B, layer: 1, pos: 873
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1373
type: A, layer: 1, pos: 1373
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1588
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1555
type: A, layer: 1, pos: 1555
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1368
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1685
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 899
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1762
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1512
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1561
type: B, layer: 1, pos: 865
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: B, layer: 1, pos: 1443
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1571
type: B, layer: 1, pos: 1571
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1000
type: B, layer: 1, pos: 1407
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1382
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1500
type: B, layer: 1, pos: 1500
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1663
type: A, layer: 1, pos: 1663
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1316
type: B, layer: 1, pos: 1316
type: A, layer: 1, pos: 1490
type: B, layer: 1, pos: 1490
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 1363
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 1745
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1567
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1299
type: B, layer: 1, pos: 1299
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1576
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1574
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1300
type: A, layer: 1, pos: 535
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1778
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1371
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1637
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 866
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 854
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 518
type: B, layer: 1, pos: 518
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: B, layer: 1, pos: 1477
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 1474
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1513
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1008
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 849
type: B, layer: 1, pos: 849
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 578
type: A, layer: 1, pos: 578
type: B, layer: 1, pos: 1465
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1594
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1007
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1523
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 809
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1602
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1356
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 898
type: A, layer: 1, pos: 898
type: B, layer: 1, pos: 1612
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 895
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1305
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1570
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1533
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 831
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 831
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 800
type: B, layer: 1, pos: 800
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 955
type: A, layer: 1, pos: 955
type: B, layer: 1, pos: 1540
type: A, layer: 1, pos: 1540
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1586
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1338
type: B, layer: 1, pos: 1338
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 775
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 992
type: A, layer: 1, pos: 992
type: B, layer: 1, pos: 1354
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 792
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 1009
type: B, layer: 1, pos: 1009
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0172567, upper bound: 18.0191636
time: 40.56 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0172567, upper bound: 18.0351270
time: 32.72 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -38.5437126, 25.2937012, -38.5760269, 25.3218117, -63.2505188, 63.2537079
1: -21.3267670, 21.3337765, -21.3537216, 21.3604317, -42.1486282, 42.1460266
2: -15.3749552, 23.6365433, -15.3863258, 23.6579018, -37.5819473, 37.5672684
3: -28.4838142, 18.8731117, -28.4897976, 18.9163971, -47.4002113, 47.3629074
4: -26.2763195, 22.7246170, -26.2831383, 22.7755470, -49.0518646, 49.0077553
5: -24.1817341, 24.2705231, -24.1931801, 24.2987347, -48.4804688, 48.4637032
6: -53.7456818, -5.1688080, -53.7604523, -5.1311903, -41.8011627, 41.7974968
7: -34.4530869, 15.3189220, -34.4934425, 15.3238506, -48.2281036, 48.2630539
8: -40.0399475, 17.2317238, -40.0634689, 17.2554913, -57.2682648, 57.2742844
9: -23.2216835, 25.8773708, -23.2384987, 25.9420471, -49.1637306, 49.1158676
10: -36.0562592, 24.9835377, -36.1159515, 25.0042286, -61.0604858, 61.0994873
11: -27.1885071, 14.1039124, -27.3947792, 14.1120939, -41.3006020, 41.4986916
12: -45.4625282, 18.8330154, -45.4855499, 18.8806515, -62.1126556, 62.0933838
13: -37.8315506, 26.0720482, -37.8468857, 26.1799126, -64.0114594, 63.9189339
14: -52.1781006, 16.0759983, -52.2825203, 16.0806351, -68.0911255, 68.1933289
15: -21.6706963, 29.1479263, -21.6820774, 29.2365417, -50.9072380, 50.8300018
16: -35.7325478, 20.8126316, -35.8206100, 20.8249245, -55.5072250, 55.5854340
17: -48.7058029, 12.4752846, -48.9004135, 12.4836512, -61.1894531, 61.3756981
18: -22.0027809, 30.1165085, -22.0562553, 30.1274071, -52.1301880, 52.1727638
19: -17.5830612, 19.9694901, -17.6989994, 19.9696960, -37.5527573, 37.6684875
20: -24.8835888, 14.0673895, -24.9385319, 14.0741453, -38.9577332, 39.0059204
21: -22.7610607, 21.9453049, -22.9056149, 21.9456596, -44.7067184, 44.8509216
22: -13.9400539, 27.4251328, -13.9878349, 27.4301491, -37.2841187, 37.3196182
23: -16.3256798, 22.9298668, -16.4502316, 22.9369087, -39.2625885, 39.3800964
24: -16.5030937, 23.8791943, -16.6103153, 23.8833065, -40.3863983, 40.4895096
25: -16.7601738, 28.0311527, -16.8542843, 28.0368042, -43.3195801, 43.4078674
26: -27.1810646, 35.4446793, -27.2495728, 35.4598083, -62.6408730, 62.6942520
27: -22.0486031, 19.6017609, -22.1006603, 19.6080875, -41.6566925, 41.7024231
28: -17.2209930, 25.1253891, -17.3139839, 25.1318531, -42.1157379, 42.2029305
29: -9.9312067, 22.7342873, -10.0504389, 22.7393303, -30.7159958, 30.8307495
30: -29.0344238, 15.6347523, -29.1955090, 15.6375456, -44.6719704, 44.8302612
31: -23.9184303, 24.2410069, -24.0402985, 24.2445869, -48.1630173, 48.2813034
32: -47.7712173, -1.8779736, -47.7847824, -1.7706466, -40.9473190, 40.8518219
33: -70.7100601, 5.5377827, -70.7418137, 5.6503973, -67.3173828, 67.2307434
34: -63.6757050, -5.6012573, -63.6929359, -5.5004144, -46.4180298, 46.3449326
35: -41.9744759, 14.3454695, -41.9980507, 14.4152298, -53.9063873, 53.8524551
36: -42.4437790, 16.1383591, -42.4518280, 16.2328434, -58.1529541, 58.0664291
37: -71.0505676, -1.2153358, -71.0821533, -1.1809320, -66.5766144, 66.5787811
38: -59.6305847, 8.7928143, -59.6327362, 8.9005461, -68.5311279, 68.4255524
39: -74.2859955, 1.1093917, -74.3105927, 1.2346520, -74.5189209, 74.4165955
40: -70.9783401, -18.7165833, -70.9939346, -18.6066895, -41.9693909, 41.8778381
41: -47.8778076, 3.7660408, -47.8988724, 3.8248277, -46.6089478, 46.5735779
42: -46.9670486, -3.8197708, -46.9800606, -3.7903385, -39.1442261, 39.1788864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=257, inp2_unstable=256, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1335
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1767
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1674
type: B, layer: 1, pos: 1674
type: A, layer: 1, pos: 1729
type: B, layer: 1, pos: 1729
type: A, layer: 1, pos: 1734
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1591
type: A, layer: 1, pos: 1591
type: B, layer: 1, pos: 1587
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 1486
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1511
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 873
type: B, layer: 1, pos: 1428
type: A, layer: 1, pos: 1428
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1667
type: B, layer: 1, pos: 1667
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1543
type: A, layer: 1, pos: 1543
type: B, layer: 1, pos: 1595
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1622
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1630
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 999
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 893
type: A, layer: 1, pos: 893
type: B, layer: 1, pos: 1646
type: A, layer: 1, pos: 1646
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 1351
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 876
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 978
type: A, layer: 1, pos: 978
type: B, layer: 1, pos: 1309
type: A, layer: 1, pos: 1309
type: B, layer: 1, pos: 1580
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1563
type: B, layer: 1, pos: 1563
type: A, layer: 1, pos: 1366
type: B, layer: 1, pos: 1404
type: A, layer: 1, pos: 1404
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1366
type: A, layer: 1, pos: 987
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1628
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1628
type: B, layer: 1, pos: 1744
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 906
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1788
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1668
type: B, layer: 1, pos: 1668
type: A, layer: 1, pos: 869
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 869
type: A, layer: 1, pos: 1588
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 519
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 519
type: B, layer: 1, pos: 991
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1445
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 1714
type: A, layer: 1, pos: 1714
type: B, layer: 1, pos: 1564
type: A, layer: 1, pos: 1564
type: B, layer: 1, pos: 1381
type: A, layer: 1, pos: 1381
type: B, layer: 1, pos: 1341
type: A, layer: 1, pos: 1412
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: B, layer: 1, pos: 1554
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1768
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1332
type: A, layer: 1, pos: 1332
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 941
type: A, layer: 1, pos: 941
type: B, layer: 1, pos: 1685
type: A, layer: 1, pos: 899
type: B, layer: 1, pos: 892
type: A, layer: 1, pos: 892
type: B, layer: 1, pos: 1558
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1297
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1603
type: A, layer: 1, pos: 1603
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1368
type: A, layer: 1, pos: 1762
type: B, layer: 1, pos: 1420
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 1698
type: B, layer: 1, pos: 1698
type: A, layer: 1, pos: 1325
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 961
type: A, layer: 1, pos: 961
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1618
type: A, layer: 1, pos: 1618
type: B, layer: 1, pos: 1561
type: A, layer: 1, pos: 1372
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 914
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 865
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1340
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 964
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1000
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1545
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1545
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1407
type: B, layer: 1, pos: 1382
type: A, layer: 1, pos: 1620
type: B, layer: 1, pos: 1620
type: A, layer: 1, pos: 975
type: B, layer: 1, pos: 975
type: A, layer: 1, pos: 1682
type: B, layer: 1, pos: 1682
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1636
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1502
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1502
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1490
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1363
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1745
type: B, layer: 1, pos: 1320
type: A, layer: 1, pos: 1320
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 959
type: A, layer: 1, pos: 959
type: B, layer: 1, pos: 1308
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1308
type: B, layer: 1, pos: 1590
type: A, layer: 1, pos: 1590
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 543
type: A, layer: 1, pos: 1435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1652
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 535
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1478
type: A, layer: 1, pos: 1576
type: B, layer: 1, pos: 1388
type: A, layer: 1, pos: 1388
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1478
type: B, layer: 1, pos: 1574
type: A, layer: 1, pos: 1300
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 1559
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 812
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 812
type: A, layer: 1, pos: 1778
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1348
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1331
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1637
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1331
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 1765
type: A, layer: 1, pos: 1765
type: B, layer: 1, pos: 1542
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1542
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1371
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 946
type: A, layer: 1, pos: 946
type: B, layer: 1, pos: 1577
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1577
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 866
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1683
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 797
type: B, layer: 1, pos: 797
type: A, layer: 1, pos: 960
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1375
type: A, layer: 1, pos: 1375
type: B, layer: 1, pos: 1306
type: A, layer: 1, pos: 1306
type: B, layer: 1, pos: 854
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1453
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1604
type: A, layer: 1, pos: 1604
type: B, layer: 1, pos: 1513
type: A, layer: 1, pos: 527
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 884
type: A, layer: 1, pos: 884
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1599
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 851
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1477
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 998
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 998
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 997
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 997
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 794
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 794
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1379
type: B, layer: 1, pos: 1379
type: A, layer: 1, pos: 1023
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 1023
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 944
type: A, layer: 1, pos: 944
type: B, layer: 1, pos: 1610
type: A, layer: 1, pos: 1610
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 881
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1510
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1510
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1594
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 1602
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 809
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1355
type: B, layer: 1, pos: 1355
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1357
type: A, layer: 1, pos: 1356
type: B, layer: 1, pos: 1766
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1357
type: B, layer: 1, pos: 1619
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1583
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1583
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1538
type: B, layer: 1, pos: 1538
type: A, layer: 1, pos: 1612
type: B, layer: 1, pos: 836
type: A, layer: 1, pos: 836
type: B, layer: 1, pos: 1380
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 1305
type: A, layer: 1, pos: 1380
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 1458
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1507
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 894
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 894
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 810
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 1549
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 1570
type: A, layer: 1, pos: 831
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1458
type: A, layer: 1, pos: 1549
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1284
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1284
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1551
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1533
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 945
type: A, layer: 1, pos: 945
type: B, layer: 1, pos: 1441
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1441
type: B, layer: 1, pos: 1006
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1006
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1281
type: B, layer: 1, pos: 1281
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1586
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1647
type: B, layer: 1, pos: 897
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 796
type: B, layer: 1, pos: 796
type: A, layer: 1, pos: 882
type: B, layer: 1, pos: 882
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1338
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1546
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1354
type: B, layer: 1, pos: 792
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1387
type: B, layer: 1, pos: 1387
type: A, layer: 1, pos: 783
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1009

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0351270, upper bound: 18.0191636
time: 29.96 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0351270, upper bound: 18.0351270
time: 45.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 78.21 seconds
IS_B1_B1, status: Status.VERIFIED, split count: 2, time: 78.21
Output dim: 29, lower bound: -18.0172567, upper bound: 18.0191636
IS_B1_B2, status: Status.VERIFIED, split count: 2, time: 78.21
Output dim: 29, lower bound: -18.0172567, upper bound: 18.0351270
IS_B2_B1, status: Status.VERIFIED, split count: 2, time: 78.21
Output dim: 29, lower bound: -18.0351270, upper bound: 18.0191636
IS_B2_B2, status: Status.VERIFIED, split count: 2, time: 78.21
Output dim: 29, lower bound: -18.0351270, upper bound: 18.0351270

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 49.96 + 236.32 = 286.28 seconds
