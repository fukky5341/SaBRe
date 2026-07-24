## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 3600 seconds
Split limit: 100


## IAR start

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
execution time: IAR + RelationalAnalysis = 2.26 + 49.58 = 51.84 seconds
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
time: 31.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0486932, upper bound: 18.0486932
time: 45.84 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 77.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 77.76
Output dim: 29, lower bound: -18.0486932, upper bound: 18.0598346
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 77.76
Output dim: 29, lower bound: -18.0486932, upper bound: 18.0486932

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

Time for backsubstitution: 1.85 seconds

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
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0315306, upper bound: 18.0248515
time: 41.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0136674, upper bound: 18.0427281
time: 42.09 seconds

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

Time for backsubstitution: 1.83 seconds

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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1639

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0427281, upper bound: 18.0136674
time: 41.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0248515, upper bound: 18.0315306
time: 50.35 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 93.77 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 93.77
Output dim: 29, lower bound: -18.0315306, upper bound: 18.0248515
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 93.77
Output dim: 29, lower bound: -18.0136674, upper bound: 18.0427281
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 93.77
Output dim: 29, lower bound: -18.0427281, upper bound: 18.0136674
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 93.77
Output dim: 29, lower bound: -18.0248515, upper bound: 18.0315306

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2174988, 63.2165070
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1237411, 42.1233673
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5557556, 37.5547256
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7553406, 41.7604637
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2277832, 48.2277145
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2514801, 57.2518463
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0617981, 62.0637512
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0905609, 68.0895081
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5030670, 55.5031433
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2840805, 37.2831497
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3284683, 43.3277359
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1182480, 42.1175880
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7181549, 30.7178879
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8262482, 40.8300171
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1967010, 67.1997375
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3041840, 46.3089066
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8296661, 53.8304367
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0671082, 58.0678101
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5479279, 66.5485382
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3989563, 74.3991241
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8575134, 41.8625717
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5435028, 46.5464172
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1291275, 39.1349182

Time for backsubstitution: 1.78 seconds

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

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0193652, upper bound: 18.0241501
time: 42.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0307627, upper bound: 18.0122187
time: 38.73 seconds

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

Time for backsubstitution: 1.84 seconds

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
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0015067, upper bound: 18.0420253
time: 43.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0129031, upper bound: 18.0300787
time: 39.63 seconds

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

Time for backsubstitution: 1.85 seconds

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
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0300787, upper bound: 18.0129031
time: 41.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0420254, upper bound: 18.0015066
time: 39.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.2165070, 63.2174988
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1233749, 42.1237411
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5547180, 37.5557480
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7604675, 41.7553406
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2277222, 48.2277756
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2518463, 57.2514801
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0637512, 62.0617981
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0895081, 68.0905609
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5031433, 55.5030670
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2831421, 37.2840805
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3277359, 43.3284760
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1175919, 42.1182480
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7178802, 30.7181549
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8300171, 40.8262482
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1997375, 67.1967010
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.3088989, 46.3041840
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8304291, 53.8296585
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0678101, 58.0671082
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5485382, 66.5479126
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3991089, 74.3989868
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8625793, 41.8575211
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5464325, 46.5434875
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1349182, 39.1291275

Time for backsubstitution: 1.85 seconds

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

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0122187, upper bound: 18.0307627
time: 57.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0241501, upper bound: 18.0193652
time: 41.64 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 101.04 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 101.04
Output dim: 29, lower bound: -18.0193652, upper bound: 18.0241501
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 101.04
Output dim: 29, lower bound: -18.0307627, upper bound: 18.0122187
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 101.04
Output dim: 29, lower bound: -18.0015067, upper bound: 18.0420253
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 101.04
Output dim: 29, lower bound: -18.0129031, upper bound: 18.0300787
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 101.04
Output dim: 29, lower bound: -18.0300787, upper bound: 18.0129031
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 101.04
Output dim: 29, lower bound: -18.0420254, upper bound: 18.0015066
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 101.04
Output dim: 29, lower bound: -18.0122187, upper bound: 18.0307627
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 101.04
Output dim: 29, lower bound: -18.0241501, upper bound: 18.0193652

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1956482, 63.1927032
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1232300, 42.1228409
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5454254, 37.5430832
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7068710, 41.7137756
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2308197, 48.2308044
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2522278, 57.2525101
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0367889, 62.0416412
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0909729, 68.0900116
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5064545, 55.5070572
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2903137, 37.2888718
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3330688, 43.3318787
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1205750, 42.1196709
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7193756, 30.7190018
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8007965, 40.8069382
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1917267, 67.1949005
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2686920, 46.2767334
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8207245, 53.8224869
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0651703, 58.0659485
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5554810, 66.5553589
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3956299, 74.3955688
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8487091, 41.8552856
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5343857, 46.5379944
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1040115, 39.1103096

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1674

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0193486, upper bound: 18.0111951
time: 35.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0064403, upper bound: 18.0241321
time: 38.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1936951, 63.1948242
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1232147, 42.1228485
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5441132, 37.5444031
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7085953, 41.7119904
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2308502, 48.2307587
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2521362, 57.2525940
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0397034, 62.0387421
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0910492, 68.0899048
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5069885, 55.5065308
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2898102, 37.2893906
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3326111, 43.3323059
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1203308, 42.1198845
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7192688, 30.7191086
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8029327, 40.8045731
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1918945, 67.1947784
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2720184, 46.2734146
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8217010, 53.8215179
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0652313, 58.0658722
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5547485, 66.5560913
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3954163, 74.3957520
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8500214, 41.8537674
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5348434, 46.5373230
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1046066, 39.1098022

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1674

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0307462, upper bound: 17.9992816
time: 39.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0178214, upper bound: 18.0122011
time: 33.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1954956, 63.1928558
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1230621, 42.1230011
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5451508, 37.5433655
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7120743, 41.7085648
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2308350, 48.2307892
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2525940, 57.2521439
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0371704, 62.0412598
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0899353, 68.0910339
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5064240, 55.5070801
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2896576, 37.2895279
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3329773, 43.3319778
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1199799, 42.1202545
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7191467, 30.7192230
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8035431, 40.8041916
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1940155, 67.1926117
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2702026, 46.2752304
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8202972, 53.8229370
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0658722, 58.0652618
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5560455, 66.5547943
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3956604, 74.3955231
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8525543, 41.8514557
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5368881, 46.5354919
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1096039, 39.1047173

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1674

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0014902, upper bound: 18.0290739
time: 37.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9885897, upper bound: 18.0420077
time: 54.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1935425, 63.1949768
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1230621, 42.1230087
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5438232, 37.5446930
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7138062, 41.7067795
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2308655, 48.2307434
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2525177, 57.2522278
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0400696, 62.0383606
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0900269, 68.0909424
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5069580, 55.5065460
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2891617, 37.2900467
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3325195, 43.3324051
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1197357, 42.1204681
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7190475, 30.7193298
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8056793, 40.8018265
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1941833, 67.1924896
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2735291, 46.2719116
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8212738, 53.8219681
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0659332, 58.0651855
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5553131, 66.5555267
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3954773, 74.3957062
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8538513, 41.8499298
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5373459, 46.5348206
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1101990, 39.1042099

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1674

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0128858, upper bound: 18.0171351
time: 49.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9999696, upper bound: 18.0300619
time: 540.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1949921, 63.1935425
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1230164, 42.1230545
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5446930, 37.5438232
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7067795, 41.7138062
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2307434, 48.2308655
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2522278, 57.2525177
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0383606, 62.0400696
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0909424, 68.0900269
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5065460, 55.5069580
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2900467, 37.2891617
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3323975, 43.3325195
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1204681, 42.1197433
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7193298, 30.7190475
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8018341, 40.8056717
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1924744, 67.1941833
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2719116, 46.2735214
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8219757, 53.8212585
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0651855, 58.0659332
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5555267, 66.5553131
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3956909, 74.3954773
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8499298, 41.8538437
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5348129, 46.5373459
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1042099, 39.1101990

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1674

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0300619, upper bound: 17.9999696
time: 40.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0171351, upper bound: 18.0128857
time: 36.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1928558, 63.1954956
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1230011, 42.1230621
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5433655, 37.5451508
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7085648, 41.7120781
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2307892, 48.2308273
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2521362, 57.2525940
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0412598, 62.0371704
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0910339, 68.0899353
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5070801, 55.5064316
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2895279, 37.2896576
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3319702, 43.3329697
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1202545, 42.1199875
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7192230, 30.7191505
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8041840, 40.8035507
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1926117, 67.1940155
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2752228, 46.2702026
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8229218, 53.8202896
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0652618, 58.0658722
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5547943, 66.5560455
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3955078, 74.3956757
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8514557, 41.8525467
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5354843, 46.5368958
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1047134, 39.1096039

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1674

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0420077, upper bound: 17.9885897
time: 34.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0290739, upper bound: 18.0014902
time: 44.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1948395, 63.1936951
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1228485, 42.1232147
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5444183, 37.5440979
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7119904, 41.7085953
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2307587, 48.2308502
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2525940, 57.2521439
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0387421, 62.0396881
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0899048, 68.0910492
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5065308, 55.5069809
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2893982, 37.2898102
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3323059, 43.3326187
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1198883, 42.1203308
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7191086, 30.7192688
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8045807, 40.8029251
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1947784, 67.1918945
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2734070, 46.2720184
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8215179, 53.8217087
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0658722, 58.0652313
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5560913, 66.5547485
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3957520, 74.3954315
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8537598, 41.8500137
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5373154, 46.5348434
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1098022, 39.1046066

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1674

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0122012, upper bound: 18.0178214
time: 35.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9992816, upper bound: 18.0307461
time: 43.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1927032, 63.1956482
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1228485, 42.1232224
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5430756, 37.5454254
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.7137756, 41.7068672
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2308044, 48.2308197
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2525177, 57.2522278
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0416412, 62.0367889
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0900116, 68.0909729
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5070496, 55.5064545
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2888718, 37.2903061
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3318787, 43.3330688
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1196747, 42.1205750
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7190018, 30.7193718
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8069305, 40.8008041
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1949005, 67.1917267
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2767334, 46.2686920
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8224945, 53.8207397
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0659485, 58.0651703
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5553589, 66.5554810
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3955688, 74.3956146
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8552856, 41.8487167
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5379868, 46.5343933
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1103134, 39.1040115

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1674

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0241321, upper bound: 18.0064403
time: 39.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0111951, upper bound: 18.0193486
time: 43.90 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 85.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -18.0193486, upper bound: 18.0111951
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -18.0064403, upper bound: 18.0241321
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -18.0307462, upper bound: 17.9992816
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -18.0178214, upper bound: 18.0122011
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -18.0014902, upper bound: 18.0290739
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -17.9885897, upper bound: 18.0420077
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -18.0128858, upper bound: 18.0171351
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -17.9999696, upper bound: 18.0300619
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -18.0300619, upper bound: 17.9999696
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -18.0171351, upper bound: 18.0128857
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -18.0420077, upper bound: 17.9885897
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -18.0290739, upper bound: 18.0014902
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -18.0122012, upper bound: 18.0178214
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -17.9992816, upper bound: 18.0307461
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -18.0241321, upper bound: 18.0064403
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 85.85
Output dim: 29, lower bound: -18.0111951, upper bound: 18.0193486

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1741333, 63.1740112
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1292419, 42.1278152
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5481720, 37.5455704
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6924362, 41.6994247
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2468414, 48.2443695
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2566681, 57.2565918
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0383453, 62.0433655
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0842590, 68.0822144
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5046844, 55.5045166
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2904434, 37.2895660
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3363647, 43.3353806
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1151657, 42.1134262
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7207031, 30.7204056
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8005066, 40.8067551
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1684723, 67.1773987
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2277298, 46.2407074
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8106537, 53.8153839
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0655060, 58.0669250
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5772400, 66.5827332
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3912659, 74.3945007
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8449173, 41.8529587
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5317307, 46.5371628
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1117172, 39.1201439

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0119298, upper bound: 18.0100868
time: 43.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0182546, upper bound: 18.0037890
time: 41.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1769409, 63.1712036
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1282043, 42.1288605
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5479279, 37.5458374
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6925125, 41.6993408
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2444000, 48.2468109
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2563019, 57.2569504
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0385132, 62.0431976
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0831604, 68.0833130
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5039062, 55.5052948
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2910080, 37.2890091
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3365784, 43.3351669
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1143265, 42.1142731
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7207794, 30.7203293
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8006287, 40.8066406
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1742249, 67.1716461
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2326584, 46.2357635
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8136292, 53.8124008
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0661469, 58.0662842
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5828400, 66.5771179
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3945618, 74.3912354
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8463821, 41.8514938
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5335617, 46.5353394
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1138458, 39.1180115

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9990071, upper bound: 18.0230192
time: 60.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0053390, upper bound: 18.0167414
time: 44.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1721954, 63.1761169
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1292419, 42.1278229
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5468597, 37.5468979
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6941605, 41.6976395
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2468719, 48.2443237
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2565765, 57.2566757
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0412445, 62.0404510
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0843506, 68.0821228
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5052185, 55.5039902
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2899475, 37.2900925
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3359070, 43.3358154
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1149216, 42.1136398
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7206039, 30.7205162
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8026276, 40.8043900
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1686401, 67.1772614
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2310410, 46.2373810
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8116302, 53.8144073
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0655670, 58.0668488
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5765076, 66.5834656
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3910828, 74.3946991
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8462296, 41.8514328
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5321884, 46.5364914
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1123123, 39.1196365

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0233299, upper bound: 17.9981709
time: 69.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0296439, upper bound: 17.9918698
time: 40.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1749878, 63.1733246
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1282043, 42.1288681
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5465851, 37.5471649
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6942444, 41.6975555
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2444305, 48.2467651
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2562256, 57.2570343
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0414124, 62.0402985
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0832520, 68.0832214
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5044403, 55.5047607
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2905045, 37.2895355
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3361206, 43.3355942
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1140823, 42.1144867
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7206802, 30.7204361
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8027496, 40.8042755
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1743927, 67.1715240
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2359848, 46.2324448
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8146057, 53.8114319
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0662079, 58.0662079
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5821075, 66.5778503
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3943787, 74.3914185
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8476944, 41.8499680
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5340195, 46.5346603
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1144409, 39.1175041

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9925547, upper bound: 18.0110955
time: 43.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0167110, upper bound: 18.0047935
time: 53.20 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1739807, 63.1741638
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1290894, 42.1279755
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5478973, 37.5458527
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6976471, 41.6942139
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2468414, 48.2443542
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2570343, 57.2562256
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0387268, 62.0429840
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0832367, 68.0832520
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5046692, 55.5045395
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2897949, 37.2902222
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3362579, 43.3354797
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1145859, 42.1140137
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7204819, 30.7206306
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8032532, 40.8040085
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1707611, 67.1751099
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2292404, 46.2391968
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8101959, 53.8158264
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0662079, 58.0662231
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5778046, 66.5821686
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3913269, 74.3944550
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8487473, 41.8491287
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5342331, 46.5346603
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1173096, 39.1145515

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9940717, upper bound: 18.0279607
time: 65.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0003976, upper bound: 18.0216663
time: 35.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1767883, 63.1713562
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1280518, 42.1290207
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5476379, 37.5461197
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6977234, 41.6941299
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2444000, 48.2467957
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2566681, 57.2565765
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0388947, 62.0428162
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0821381, 68.0843506
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5038910, 55.5053177
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2903519, 37.2896652
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3364716, 43.3352661
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1137466, 42.1148605
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7205582, 30.7205505
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8033752, 40.8038940
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1765137, 67.1693573
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2341843, 46.2342606
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8131866, 53.8128510
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0668488, 58.0655975
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5834198, 66.5765533
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3946228, 74.3911896
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8502121, 41.8476562
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5360641, 46.5328369
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1194382, 39.1124229

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9811604, upper bound: 18.0408953
time: 37.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9874928, upper bound: 18.0346092
time: 18.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1720428, 63.1762695
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1290894, 42.1279831
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5465851, 37.5471802
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6993713, 41.6924286
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2468719, 48.2443085
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2569427, 57.2563019
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0416260, 62.0400848
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0833130, 68.0831451
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5051880, 55.5040131
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2892990, 37.2907410
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3358002, 43.3359070
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1143417, 42.1142273
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7203751, 30.7207375
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8053741, 40.8016434
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1709290, 67.1749725
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2325516, 46.2358780
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8111725, 53.8148575
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0662689, 58.0661469
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5770721, 66.5828857
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3911438, 74.3946533
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8500443, 41.8476028
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5346909, 46.5339890
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1179047, 39.1140442

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0054766, upper bound: 18.0160243
time: 42.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0117813, upper bound: 18.0097203
time: 34.32 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1748352, 63.1734772
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1280212, 42.1290283
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5463104, 37.5474472
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6994553, 41.6923447
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2444305, 48.2467575
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2565918, 57.2566605
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0417938, 62.0399170
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0822144, 68.0842438
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5044250, 55.5047836
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2898483, 37.2901840
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3360138, 43.3356934
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1135025, 42.1150742
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7204590, 30.7206612
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8054962, 40.8015289
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1766815, 67.1692352
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2374954, 46.2309418
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8141479, 53.8118820
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0668945, 58.0655212
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5826874, 66.5772858
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3944092, 74.3913879
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8515091, 41.8461380
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5365219, 46.5321579
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1200333, 39.1119156

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9925547, upper bound: 18.0289576
time: 43.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9988590, upper bound: 18.0226487
time: 45.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1734772, 63.1748352
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1290283, 42.1280289
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5474396, 37.5463104
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6923447, 41.6994553
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2467499, 48.2444305
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2566681, 57.2565918
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0399170, 62.0417938
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0842438, 68.0822296
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5047760, 55.5044250
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2901840, 37.2898560
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3356934, 43.3360214
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1150742, 42.1134987
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7206650, 30.7204590
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8015289, 40.8054962
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1692352, 67.1766815
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2309494, 46.2374954
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8118744, 53.8141479
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0655212, 58.0668945
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5772858, 66.5826874
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3913879, 74.3944092
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8461380, 41.8515167
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5321579, 46.5365143
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1119156, 39.1200333

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0226487, upper bound: 17.9988590
time: 42.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0289576, upper bound: 17.9925547
time: 47.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1762848, 63.1720276
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1279907, 42.1290741
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5471802, 37.5465775
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6924286, 41.6993713
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2443085, 48.2468719
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2563019, 57.2569504
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0400848, 62.0416260
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0831451, 68.0833282
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5040131, 55.5051956
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2907410, 37.2892990
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3359070, 43.3358078
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1142197, 42.1143417
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7207413, 30.7203789
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8016357, 40.8053741
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1749725, 67.1709290
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2358780, 46.2325516
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8148651, 53.8111725
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0661469, 58.0662689
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5828857, 66.5770721
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3946533, 74.3911438
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8476028, 41.8500519
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5339890, 46.5346909
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1140442, 39.1179008

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0097203, upper bound: 18.0117812
time: 43.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0160243, upper bound: 18.0054766
time: 41.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1713562, 63.1767883
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1290283, 42.1280365
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5461273, 37.5476379
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6941299, 41.6977234
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2468109, 48.2444000
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2565765, 57.2566757
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0428162, 62.0388947
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0843353, 68.0821533
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5053101, 55.5038910
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2896652, 37.2903519
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3352661, 43.3364716
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1148605, 42.1137428
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7205505, 30.7205582
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8038940, 40.8033676
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1693573, 67.1765137
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2342606, 46.2341766
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8128510, 53.8131790
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0655975, 58.0668488
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5765533, 66.5834198
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3912048, 74.3946075
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8476639, 41.8502121
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5328293, 46.5360641
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1124268, 39.1194382

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0346092, upper bound: 17.9874928
time: 40.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0408953, upper bound: 17.9811603
time: 45.26 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1741638, 63.1739807
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1279907, 42.1290817
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5458527, 37.5479050
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6942139, 41.6976471
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2443695, 48.2468414
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2562256, 57.2570343
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0429840, 62.0387268
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0832367, 68.0832520
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5045319, 55.5046692
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2902145, 37.2897949
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3354797, 43.3362579
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1140060, 42.1145897
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7206345, 30.7204819
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8040161, 40.8032532
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1751099, 67.1707611
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2391891, 46.2292328
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8158264, 53.8102036
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0662231, 58.0662079
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5821686, 66.5778046
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3944397, 74.3913422
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8491287, 41.8487473
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5346603, 46.5342407
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1145477, 39.1173096

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0216663, upper bound: 18.0003976
time: 33.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0279607, upper bound: 17.9940717
time: 39.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1733246, 63.1749878
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1288757, 42.1281891
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5471649, 37.5465927
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6975555, 41.6942444
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2467804, 48.2444153
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2570343, 57.2562256
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0402985, 62.0414124
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0832214, 68.0832520
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5047607, 55.5044403
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2895279, 37.2905045
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3356018, 43.3361206
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1144943, 42.1140823
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7204361, 30.7206802
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8042755, 40.8027496
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1715240, 67.1743927
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2324448, 46.2359848
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8114319, 53.8146057
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0662079, 58.0662079
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5778503, 66.5821228
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3914185, 74.3943634
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8499680, 41.8476868
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5346603, 46.5340118
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1175079, 39.1144409

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0047935, upper bound: 18.0167110
time: 32.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0110955, upper bound: 18.0103985
time: 47.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1761322, 63.1721802
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1278381, 42.1292343
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5468903, 37.5468521
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6976395, 41.6941605
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2443390, 48.2468567
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2566681, 57.2565765
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0404510, 62.0412598
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0821228, 68.0843506
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5039825, 55.5052185
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2900925, 37.2899475
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3358154, 43.3359070
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1136398, 42.1149292
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7205124, 30.7206039
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8043823, 40.8026276
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1772614, 67.1686401
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2373886, 46.2310486
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8144073, 53.8116226
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0668488, 58.0655670
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5834503, 66.5765076
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3946838, 74.3910980
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8514328, 41.8462143
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5364914, 46.5321884
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1196365, 39.1123123

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9918698, upper bound: 18.0296439
time: 35.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9981709, upper bound: 18.0233298
time: 41.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1712036, 63.1769409
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1288757, 42.1281967
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5458374, 37.5479202
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6993408, 41.6925125
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2468109, 48.2443848
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2569427, 57.2563019
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0431976, 62.0385132
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0833130, 68.0831757
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5052948, 55.5039139
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2890091, 37.2910080
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3351593, 43.3365707
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1142807, 42.1143265
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7203293, 30.7207794
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8066406, 40.8006210
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1716461, 67.1742249
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2357712, 46.2326660
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8123932, 53.8136292
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0662842, 58.0661469
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5771179, 66.5828552
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3912354, 74.3945618
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8514786, 41.8463821
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5353317, 46.5335617
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1180115, 39.1138458

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
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
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0167415, upper bound: 18.0053390
time: 45.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0230193, upper bound: 17.9990071
time: 57.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1740112, 63.1741333
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1278076, 42.1292496
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5455780, 37.5481796
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6994247, 41.6924362
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2443695, 48.2468262
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2565918, 57.2566605
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0433502, 62.0383453
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0822144, 68.0842743
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.5045166, 55.5046844
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2895660, 37.2904510
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3353729, 43.3363571
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1134262, 42.1151733
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7204056, 30.7207031
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.8067627, 40.8005066
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1773987, 67.1684723
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2407150, 46.2277222
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8153839, 53.8106537
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0669250, 58.0655060
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5827332, 66.5772400
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3945007, 74.3912964
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8529587, 41.8449173
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5371628, 46.5317383
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1201477, 39.1117172

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1620

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0037891, upper bound: 18.0182546
time: 42.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0100869, upper bound: 18.0119298
time: 46.03 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 90.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0119298, upper bound: 18.0100868
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0182546, upper bound: 18.0037890
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -17.9990071, upper bound: 18.0230192
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0053390, upper bound: 18.0167414
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0233299, upper bound: 17.9981709
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0296439, upper bound: 17.9918698
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -17.9925547, upper bound: 18.0110955
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0167110, upper bound: 18.0047935
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -17.9940717, upper bound: 18.0279607
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0003976, upper bound: 18.0216663
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -17.9811604, upper bound: 18.0408953
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -17.9874928, upper bound: 18.0346092
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0054766, upper bound: 18.0160243
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0117813, upper bound: 18.0097203
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -17.9925547, upper bound: 18.0289576
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -17.9988590, upper bound: 18.0226487
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0226487, upper bound: 17.9988590
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0289576, upper bound: 17.9925547
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0097203, upper bound: 18.0117812
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0160243, upper bound: 18.0054766
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0346092, upper bound: 17.9874928
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0408953, upper bound: 17.9811603
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0216663, upper bound: 18.0003976
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0279607, upper bound: 17.9940717
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0047935, upper bound: 18.0167110
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0110955, upper bound: 18.0103985
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -17.9918698, upper bound: 18.0296439
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -17.9981709, upper bound: 18.0233298
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0167415, upper bound: 18.0053390
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0230193, upper bound: 17.9990071
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0037891, upper bound: 18.0182546
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 90.83
Output dim: 29, lower bound: -18.0100869, upper bound: 18.0119298

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1536560, 63.1509094
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1266174, 42.1249542
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5389709, 37.5353928
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6526871, 41.6642838
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2489624, 48.2468109
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2534027, 57.2533417
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0299530, 62.0356445
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0785675, 68.0761261
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4991760, 55.4987869
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2847595, 37.2840118
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3357849, 43.3347931
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1115341, 42.1099319
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7137985, 30.7139893
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.7832031, 40.7914047
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1637268, 67.1732635
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2046051, 46.2197037
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8110962, 53.8157730
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0655670, 58.0670013
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5842285, 66.5890808
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3876038, 74.3904724
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8302155, 41.8398895
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5312347, 46.5366821
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1041641, 39.1130829

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1619

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0042196, upper bound: 18.0096158
time: 56.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0114530, upper bound: 18.0025361
time: 43.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1510468, 63.1535187
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1263733, 42.1251755
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5379944, 37.5363693
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6572952, 41.6596756
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2492676, 48.2464905
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2534180, 57.2533264
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0306396, 62.0349579
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0781860, 68.0765228
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4989471, 55.4990158
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2848892, 37.2838745
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3357849, 43.3348007
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1116714, 42.1097832
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7142868, 30.7135010
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.7851562, 40.7894516
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1643372, 67.1726532
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2067261, 46.2175903
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8110504, 53.8158264
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0655823, 58.0669861
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5835876, 66.5897217
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3872681, 74.3908234
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8318481, 41.8382568
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5312500, 46.5366592
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1046524, 39.1125946

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1619

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0106961, upper bound: 18.0033136
time: 26.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0177796, upper bound: 17.9960889
time: 36.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1564484, 63.1481171
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1255493, 42.1259995
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5387115, 37.5356522
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6527634, 41.6642036
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2465210, 48.2492523
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2530365, 57.2537003
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0301208, 62.0354767
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0774689, 68.0772247
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4984131, 55.4995575
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2853165, 37.2834473
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3359985, 43.3345795
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1106949, 42.1107788
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7138748, 30.7139130
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.7833252, 40.7912903
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1694794, 67.1675262
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2095490, 46.2147675
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8140717, 53.8127899
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0662079, 58.0663605
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5898438, 66.5834808
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3908691, 74.3872070
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8316803, 41.8384247
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5330658, 46.5348511
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1063004, 39.1109505

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1619

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9912843, upper bound: 18.0225507
time: 41.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9985236, upper bound: 18.0154785
time: 45.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1538544, 63.1507111
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1253357, 42.1262207
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5377350, 37.5366287
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6573715, 41.6595917
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2468262, 48.2489319
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2530518, 57.2536850
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0308075, 62.0347900
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0770874, 68.0776215
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4981689, 55.4997940
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2854462, 37.2833252
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3359985, 43.3345795
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1108322, 42.1106300
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7143631, 30.7134247
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.7852783, 40.7893372
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1700897, 67.1669006
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2116547, 46.2126465
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8140259, 53.8128510
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0662079, 58.0663452
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5892029, 66.5841217
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3905029, 74.3875580
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8333130, 41.8367920
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5330811, 46.5348358
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1067886, 39.1104660

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
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

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 1619

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9977861, upper bound: 18.0162697
time: 44.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0048667, upper bound: 18.0090471
time: 38.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -38.5503197, 25.2954807, -38.5503197, 25.2954807, -63.1517029, 63.1530457
1: -21.3310833, 21.3358307, -21.3310833, 21.3358307, -42.1265869, 42.1249619
2: -15.3783474, 23.6382294, -15.3783474, 23.6382294, -37.5376587, 37.5367203
3: -28.4871235, 18.8754463, -28.4871235, 18.8754463, -47.3625717, 47.3625717
4: -26.2838211, 22.7261581, -26.2838211, 22.7261581, -49.0099792, 49.0099792
5: -24.1822739, 24.2745323, -24.1822739, 24.2745323, -48.4568062, 48.4568062
6: -53.7474899, -5.1680412, -53.7474899, -5.1680412, -41.6544113, 41.6624985
7: -34.4538307, 15.3233147, -34.4538307, 15.3233147, -48.2489929, 48.2467651
8: -40.0405388, 17.2362328, -40.0405388, 17.2362328, -57.2533112, 57.2534256
9: -23.2319984, 25.8786621, -23.2319984, 25.8786621, -49.1106606, 49.1106606
10: -36.0579910, 24.9901237, -36.0579910, 24.9901237, -61.0481148, 61.0481148
11: -27.1906166, 14.1108942, -27.1906166, 14.1108942, -41.3015099, 41.3015099
12: -45.4676971, 18.8344555, -45.4676971, 18.8344555, -62.0328522, 62.0327454
13: -37.8403816, 26.0743637, -37.8403816, 26.0743637, -63.9147453, 63.9147453
14: -52.1818428, 16.0784225, -52.1818428, 16.0784225, -68.0786438, 68.0760345
15: -21.6753368, 29.1490746, -21.6753368, 29.1490746, -50.8244095, 50.8244095
16: -35.7352524, 20.8166046, -35.7352524, 20.8166046, -55.4997101, 55.4982605
17: -48.7089615, 12.4807968, -48.7089615, 12.4807968, -61.1897583, 61.1897583
18: -22.0064011, 30.1203804, -22.0064011, 30.1203804, -52.1267815, 52.1267815
19: -17.5853329, 19.9722385, -17.5853329, 19.9722385, -37.5575714, 37.5575714
20: -24.8853493, 14.0736284, -24.8853493, 14.0736284, -38.9589767, 38.9589767
21: -22.7633686, 21.9509792, -22.7633686, 21.9509792, -44.7143478, 44.7143478
22: -13.9427128, 27.4258881, -13.9427128, 27.4258881, -37.2842636, 37.2845306
23: -16.3272686, 22.9324493, -16.3272686, 22.9324493, -39.2597198, 39.2597198
24: -16.5055256, 23.8845997, -16.5055256, 23.8845997, -40.3901253, 40.3901253
25: -16.7627220, 28.0365276, -16.7627220, 28.0365276, -43.3353271, 43.3352203
26: -27.1851788, 35.4457932, -27.1851788, 35.4457932, -62.6309738, 62.6309738
27: -22.0505695, 19.6049366, -22.0505695, 19.6049366, -41.6555061, 41.6555061
28: -17.2227802, 25.1291656, -17.2227802, 25.1291656, -42.1112900, 42.1101456
29: -9.9339752, 22.7379303, -9.9339752, 22.7379303, -30.7136917, 30.7140961
30: -29.0362720, 15.6439238, -29.0362720, 15.6439238, -44.6801949, 44.6801949
31: -23.9211502, 24.2438602, -23.9211502, 24.2438602, -48.1650085, 48.1650085
32: -47.7756042, -1.8755980, -47.7756042, -1.8755980, -40.7853241, 40.7890396
33: -70.7135696, 5.5393944, -70.7135696, 5.5393944, -67.1639099, 67.1731415
34: -63.6793327, -5.5997620, -63.6793327, -5.5997620, -46.2079315, 46.2163849
35: -41.9776802, 14.3460732, -41.9776802, 14.3460732, -53.8120728, 53.8148041
36: -42.4531479, 16.1393471, -42.4531479, 16.1393471, -58.0656281, 58.0669250
37: -71.0562134, -1.2145185, -71.0562134, -1.2145185, -66.5834961, 66.5898132
38: -59.6414642, 8.7939987, -59.6414642, 8.7939987, -68.4354630, 68.4354630
39: -74.2928619, 1.1102567, -74.2928619, 1.1102567, -74.3874207, 74.3906708
40: -70.9852295, -18.7158470, -70.9852295, -18.7158470, -41.8315125, 41.8383636
41: -47.8816223, 3.7671366, -47.8816223, 3.7671366, -46.5316925, 46.5360031
42: -46.9685440, -3.8184218, -46.9685440, -3.8184218, -39.1047592, 39.1125717

Time for backsubstitution: 1.89 seconds

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 51.84 + 3550.02 = 3601.86 seconds
