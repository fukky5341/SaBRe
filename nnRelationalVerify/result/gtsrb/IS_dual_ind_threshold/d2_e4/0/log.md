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
execution time: IAR + RelationalAnalysis = 2.70 + 46.90 = 49.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 29, lower bound: -18.0602283, upper bound: 18.0602283

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 897

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 1639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0431210, upper bound: 18.0252450
time: 41.12 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0431210, upper bound: 18.0431210
time: 50.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 91.85 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 91.85
Output dim: 29, lower bound: -18.0431210, upper bound: 18.0252450
IS_A2, status: Status.UNKNOWN, split count: 1, time: 91.85
Output dim: 29, lower bound: -18.0431210, upper bound: 18.0431210

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -38.5385742, 25.2874317, -38.5447807, 25.2916927, -63.2138672, 63.2159882
1: -21.3247147, 21.3272018, -21.3280926, 21.3317108, -42.1143799, 42.1135330
2: -15.3748703, 23.6311932, -15.3767195, 23.6348953, -37.5531387, 37.5520325
3: -28.4793301, 18.8667240, -28.4834423, 18.8713226, -47.3506546, 47.3501663
4: -26.2683697, 22.7180367, -26.2765656, 22.7223015, -48.9906693, 48.9946022
5: -24.1805725, 24.2645149, -24.1814823, 24.2698212, -48.4503937, 48.4459991
6: -53.7379837, -5.1716099, -53.7430420, -5.1697340, -41.7644882, 41.7677231
7: -34.4496880, 15.3152866, -34.4518623, 15.3195152, -48.2187805, 48.2167130
8: -40.0367813, 17.2212582, -40.0387306, 17.2291374, -57.2403107, 57.2340240
9: -23.1917877, 25.8734798, -23.2131386, 25.8762131, -49.0680008, 49.0866165
10: -36.0468597, 24.9720745, -36.0527191, 24.9815216, -61.0283813, 61.0247955
11: -27.1775646, 14.0612011, -27.1844368, 14.0870647, -41.2646294, 41.2456360
12: -45.4562149, 18.8261929, -45.4622421, 18.8305683, -62.0605469, 62.0614777
13: -37.7976494, 26.0633278, -37.8202667, 26.0692253, -63.8668747, 63.8835945
14: -52.1660690, 16.0544891, -52.1743660, 16.0671139, -68.0682373, 68.0632629
15: -21.6437187, 29.1398621, -21.6603889, 29.1446953, -50.7884140, 50.8002510
16: -35.7171173, 20.8074474, -35.7266884, 20.8122559, -55.4781494, 55.4824982
17: -48.6928940, 12.4333820, -48.7013206, 12.4585686, -61.1514626, 61.1347046
18: -21.9935226, 30.1074486, -22.0003052, 30.1141338, -52.1076584, 52.1077538
19: -17.5743179, 19.9332848, -17.5800991, 19.9539795, -37.5282974, 37.5133820
20: -24.8760929, 14.0605106, -24.8809433, 14.0673199, -38.9434128, 38.9414520
21: -22.7472515, 21.9115543, -22.7557468, 21.9324818, -44.6797333, 44.6673012
22: -13.9322071, 27.4214077, -13.9377193, 27.4237328, -37.2656555, 37.2680740
23: -16.3181458, 22.8979950, -16.3229446, 22.9161873, -39.2343330, 39.2209396
24: -16.4955196, 23.8607407, -16.5007668, 23.8734035, -40.3689232, 40.3615074
25: -16.7511101, 28.0072250, -16.7571850, 28.0228157, -43.2956924, 43.2852554
26: -27.1705723, 35.4376793, -27.1782207, 35.4419479, -62.6125183, 62.6158981
27: -22.0408192, 19.5972404, -22.0459766, 19.6013050, -41.6421242, 41.6432190
28: -17.2144947, 25.1102257, -17.2188244, 25.1202946, -42.1004181, 42.0948105
29: -9.9229107, 22.7161407, -9.9287300, 22.7277031, -30.6962433, 30.6905785
30: -29.0236111, 15.6072922, -29.0302258, 15.6267366, -44.6503487, 44.6375198
31: -23.9069920, 24.1990376, -23.9144917, 24.2228298, -48.1298218, 48.1135292
32: -47.7409821, -1.8860030, -47.7593765, -1.8805642, -40.8082962, 40.8201904
33: -70.6685333, 5.5348988, -70.6923828, 5.5372896, -67.1641846, 67.1850433
34: -63.6541100, -5.6090522, -63.6674232, -5.6042328, -46.3091049, 46.3156967
35: -41.9569893, 14.3424425, -41.9678116, 14.3443766, -53.8158722, 53.8250809
36: -42.4185524, 16.1321507, -42.4368515, 16.1359444, -58.0315704, 58.0457764
37: -71.0387878, -1.2176418, -71.0478973, -1.2159595, -66.5318451, 66.5388489
38: -59.6169891, 8.7835312, -59.6299934, 8.7890072, -68.4059982, 68.4135284
39: -74.2395859, 1.1066222, -74.2676239, 1.1085887, -74.3434143, 74.3694763
40: -70.9537888, -18.7197552, -70.9704742, -18.7177086, -41.8384781, 41.8522568
41: -47.8647003, 3.7607098, -47.8736496, 3.7640963, -46.5376205, 46.5429153
42: -46.9621964, -3.8241773, -46.9655151, -3.8211365, -39.1296082, 39.1272926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 897

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0013104
time: 30.37 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0172566
time: 31.63 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -38.5760269, 25.3218117, -38.5437126, 25.2937012, -63.2537079, 63.2505035
1: -21.3537216, 21.3604317, -21.3267670, 21.3337765, -42.1460266, 42.1486359
2: -15.3863258, 23.6579018, -15.3749552, 23.6365433, -37.5672684, 37.5819550
3: -28.4897976, 18.9163971, -28.4838142, 18.8731117, -47.3629074, 47.4002113
4: -26.2831383, 22.7755470, -26.2763195, 22.7246170, -49.0077553, 49.0518646
5: -24.1931801, 24.2987347, -24.1817341, 24.2705231, -48.4637032, 48.4804688
6: -53.7604523, -5.1311903, -53.7456818, -5.1688080, -41.7974930, 41.8011627
7: -34.4934425, 15.3238506, -34.4530869, 15.3189220, -48.2630615, 48.2281036
8: -40.0634689, 17.2554913, -40.0399475, 17.2317238, -57.2742920, 57.2682571
9: -23.2384987, 25.9420471, -23.2216835, 25.8773708, -49.1158676, 49.1637306
10: -36.1159515, 25.0042286, -36.0562592, 24.9835377, -61.0994873, 61.0604858
11: -27.3947792, 14.1120939, -27.1885071, 14.1039124, -41.4986916, 41.3006020
12: -45.4855499, 18.8806515, -45.4625282, 18.8330154, -62.0933838, 62.1126556
13: -37.8468857, 26.1799126, -37.8315506, 26.0720482, -63.9189339, 64.0114594
14: -52.2825203, 16.0806351, -52.1781006, 16.0759983, -68.1933136, 68.0911255
15: -21.6820774, 29.2365417, -21.6706963, 29.1479263, -50.8300018, 50.9072380
16: -35.8206100, 20.8249245, -35.7325478, 20.8126316, -55.5854340, 55.5072327
17: -48.9004135, 12.4836512, -48.7058029, 12.4752846, -61.3756981, 61.1894531
18: -22.0562553, 30.1274071, -22.0027809, 30.1165085, -52.1727638, 52.1301880
19: -17.6989994, 19.9696960, -17.5830612, 19.9694901, -37.6684875, 37.5527573
20: -24.9385319, 14.0741453, -24.8835888, 14.0673895, -39.0059204, 38.9577332
21: -22.9056149, 21.9456596, -22.7610607, 21.9453049, -44.8509216, 44.7067184
22: -13.9878349, 27.4301491, -13.9400539, 27.4251328, -37.3196106, 37.2841263
23: -16.4502316, 22.9369087, -16.3256798, 22.9298668, -39.3800964, 39.2625885
24: -16.6103153, 23.8833065, -16.5030937, 23.8791943, -40.4895096, 40.3863983
25: -16.8542843, 28.0368042, -16.7601738, 28.0311527, -43.4078827, 43.3195724
26: -27.2495728, 35.4598083, -27.1810646, 35.4446793, -62.6942520, 62.6408730
27: -22.1006603, 19.6080875, -22.0486031, 19.6017609, -41.7024231, 41.6566925
28: -17.3139839, 25.1318531, -17.2209930, 25.1253891, -42.2029343, 42.1157455
29: -10.0504389, 22.7393303, -9.9312067, 22.7342873, -30.8307495, 30.7159958
30: -29.1955090, 15.6375456, -29.0344238, 15.6347523, -44.8302612, 44.6719704
31: -24.0402985, 24.2445869, -23.9184303, 24.2410069, -48.2813034, 48.1630173
32: -47.7847824, -1.7706466, -47.7712173, -1.8779736, -40.8518295, 40.9473267
33: -70.7418137, 5.6503973, -70.7100601, 5.5377827, -67.2307434, 67.3173828
34: -63.6929359, -5.5004144, -63.6757050, -5.6012573, -46.3449402, 46.4180298
35: -41.9980507, 14.4152298, -41.9744759, 14.3454695, -53.8524475, 53.9063950
36: -42.4518280, 16.2328434, -42.4437790, 16.1383591, -58.0664368, 58.1529541
37: -71.0821533, -1.1809320, -71.0505676, -1.2153358, -66.5787811, 66.5766144
38: -59.6327362, 8.9005461, -59.6305847, 8.7928143, -68.4255524, 68.5311279
39: -74.3105927, 1.2346520, -74.2859955, 1.1093917, -74.4165955, 74.5189056
40: -70.9939346, -18.6066895, -70.9783401, -18.7165833, -41.8778381, 41.9693909
41: -47.8988724, 3.8248277, -47.8778076, 3.7660408, -46.5735779, 46.6089478
42: -46.9800606, -3.7903385, -46.9670486, -3.8197708, -39.1788864, 39.1442261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=256, inp2_unstable=257, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=219, inp2_unstable=219, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=21, inp2_unstable=21, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=37, inp2_unstable=37, delta_unstable=43

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 897

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0191635
time: 36.71 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0351270
time: 36.48 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 75.41 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 75.41
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0013104
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 75.41
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0172566
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 75.41
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0191635
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 75.41
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0351270

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 49.60 + 231.48 = 281.08 seconds
