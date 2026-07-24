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
execution time: IAR + RelationalAnalysis = 2.50 + 53.12 = 55.62 seconds
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0431210, upper bound: 18.0252450
time: 47.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0431210, upper bound: 18.0431210
time: 58.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 106.14 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 106.14
Output dim: 29, lower bound: -18.0431210, upper bound: 18.0252450
IS_A2, status: Status.UNKNOWN, split count: 1, time: 106.14
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

Time for backsubstitution: 2.06 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0013104
time: 34.90 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0172566
time: 36.84 seconds

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

Time for backsubstitution: 2.11 seconds

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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0191635
time: 44.38 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0351270
time: 44.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 91.16 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 91.16
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0013104
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 91.16
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0172566
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 91.16
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0191635
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 91.16
Output dim: 29, lower bound: -18.0351271, upper bound: 18.0351270

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -38.5230331, 25.2251053, -38.5184822, 25.1873436, -63.0939789, 63.1291504
1: -21.3191319, 21.2741432, -21.3186283, 21.2416191, -42.0187836, 42.0509872
2: -15.3693123, 23.5854950, -15.3672943, 23.5573215, -37.4703293, 37.4973984
3: -28.4764938, 18.7988911, -28.4786987, 18.7561722, -47.2326660, 47.2775879
4: -26.2624969, 22.6588058, -26.2667007, 22.6218987, -48.8843956, 48.9255066
5: -24.1763725, 24.2129803, -24.1744328, 24.1829338, -48.3593063, 48.3874130
6: -53.7088814, -5.1781588, -53.6936340, -5.1807680, -41.7158813, 41.6974487
7: -34.4456139, 15.2706118, -34.4450874, 15.2435369, -48.1395416, 48.1655960
8: -40.0315247, 17.1596756, -40.0299606, 17.1247482, -57.1309662, 57.1637726
9: -23.1817360, 25.8323059, -23.1962204, 25.8066425, -48.9883804, 49.0285263
10: -36.0309143, 24.9633636, -36.0258102, 24.9668331, -60.9977493, 60.9891739
11: -27.1387100, 14.0553608, -27.1192055, 14.0771236, -41.2158356, 41.1745682
12: -45.4044037, 18.8200912, -45.3746719, 18.8202858, -61.9973450, 61.9661102
13: -37.7885208, 26.0389519, -37.8047867, 26.0278454, -63.8163681, 63.8437386
14: -52.1286850, 16.0505695, -52.1118698, 16.0606728, -68.0156097, 67.9823914
15: -21.6372147, 29.0951462, -21.6494331, 29.0695820, -50.7067947, 50.7445793
16: -35.7014389, 20.7915611, -35.7000809, 20.7854538, -55.4344177, 55.4387283
17: -48.6502151, 12.4263048, -48.6293640, 12.4466095, -61.0968246, 61.0556679
18: -21.9592667, 30.1009426, -21.9424572, 30.1031685, -52.0624352, 52.0433998
19: -17.5276489, 19.9320793, -17.5009918, 19.9519596, -37.4796066, 37.4330711
20: -24.8307858, 14.0554495, -24.8046246, 14.0586834, -38.8894691, 38.8600731
21: -22.7078857, 21.9085426, -22.6900291, 21.9272881, -44.6351738, 44.5985718
22: -13.8991508, 27.4183407, -13.8817329, 27.4185715, -37.2251053, 37.2054291
23: -16.2661114, 22.8938808, -16.2347507, 22.9092350, -39.1753464, 39.1286316
24: -16.4455299, 23.8572006, -16.4162884, 23.8674259, -40.3129578, 40.2734909
25: -16.6914558, 28.0030270, -16.6560001, 28.0156651, -43.2283554, 43.1790695
26: -27.1260910, 35.4297676, -27.1032295, 35.4285164, -62.5546074, 62.5329971
27: -22.0142651, 19.5925217, -22.0011539, 19.5933456, -41.6076126, 41.5936737
28: -17.1643887, 25.1047821, -17.1338825, 25.1110840, -42.0415497, 42.0046082
29: -9.8908072, 22.7131310, -9.8744545, 22.7226658, -30.6586380, 30.6325836
30: -28.9784527, 15.6023102, -28.9544716, 15.6183672, -44.5968208, 44.5567818
31: -23.8402481, 24.1950722, -23.8012867, 24.2161388, -48.0563889, 47.9963608
32: -47.7138176, -1.8913774, -47.7139969, -1.8895383, -40.7705002, 40.7659073
33: -70.6503983, 5.5269823, -70.6618195, 5.5239706, -67.1296844, 67.1431427
34: -63.6388245, -5.6147704, -63.6415138, -5.6138926, -46.2776947, 46.2769012
35: -41.9471054, 14.3366718, -41.9511452, 14.3346701, -53.7912292, 53.7961884
36: -42.3955307, 16.1274490, -42.3983917, 16.1280155, -57.9997253, 58.0014114
37: -70.9952393, -1.2242432, -70.9758759, -1.2272282, -66.4784851, 66.4612427
38: -59.5808144, 8.7747173, -59.5693512, 8.7741127, -68.3549271, 68.3440704
39: -74.2139969, 1.1023769, -74.2245789, 1.1014338, -74.3113403, 74.3241577
40: -70.9378357, -18.7277145, -70.9434662, -18.7310867, -41.8092499, 41.8162766
41: -47.8481674, 3.7545800, -47.8456497, 3.7537680, -46.5076752, 46.5042191
42: -46.9428864, -3.8294239, -46.9329224, -3.8299108, -39.0909348, 39.0713959

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
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
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
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
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 836
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
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1540
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

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1767

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0094570, upper bound: 17.9909350
time: 43.80 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0344396, upper bound: 18.0006226
time: 57.26 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -38.5365372, 25.2803574, -38.7278900, 25.2967529, -63.2128601, 63.3959961
1: -21.3239479, 21.3212585, -21.4826374, 21.3385429, -42.1168747, 42.2630997
2: -15.3741131, 23.6266251, -15.4881973, 23.6454086, -37.5597229, 37.6604919
3: -28.4787731, 18.8599110, -28.6464882, 18.8910065, -47.3697815, 47.5064011
4: -26.2673931, 22.7121468, -26.4313049, 22.7316818, -48.9990768, 49.1434517
5: -24.1797371, 24.2592621, -24.3201694, 24.2832489, -48.4629860, 48.5794296
6: -53.7330856, -5.1724358, -53.7554359, -5.1108170, -41.8456345, 41.7633247
7: -34.4488449, 15.3106871, -34.5982437, 15.3277283, -48.2200623, 48.3605804
8: -40.0359650, 17.2149887, -40.2140656, 17.2436371, -57.2486420, 57.4040680
9: -23.1904602, 25.8686714, -23.3660126, 25.8854790, -49.0759392, 49.2346840
10: -36.0448074, 24.9706631, -36.1411324, 25.0148144, -61.0596237, 61.1117935
11: -27.1729641, 14.0604620, -27.2216625, 14.1866531, -41.3596191, 41.2821236
12: -45.4506760, 18.8249187, -45.4832916, 18.9586105, -62.1953735, 62.0784454
13: -37.7963181, 26.0542202, -37.9076347, 26.0882530, -63.8845711, 63.9618530
14: -52.1603317, 16.0536976, -52.2444344, 16.1144257, -68.1297760, 68.1058502
15: -21.6426182, 29.1347942, -21.7946815, 29.1610470, -50.8036652, 50.9294739
16: -35.7151184, 20.8044872, -35.8654442, 20.8250599, -55.4852142, 55.6181030
17: -48.6883240, 12.4319515, -48.7603378, 12.5773487, -61.2656708, 61.1922913
18: -21.9894753, 30.1064911, -22.0366821, 30.2364826, -52.2259598, 52.1431732
19: -17.5693207, 19.9330940, -17.6059227, 20.0724583, -37.6417770, 37.5390167
20: -24.8711624, 14.0599327, -24.8994141, 14.1832638, -39.0544281, 38.9593468
21: -22.7428913, 21.9111595, -22.7905064, 22.0210934, -44.7639847, 44.7016678
22: -13.9285336, 27.4208984, -13.9602470, 27.5306206, -37.3702850, 37.2743912
23: -16.3126907, 22.8973885, -16.3407383, 23.0381069, -39.3507996, 39.2381287
24: -16.4900875, 23.8601913, -16.5209560, 23.9853077, -40.4753952, 40.3811493
25: -16.7448959, 28.0065269, -16.7856712, 28.1809616, -43.4501648, 43.3074951
26: -27.1650276, 35.4365463, -27.2084942, 35.5836220, -62.7486496, 62.6450424
27: -22.0364513, 19.5966358, -22.0559006, 19.6539688, -41.6904221, 41.6525345
28: -17.2090740, 25.1094475, -17.2348137, 25.2499447, -42.2266006, 42.1055603
29: -9.9194555, 22.7157440, -9.9564018, 22.8105621, -30.7756348, 30.7104874
30: -29.0183105, 15.6065283, -29.0518913, 15.7330856, -44.7513962, 44.6584206
31: -23.8999367, 24.1984940, -23.9414501, 24.3714314, -48.2713699, 48.1399460
32: -47.7377281, -1.8868685, -47.7723236, -1.8150597, -40.8791504, 40.8282547
33: -70.6644592, 5.5338221, -70.7062988, 5.6154709, -67.2441711, 67.1993103
34: -63.6505623, -5.6097655, -63.6776505, -5.5132017, -46.4055634, 46.3276672
35: -41.9554634, 14.3416843, -41.9830322, 14.4172659, -53.8794556, 53.8429184
36: -42.4145889, 16.1315060, -42.4455910, 16.2246571, -58.1207733, 58.0517578
37: -71.0334396, -1.2185154, -71.0769882, -1.1172256, -66.6340790, 66.5603485
38: -59.6120567, 8.7822952, -59.6592178, 8.8875999, -68.4996567, 68.4415131
39: -74.2344055, 1.1059303, -74.2962875, 1.1749063, -74.4062805, 74.3961639
40: -70.9498749, -18.7207146, -70.9873428, -18.6710548, -41.8923569, 41.8659744
41: -47.8614426, 3.7598534, -47.8842773, 3.8030171, -46.5918427, 46.5494385
42: -46.9590302, -3.8251333, -46.9792099, -3.7571020, -39.2337189, 39.1168976

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 873
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
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
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
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 793
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
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1306
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1767

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0094570, upper bound: 18.0068306
time: 36.92 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0344396, upper bound: 18.0165715
time: 40.54 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -38.5604744, 25.2595768, -38.5174179, 25.1893425, -63.1337128, 63.1636963
1: -21.3481674, 21.3074093, -21.3173332, 21.2436523, -42.0504456, 42.0861511
2: -15.3807697, 23.6122208, -15.3655291, 23.5589619, -37.4844513, 37.5273285
3: -28.4869804, 18.8485527, -28.4790382, 18.7579269, -47.2449074, 47.3275909
4: -26.2772751, 22.7162704, -26.2664070, 22.6242294, -48.9015045, 48.9826775
5: -24.1890106, 24.2472115, -24.1747627, 24.1836205, -48.3726311, 48.4219742
6: -53.7313690, -5.1377182, -53.6963348, -5.1798391, -41.7488861, 41.7309227
7: -34.4893188, 15.2791519, -34.4462662, 15.2429142, -48.1836700, 48.1769714
8: -40.0582428, 17.1939583, -40.0310783, 17.1273422, -57.1649170, 57.1981049
9: -23.2285709, 25.9009018, -23.2048225, 25.8077812, -49.0363541, 49.1057243
10: -36.1000824, 24.9955463, -36.0293121, 24.9688931, -61.0689774, 61.0248566
11: -27.3559341, 14.1062298, -27.1232796, 14.0939770, -41.4499130, 41.2295074
12: -45.4337387, 18.8745384, -45.3750114, 18.8227768, -62.0302734, 62.0172577
13: -37.8377457, 26.1554375, -37.8161354, 26.0307369, -63.8684845, 63.9715729
14: -52.2450676, 16.0767899, -52.1156578, 16.0694427, -68.1405640, 68.0102997
15: -21.6755486, 29.1918144, -21.6597557, 29.0727997, -50.7483482, 50.8515701
16: -35.8049316, 20.8090172, -35.7059326, 20.7858219, -55.5415955, 55.4633789
17: -48.8577042, 12.4766026, -48.6338348, 12.4633007, -61.3210068, 61.1104355
18: -22.0220146, 30.1208744, -21.9449387, 30.1055450, -52.1275597, 52.0658112
19: -17.6523151, 19.9684620, -17.5039349, 19.9674530, -37.6197662, 37.4723969
20: -24.8932304, 14.0691013, -24.8072910, 14.0587645, -38.9519958, 38.8763924
21: -22.8661804, 21.9425983, -22.6952858, 21.9401474, -44.8063278, 44.6378860
22: -13.9547844, 27.4270859, -13.8840866, 27.4199696, -37.2790833, 37.2215042
23: -16.3981743, 22.9327316, -16.2374725, 22.9229069, -39.3210831, 39.1702042
24: -16.5602970, 23.8797741, -16.4185944, 23.8732281, -40.4335251, 40.2983704
25: -16.7946491, 28.0326118, -16.6589851, 28.0240459, -43.3405228, 43.2133408
26: -27.2051239, 35.4518852, -27.1060734, 35.4312325, -62.6363564, 62.5579605
27: -22.0741158, 19.6033630, -22.0037956, 19.5937920, -41.6679077, 41.6071587
28: -17.2638550, 25.1264000, -17.1360073, 25.1161766, -42.1440277, 42.0254936
29: -10.0183163, 22.7363510, -9.8769026, 22.7292442, -30.7931061, 30.6580200
30: -29.1503601, 15.6326151, -28.9586315, 15.6263866, -44.7767487, 44.5912476
31: -23.9735699, 24.2406101, -23.8052540, 24.2343407, -48.2079086, 48.0458641
32: -47.7576141, -1.7759910, -47.7258530, -1.8869390, -40.8140717, 40.8930740
33: -70.7237549, 5.6424999, -70.6794739, 5.5244722, -67.1963348, 67.2755890
34: -63.6776123, -5.5062037, -63.6497765, -5.6109428, -46.3134842, 46.3791656
35: -41.9882278, 14.4095144, -41.9578400, 14.3357744, -53.8278046, 53.8775024
36: -42.4287758, 16.2281342, -42.4052811, 16.1304131, -58.0345154, 58.1086121
37: -71.0385437, -1.1875734, -70.9786072, -1.2265148, -66.5253448, 66.4990234
38: -59.5965691, 8.8916931, -59.5699463, 8.7778893, -68.3744583, 68.4616394
39: -74.2849731, 1.2304988, -74.2430191, 1.1022959, -74.3844299, 74.4736176
40: -70.9779510, -18.6146870, -70.9513016, -18.7299576, -41.8485718, 41.9334717
41: -47.8823471, 3.8187146, -47.8497620, 3.7557859, -46.5436783, 46.5702362
42: -46.9607582, -3.7955494, -46.9344597, -3.8285565, -39.1402130, 39.0883369

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1335
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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1762
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
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1281
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
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1538
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
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 897

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1767

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0094570, upper bound: 18.0088683
time: 43.96 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0344396, upper bound: 18.0184756
time: 44.21 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -38.5739746, 25.3147373, -38.7268295, 25.2987843, -63.2526245, 63.4304810
1: -21.3530045, 21.3545456, -21.4813538, 21.3405762, -42.1485138, 42.2982101
2: -15.3855429, 23.6533241, -15.4864416, 23.6470413, -37.5738220, 37.6904449
3: -28.4892883, 18.9095898, -28.6469193, 18.8927917, -47.3820801, 47.5565109
4: -26.2821922, 22.7696095, -26.4310036, 22.7339478, -49.0161400, 49.2006149
5: -24.1923256, 24.2934647, -24.3204765, 24.2840023, -48.4763260, 48.6139412
6: -53.7555618, -5.1320009, -53.7581100, -5.1098661, -41.8786545, 41.7968140
7: -34.4925919, 15.3192835, -34.5995064, 15.3271074, -48.2643127, 48.3719788
8: -40.0626907, 17.2492409, -40.2152023, 17.2462807, -57.2825775, 57.4382553
9: -23.2372475, 25.9371853, -23.3746605, 25.8866100, -49.1238556, 49.3118439
10: -36.1139603, 25.0028172, -36.1446381, 25.0168476, -61.1308060, 61.1474533
11: -27.3902092, 14.1113491, -27.2256851, 14.2035027, -41.5937119, 41.3370361
12: -45.4800224, 18.8793774, -45.4835815, 18.9610348, -62.2281799, 62.1295166
13: -37.8454971, 26.1707363, -37.9189262, 26.0911064, -63.9366035, 64.0896606
14: -52.2767982, 16.0798264, -52.2481651, 16.1232662, -68.2547913, 68.1337280
15: -21.6809330, 29.2314377, -21.8049583, 29.1643105, -50.8452454, 51.0363960
16: -35.8185654, 20.8219681, -35.8713493, 20.8254013, -55.5924683, 55.6427155
17: -48.8957939, 12.4822235, -48.7648811, 12.5941114, -61.4899063, 61.2471046
18: -22.0522137, 30.1264572, -22.0391541, 30.2388878, -52.2910995, 52.1656113
19: -17.6939564, 19.9694901, -17.6088600, 20.0879745, -37.7819290, 37.5783501
20: -24.9336128, 14.0735693, -24.9020786, 14.1833363, -39.1169510, 38.9756470
21: -22.9012566, 21.9452724, -22.7957592, 22.0339146, -44.9351730, 44.7410316
22: -13.9841452, 27.4296532, -13.9626131, 27.5319824, -37.4242325, 37.2904739
23: -16.4447327, 22.9362335, -16.3434486, 23.0517902, -39.4965210, 39.2796822
24: -16.6048737, 23.8827667, -16.5232906, 23.9911156, -40.5959892, 40.4060593
25: -16.8480721, 28.0360699, -16.7886200, 28.1893044, -43.5623550, 43.3417740
26: -27.2440720, 35.4586334, -27.2113571, 35.5863037, -62.8303757, 62.6699905
27: -22.0963497, 19.6074867, -22.0585155, 19.6544056, -41.7507553, 41.6660004
28: -17.3085423, 25.1311016, -17.2369499, 25.2550335, -42.3291550, 42.1264496
29: -10.0470209, 22.7389145, -9.9588528, 22.8171120, -30.9101219, 30.7359390
30: -29.1902332, 15.6368332, -29.0560398, 15.7411137, -44.9313469, 44.6928711
31: -24.0332584, 24.2440758, -23.9453697, 24.3896503, -48.4229088, 48.1894455
32: -47.7815247, -1.7715049, -47.7841873, -1.8125157, -40.9227066, 40.9554596
33: -70.7377243, 5.6492901, -70.7240143, 5.6160297, -67.3107300, 67.3317261
34: -63.6893616, -5.5011091, -63.6858902, -5.5102692, -46.4413834, 46.4300461
35: -41.9965248, 14.4144850, -41.9897232, 14.4183912, -53.9160004, 53.9243317
36: -42.4478683, 16.2321663, -42.4525375, 16.2270470, -58.1556091, 58.1590576
37: -71.0767670, -1.1818504, -71.0796509, -1.1165400, -66.6810608, 66.5981750
38: -59.6277695, 8.8993635, -59.6598434, 8.8912678, -68.5190353, 68.5592041
39: -74.3054276, 1.2340050, -74.3146591, 1.1757531, -74.4794922, 74.5455933
40: -70.9900818, -18.6076546, -70.9952011, -18.6698799, -41.9317093, 41.9831924
41: -47.8956070, 3.8239417, -47.8884010, 3.8049841, -46.6277771, 46.6154556
42: -46.9768982, -3.7912645, -46.9807510, -3.7556920, -39.2829742, 39.1338501

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1335
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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1502
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
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1594
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
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1507
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
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1281
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
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1667
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
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1540
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
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 897

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1767

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0094570, upper bound: 18.0247692
time: 54.53 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0344396, upper bound: 18.0344396
time: 46.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 103.42 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 103.42
Output dim: 29, lower bound: -18.0094570, upper bound: 17.9909350
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 103.42
Output dim: 29, lower bound: -18.0344396, upper bound: 18.0006226
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 103.42
Output dim: 29, lower bound: -18.0094570, upper bound: 18.0068306
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 103.42
Output dim: 29, lower bound: -18.0344396, upper bound: 18.0165715
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 103.42
Output dim: 29, lower bound: -18.0094570, upper bound: 18.0088683
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 103.42
Output dim: 29, lower bound: -18.0344396, upper bound: 18.0184756
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 103.42
Output dim: 29, lower bound: -18.0094570, upper bound: 18.0247692
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 103.42
Output dim: 29, lower bound: -18.0344396, upper bound: 18.0344396

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -38.3956413, 25.0944901, -38.4947319, 25.1118011, -62.8878937, 62.9677124
1: -21.2420731, 21.1736488, -21.3068047, 21.1851692, -41.8858261, 41.9324036
2: -15.3088799, 23.4852428, -15.3613625, 23.5008183, -37.3607712, 37.3903427
3: -28.4227676, 18.7046204, -28.4733582, 18.7032146, -47.1259842, 47.1779785
4: -26.1757507, 22.5424690, -26.2584801, 22.5552750, -48.7310257, 48.8009491
5: -24.1091957, 24.1238956, -24.1680412, 24.1317844, -48.2409821, 48.2919388
6: -53.6154327, -5.2305326, -53.6411324, -5.1894875, -41.6090317, 41.7215538
7: -34.3819504, 15.1831417, -34.4363632, 15.1940498, -48.0214539, 48.0623856
8: -39.9453773, 17.0000935, -40.0245323, 17.0335617, -56.9520416, 56.9958801
9: -23.0946388, 25.7282410, -23.1860981, 25.7473564, -48.8419952, 48.9143372
10: -35.9867859, 24.8908787, -36.0119171, 24.9313087, -60.9180946, 60.9027939
11: -27.0097961, 13.9862890, -27.0469379, 14.0702667, -41.0800629, 41.0332260
12: -45.3002167, 18.7386017, -45.3161545, 18.8081722, -61.8809509, 61.8264923
13: -37.7251740, 25.9019814, -37.7910652, 25.9558125, -63.6809845, 63.6930466
14: -52.0662003, 15.9915705, -52.0946350, 16.0296402, -67.9223480, 67.9024048
15: -21.5503483, 28.9817505, -21.6385803, 29.0061951, -50.5565414, 50.6203308
16: -35.6215057, 20.7265797, -35.6777916, 20.7494373, -55.3225250, 55.3509827
17: -48.5617943, 12.4078455, -48.5904312, 12.4401970, -61.0019913, 60.9982758
18: -21.7912025, 30.0344219, -21.8519115, 30.0984383, -51.8896408, 51.8863335
19: -17.4385509, 19.8966713, -17.4541855, 19.9510803, -37.3896332, 37.3508568
20: -24.7431583, 13.9865150, -24.7582188, 14.0489674, -38.7921257, 38.7447357
21: -22.5867023, 21.8496647, -22.6271420, 21.9239082, -44.5106125, 44.4768066
22: -13.8141270, 27.3857841, -13.8375454, 27.4162941, -37.1374664, 37.1500397
23: -16.1478653, 22.8237572, -16.1704578, 22.9015636, -39.0494308, 38.9942169
24: -16.3261089, 23.8157234, -16.3504639, 23.8630905, -40.1741943, 40.1607056
25: -16.5956879, 27.9482746, -16.6055183, 28.0098515, -43.1264954, 43.1191483
26: -26.9885120, 35.3511009, -27.0292568, 35.4218483, -62.4103622, 62.3803558
27: -21.9250813, 19.5709038, -21.9530849, 19.5846596, -41.5097427, 41.5239868
28: -17.0513268, 25.0408287, -17.0723534, 25.1026802, -41.9209976, 41.8787537
29: -9.8037300, 22.6863747, -9.8288918, 22.7196388, -30.5673218, 30.5804138
30: -28.8398571, 15.5352478, -28.8768768, 15.6075220, -44.4473801, 44.4121246
31: -23.7061596, 24.1458702, -23.7291851, 24.2128868, -47.9190445, 47.8750534
32: -47.6273956, -1.9474401, -47.6661530, -1.9063349, -40.6666412, 40.6796570
33: -70.5802612, 5.4950237, -70.6263580, 5.5133524, -67.0391541, 67.0740814
34: -63.5371513, -5.6752577, -63.5861473, -5.6237779, -46.1119308, 46.1598892
35: -41.8713379, 14.2998552, -41.9108734, 14.3277225, -53.6483154, 53.7214966
36: -42.3348274, 16.0941124, -42.3649559, 16.1218719, -57.9335022, 57.9436951
37: -70.9026794, -1.2652760, -70.9261322, -1.2343416, -66.2944489, 66.3591614
38: -59.4758072, 8.7210360, -59.5146713, 8.7589893, -68.2347946, 68.2357101
39: -74.1696701, 1.0605683, -74.2031403, 1.0808182, -74.2688293, 74.2595367
40: -70.8758850, -18.7623558, -70.9125366, -18.7463760, -41.7235870, 41.7429276
41: -47.7692413, 3.7155991, -47.8010559, 3.7430844, -46.3979950, 46.4180527
42: -46.8869934, -3.8695307, -46.9017868, -3.8411903, -39.0242157, 39.0430679

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1698
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
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1404
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
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 911
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
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
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
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 894
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
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1766
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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1335

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9965380, upper bound: 17.9886508
time: 48.99 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0074334, upper bound: 17.9889105
time: 47.18 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -38.5206985, 25.2224960, -38.5170898, 25.1848679, -63.0888367, 63.1023712
1: -21.3171539, 21.2719326, -21.3174934, 21.2397366, -42.0146942, 42.0461807
2: -15.3685112, 23.5821686, -15.3668470, 23.5552902, -37.4675369, 37.4834824
3: -28.4758453, 18.7954636, -28.4782677, 18.7540131, -47.2298584, 47.2737312
4: -26.2616577, 22.6547623, -26.2661610, 22.6194267, -48.8810844, 48.9209213
5: -24.1755981, 24.2112617, -24.1739502, 24.1816216, -48.3572197, 48.3852119
6: -53.7049904, -5.1789093, -53.6910667, -5.1812592, -41.6668396, 41.6946869
7: -34.4438057, 15.2676449, -34.4439011, 15.2417021, -48.1389160, 48.1597290
8: -40.0310249, 17.1536293, -40.0296097, 17.1210747, -57.1274109, 57.1569824
9: -23.1806393, 25.8278255, -23.1955681, 25.8034954, -48.9841347, 49.0233917
10: -36.0291290, 24.9568024, -36.0246811, 24.9625511, -60.9916801, 60.9814835
11: -27.1334114, 14.0545845, -27.1153202, 14.0766296, -41.2100410, 41.1699066
12: -45.4002800, 18.8191452, -45.3722229, 18.8196049, -61.9710236, 61.9627686
13: -37.7867355, 26.0348434, -37.8036842, 26.0253563, -63.8120918, 63.8385277
14: -52.1272850, 16.0469589, -52.1109047, 16.0579491, -68.0137329, 67.9799194
15: -21.6361675, 29.0914612, -21.6487846, 29.0672531, -50.7034225, 50.7402458
16: -35.6980057, 20.7864647, -35.6979828, 20.7816734, -55.4330368, 55.4313126
17: -48.6390381, 12.4257774, -48.6226044, 12.4462643, -61.0853043, 61.0483818
18: -21.9534798, 30.1004524, -21.9389343, 30.1029091, -52.0563889, 52.0393867
19: -17.5254288, 19.9320164, -17.4994507, 19.9519119, -37.4773407, 37.4314651
20: -24.8299160, 14.0545597, -24.8039341, 14.0581770, -38.8880920, 38.8584938
21: -22.7031212, 21.9081974, -22.6869946, 21.9271412, -44.6302643, 44.5951920
22: -13.8962736, 27.4179325, -13.8796206, 27.4182892, -37.2186203, 37.2094421
23: -16.2621326, 22.8930492, -16.2320576, 22.9086876, -39.1708221, 39.1251068
24: -16.4429188, 23.8567677, -16.4145050, 23.8671513, -40.3100700, 40.2712708
25: -16.6867390, 28.0024986, -16.6528454, 28.0153904, -43.2217178, 43.1804428
26: -27.1215477, 35.4291649, -27.1002331, 35.4281158, -62.5496635, 62.5293961
27: -22.0124340, 19.5917416, -21.9999027, 19.5928631, -41.6052971, 41.5916443
28: -17.1626225, 25.1041489, -17.1326733, 25.1106930, -42.0377121, 42.0047226
29: -9.8861599, 22.7128258, -9.8714561, 22.7224808, -30.6533623, 30.6307182
30: -28.9749336, 15.6014156, -28.9516277, 15.6178417, -44.5927734, 44.5530434
31: -23.8353653, 24.1947327, -23.7981453, 24.2159557, -48.0513229, 47.9928780
32: -47.7125244, -1.8926287, -47.7131958, -1.8903561, -40.7438354, 40.7633362
33: -70.6459122, 5.5253859, -70.6589355, 5.5229692, -67.1208344, 67.1395264
34: -63.6371422, -5.6169810, -63.6401749, -5.6152344, -46.2413635, 46.2727509
35: -41.9440765, 14.3351917, -41.9491005, 14.3337555, -53.7796326, 53.7929382
36: -42.3925018, 16.1262264, -42.3964043, 16.1272526, -57.9942627, 57.9982910
37: -70.9908295, -1.2253237, -70.9728470, -1.2278662, -66.4682770, 66.4643860
38: -59.5753708, 8.7720184, -59.5660553, 8.7724628, -68.3478317, 68.3380737
39: -74.2117310, 1.0926580, -74.2230835, 1.0954370, -74.3031769, 74.3108368
40: -70.9352875, -18.7316742, -70.9418640, -18.7334900, -41.7946243, 41.8063889
41: -47.8467178, 3.7537308, -47.8448143, 3.7532034, -46.4957123, 46.5017090
42: -46.9400253, -3.8302979, -46.9308510, -3.8305354, -39.0632401, 39.0688248

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1698
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
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1404
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
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
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
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 911
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
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 836
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
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 894
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
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1533
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

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1335

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0214443, upper bound: 17.9983100
time: 52.33 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0324140, upper bound: 17.9985973
time: 37.52 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -38.4091492, 25.1496487, -38.7042084, 25.2212696, -63.0068665, 63.2345886
1: -21.2468910, 21.2207317, -21.4708595, 21.2821140, -41.9839172, 42.1445160
2: -15.3136606, 23.5263882, -15.4822054, 23.5889091, -37.4500885, 37.5534286
3: -28.4250813, 18.7656021, -28.6412239, 18.8380680, -47.2631493, 47.4068260
4: -26.1807213, 22.5957680, -26.4231148, 22.6651306, -48.8458519, 49.0188828
5: -24.1125221, 24.1701260, -24.3137550, 24.2321396, -48.3446617, 48.4838791
6: -53.6396217, -5.2247953, -53.7029037, -5.1195135, -41.7387924, 41.7873878
7: -34.3851395, 15.2232685, -34.5895462, 15.2783070, -48.1020508, 48.2574005
8: -39.9498367, 17.0553894, -40.2086563, 17.1525116, -57.0696564, 57.2361832
9: -23.1033363, 25.7645512, -23.3559456, 25.8261757, -48.9295120, 49.1204987
10: -36.0006752, 24.8981800, -36.1272736, 24.9792671, -60.9799423, 61.0254517
11: -27.0440311, 13.9914036, -27.1493320, 14.1797752, -41.2238083, 41.1407356
12: -45.3465118, 18.7434406, -45.4247475, 18.9465370, -62.0787506, 61.9387970
13: -37.7329407, 25.9171848, -37.8939056, 26.0161934, -63.7491341, 63.8110886
14: -52.0978317, 15.9946318, -52.2270508, 16.0834045, -68.0365295, 68.0257568
15: -21.5558186, 29.0213547, -21.7837811, 29.0976639, -50.6534805, 50.8051376
16: -35.6352463, 20.7395210, -35.8432045, 20.7890205, -55.3733902, 55.5304260
17: -48.5997772, 12.4134789, -48.7213326, 12.5710201, -61.1707993, 61.1348114
18: -21.8214245, 30.0399456, -21.9461479, 30.2318153, -52.0532379, 51.9860916
19: -17.4801731, 19.8976517, -17.5591450, 20.0716171, -37.5517883, 37.4567947
20: -24.7835369, 13.9910460, -24.8530025, 14.1735163, -38.9570541, 38.8440475
21: -22.6217308, 21.8522873, -22.7276134, 22.0176773, -44.6394081, 44.5799026
22: -13.8434429, 27.3883686, -13.9161224, 27.5283337, -37.2826233, 37.2190170
23: -16.1944218, 22.8272972, -16.2764778, 23.0304928, -39.2249146, 39.1037750
24: -16.3706417, 23.8187408, -16.4551468, 23.9809818, -40.3392029, 40.2625351
25: -16.6490955, 27.9517536, -16.7351952, 28.1751289, -43.3483887, 43.2475739
26: -27.0274887, 35.3578224, -27.1345711, 35.5769310, -62.6044197, 62.4923935
27: -21.9472523, 19.5750275, -22.0078259, 19.6453056, -41.5925598, 41.5828552
28: -17.0960178, 25.0455666, -17.1732731, 25.2415943, -42.1061249, 41.9797211
29: -9.8323441, 22.6889534, -9.9108295, 22.8075218, -30.6843185, 30.6583061
30: -28.8796749, 15.5394802, -28.9742298, 15.7222567, -44.6019325, 44.5137100
31: -23.7658253, 24.1492805, -23.8693256, 24.3681850, -48.1340103, 48.0186081
32: -47.6513138, -1.9429770, -47.7245026, -1.8318367, -40.7752457, 40.7419510
33: -70.5942612, 5.5018415, -70.6708221, 5.6048899, -67.1535339, 67.1301880
34: -63.5489197, -5.6701870, -63.6222229, -5.5230298, -46.2398300, 46.2108002
35: -41.8796158, 14.3048849, -41.9428482, 14.4103193, -53.7364960, 53.7682800
36: -42.3539238, 16.0981789, -42.4122162, 16.2185459, -58.0545807, 57.9940338
37: -70.9408035, -1.2594547, -71.0270691, -1.1243134, -66.4500427, 66.4583130
38: -59.5069847, 8.7286673, -59.6045341, 8.8725014, -68.3794861, 68.3331985
39: -74.1900482, 1.0639539, -74.2747803, 1.1542168, -74.3637085, 74.3314819
40: -70.8879471, -18.7554245, -70.9563980, -18.6863041, -41.8066483, 41.7925491
41: -47.7825241, 3.7208424, -47.8396835, 3.7922869, -46.4821167, 46.4632721
42: -46.9031258, -3.8652544, -46.9480896, -3.7683411, -39.1670609, 39.0885620

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 906
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
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1404
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
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 792
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
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 894
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
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 897

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1335

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9965380, upper bound: 18.0045598
time: 62.95 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0074334, upper bound: 18.0048071
time: 42.19 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -38.5342598, 25.2776318, -38.7265472, 25.2943935, -63.2077484, 63.3692322
1: -21.3219872, 21.3190308, -21.4814453, 21.3366241, -42.1127777, 42.2582855
2: -15.3733082, 23.6233025, -15.4876842, 23.6433678, -37.5568161, 37.6465912
3: -28.4781933, 18.8564796, -28.6461220, 18.8888931, -47.3670883, 47.5026016
4: -26.2665596, 22.7080994, -26.4307518, 22.7291565, -48.9957161, 49.1388512
5: -24.1789589, 24.2575607, -24.3196602, 24.2819481, -48.4609070, 48.5772209
6: -53.7292595, -5.1731548, -53.7528419, -5.1113434, -41.7966385, 41.7605286
7: -34.4470596, 15.3077488, -34.5971451, 15.3258896, -48.2195740, 48.3547134
8: -40.0354576, 17.2089291, -40.2137299, 17.2399883, -57.2450714, 57.3972626
9: -23.1893349, 25.8641167, -23.3653603, 25.8823013, -49.0716362, 49.2294769
10: -36.0430679, 24.9641209, -36.1399765, 25.0105114, -61.0535812, 61.1040955
11: -27.1676769, 14.0597286, -27.2177315, 14.1861763, -41.3538513, 41.2774582
12: -45.4465981, 18.8239555, -45.4807663, 18.9579430, -62.1689148, 62.0749664
13: -37.7945137, 26.0501328, -37.9064751, 26.0857162, -63.8802299, 63.9566078
14: -52.1589355, 16.0500393, -52.2434464, 16.1117344, -68.1279144, 68.1033630
15: -21.6415653, 29.1310730, -21.7940407, 29.1587105, -50.8002777, 50.9251137
16: -35.7116623, 20.7994156, -35.8633270, 20.8212719, -55.4838257, 55.6106567
17: -48.6771011, 12.4313946, -48.7535553, 12.5769863, -61.2540894, 61.1849518
18: -21.9837399, 30.1060467, -22.0331688, 30.2362518, -52.2199936, 52.1392136
19: -17.5670471, 19.9330215, -17.6044216, 20.0724335, -37.6394806, 37.5374451
20: -24.8703060, 14.0590992, -24.8987160, 14.1827106, -39.0530167, 38.9578171
21: -22.7381554, 21.9108639, -22.7874889, 22.0209007, -44.7590561, 44.6983528
22: -13.9256477, 27.4204865, -13.9581833, 27.5303478, -37.3637695, 37.2783890
23: -16.3086815, 22.8965778, -16.3380642, 23.0375805, -39.3462601, 39.2346420
24: -16.4875088, 23.8597584, -16.5191498, 23.9850235, -40.4725342, 40.3789062
25: -16.7401600, 28.0060081, -16.7825336, 28.1806374, -43.4435577, 43.3088379
26: -27.1605282, 35.4358864, -27.2055073, 35.5831985, -62.7437286, 62.6413956
27: -22.0346222, 19.5958672, -22.0546341, 19.6534405, -41.6880646, 41.6505013
28: -17.2073383, 25.1088467, -17.2335625, 25.2495499, -42.2227859, 42.1057053
29: -9.9148006, 22.7153950, -9.9534388, 22.8103409, -30.7703514, 30.7086525
30: -29.0147820, 15.6056480, -29.0490189, 15.7325249, -44.7473068, 44.6546669
31: -23.8950729, 24.1981812, -23.9383163, 24.3712349, -48.2663078, 48.1364975
32: -47.7364540, -1.8880897, -47.7715340, -1.8158965, -40.8525543, 40.8256683
33: -70.6599350, 5.5322266, -70.7034912, 5.6144428, -67.2352600, 67.1957245
34: -63.6488838, -5.6118984, -63.6763763, -5.5145063, -46.3692703, 46.3235855
35: -41.9523926, 14.3401918, -41.9810028, 14.4163256, -53.8678131, 53.8396988
36: -42.4115601, 16.1302605, -42.4437065, 16.2239285, -58.1153717, 58.0486755
37: -71.0290146, -1.2195606, -71.0738831, -1.1179066, -66.6239319, 66.5635376
38: -59.6066208, 8.7796326, -59.6559258, 8.8859434, -68.4925613, 68.4355621
39: -74.2321320, 1.0961256, -74.2947311, 1.1689200, -74.3982544, 74.3828583
40: -70.9473419, -18.7247086, -70.9858017, -18.6734505, -41.8777466, 41.8561401
41: -47.8599815, 3.7590451, -47.8833580, 3.8023925, -46.5798721, 46.5469131
42: -46.9561996, -3.8260021, -46.9771271, -3.7576847, -39.2059937, 39.1143570

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1698
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
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1404
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
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 792
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
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 820
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
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 894
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
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1533
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

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1335

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0214443, upper bound: 18.0142684
time: 38.08 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0324140, upper bound: 18.0145462
time: 44.16 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -38.4331055, 25.1288757, -38.4936295, 25.1138134, -62.9277649, 63.0021973
1: -21.2711773, 21.2068901, -21.3055191, 21.1872253, -41.9175415, 41.9675140
2: -15.3203726, 23.5119247, -15.3595572, 23.5023918, -37.3749161, 37.4202881
3: -28.4333038, 18.7541771, -28.4737816, 18.7049713, -47.1382751, 47.2279587
4: -26.1906128, 22.5999146, -26.2582283, 22.5576305, -48.7482452, 48.8581429
5: -24.1217842, 24.1580162, -24.1682663, 24.1325016, -48.2542877, 48.3262825
6: -53.6379547, -5.1900501, -53.6437759, -5.1885691, -41.6419754, 41.7550201
7: -34.4256706, 15.1917191, -34.4375725, 15.1934719, -48.0657349, 48.0737915
8: -39.9720764, 17.0343552, -40.0256996, 17.0362129, -56.9860840, 57.0301743
9: -23.1415157, 25.7967873, -23.1947231, 25.7485256, -48.8900414, 48.9915085
10: -36.0559235, 24.9230080, -36.0155106, 24.9332905, -60.9892120, 60.9385185
11: -27.2269173, 14.0371704, -27.0510101, 14.0871239, -41.3140411, 41.0881805
12: -45.3294983, 18.7930164, -45.3164597, 18.8106709, -61.9137573, 61.8777771
13: -37.7743759, 26.0184383, -37.8024025, 25.9586582, -63.7330322, 63.8208389
14: -52.1826248, 16.0177803, -52.0984077, 16.0384293, -68.0472565, 67.9302979
15: -21.5887108, 29.0783577, -21.6488361, 29.0093651, -50.5980759, 50.7271957
16: -35.7250214, 20.7440300, -35.6836853, 20.7498131, -55.4297714, 55.3756409
17: -48.7692070, 12.4581881, -48.5948906, 12.4569435, -61.2261505, 61.0530777
18: -21.8539200, 30.0544014, -21.8544254, 30.1008034, -51.9547234, 51.9088287
19: -17.5631580, 19.9330807, -17.4571228, 19.9665871, -37.5297470, 37.3902054
20: -24.8056183, 14.0002604, -24.7608566, 14.0490417, -38.8546600, 38.7611160
21: -22.7449894, 21.8837357, -22.6324120, 21.9367199, -44.6817093, 44.5161476
22: -13.8696613, 27.3945236, -13.8399200, 27.4176636, -37.1913910, 37.1660767
23: -16.2798824, 22.8626862, -16.1731873, 22.9152508, -39.1951332, 39.0358734
24: -16.4408836, 23.8383465, -16.3528252, 23.8688793, -40.3023987, 40.1877365
25: -16.6988449, 27.9778748, -16.6085014, 28.0181942, -43.2386932, 43.1534271
26: -27.0675735, 35.3731613, -27.0321331, 35.4245338, -62.4921074, 62.4052963
27: -21.9848747, 19.5817108, -21.9557266, 19.5850964, -41.5699692, 41.5374374
28: -17.1507683, 25.0624771, -17.0745125, 25.1077957, -42.0234146, 41.8996429
29: -9.9311428, 22.7095718, -9.8313484, 22.7262249, -30.7016983, 30.6058655
30: -29.0116863, 15.5655527, -28.8810253, 15.6155386, -44.6272240, 44.4465790
31: -23.8394375, 24.1914330, -23.7331219, 24.2310772, -48.0705147, 47.9245529
32: -47.6712189, -1.8319902, -47.6780624, -1.9037657, -40.7101517, 40.8069000
33: -70.6534424, 5.6105490, -70.6440201, 5.5138855, -67.1055908, 67.2064514
34: -63.5759506, -5.5665841, -63.5943985, -5.6207623, -46.1477432, 46.2622757
35: -41.9123497, 14.3726492, -41.9176216, 14.3288765, -53.6847992, 53.8028564
36: -42.3681335, 16.1948643, -42.3718758, 16.1243172, -57.9683228, 58.0509186
37: -70.9458466, -1.2285805, -70.9288025, -1.2336893, -66.3413696, 66.3969879
38: -59.4915886, 8.8382607, -59.5152740, 8.7626896, -68.2542801, 68.3535309
39: -74.2406311, 1.1886635, -74.2215271, 1.0816388, -74.3418884, 74.4089203
40: -70.9159775, -18.6493416, -70.9203644, -18.7452621, -41.7628250, 41.8600616
41: -47.8033905, 3.7797413, -47.8052139, 3.7450857, -46.4339371, 46.4841385
42: -46.9048462, -3.8356671, -46.9032936, -3.8397985, -39.0734940, 39.0599785

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1698
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
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1404
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
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 911
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
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
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
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 894
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
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1766
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1335

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9965380, upper bound: 18.0065851
time: 37.84 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0074334, upper bound: 18.0068438
time: 42.80 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -38.5581703, 25.2568836, -38.5159950, 25.1869392, -63.1285400, 63.1368866
1: -21.3462181, 21.3052082, -21.3160954, 21.2417660, -42.0463257, 42.0812988
2: -15.3800049, 23.6089001, -15.3651161, 23.5568829, -37.4816284, 37.5133972
3: -28.4863644, 18.8450642, -28.4786720, 18.7558098, -47.2421722, 47.3237381
4: -26.2764587, 22.7121944, -26.2659073, 22.6217117, -48.8981705, 48.9781036
5: -24.1882248, 24.2454834, -24.1742382, 24.1823273, -48.3705521, 48.4197235
6: -53.7274933, -5.1384249, -53.6937332, -5.1802998, -41.6998520, 41.7281036
7: -34.4875374, 15.2762365, -34.4451714, 15.2411213, -48.1831665, 48.1711121
8: -40.0577278, 17.1879387, -40.0307579, 17.1236954, -57.1614227, 57.1912537
9: -23.2274437, 25.8963947, -23.2041225, 25.8046303, -49.0320740, 49.1005173
10: -36.0983124, 24.9890003, -36.0282211, 24.9646263, -61.0629387, 61.0172195
11: -27.3506432, 14.1054859, -27.1194038, 14.0935144, -41.4441566, 41.2248917
12: -45.4296455, 18.8735943, -45.3725014, 18.8221111, -62.0038910, 62.0139618
13: -37.8359528, 26.1513958, -37.8149910, 26.0282230, -63.8641739, 63.9663849
14: -52.2436829, 16.0732002, -52.1146927, 16.0667763, -68.1386566, 68.0078583
15: -21.6744766, 29.1880951, -21.6590710, 29.0704460, -50.7449226, 50.8471680
16: -35.8014832, 20.8039055, -35.7038574, 20.7820511, -55.5402374, 55.4559326
17: -48.8464890, 12.4760885, -48.6270905, 12.4629965, -61.3094864, 61.1031799
18: -22.0162525, 30.1204662, -21.9414310, 30.1052246, -52.1214752, 52.0618973
19: -17.6500893, 19.9684048, -17.5024014, 19.9674282, -37.6175156, 37.4708061
20: -24.8923626, 14.0682793, -24.8066139, 14.0582685, -38.9506302, 38.8748932
21: -22.8614597, 21.9423084, -22.6922588, 21.9399834, -44.8014450, 44.6345673
22: -13.9519062, 27.4266739, -13.8819704, 27.4196892, -37.2726212, 37.2254944
23: -16.3941841, 22.9319458, -16.2347813, 22.9223862, -39.3165703, 39.1667252
24: -16.5577221, 23.8793449, -16.4168053, 23.8729782, -40.4307022, 40.2961502
25: -16.7898903, 28.0320816, -16.6558189, 28.0237198, -43.3339310, 43.2147141
26: -27.2006035, 35.4512215, -27.1031208, 35.4308281, -62.6314316, 62.5543442
27: -22.0723000, 19.6025810, -22.0025387, 19.5932961, -41.6655960, 41.6051178
28: -17.2621384, 25.1257801, -17.1347771, 25.1157665, -42.1401978, 42.0256271
29: -10.0136862, 22.7360210, -9.8739023, 22.7290459, -30.7878380, 30.6561203
30: -29.1468182, 15.6317291, -28.9557762, 15.6258621, -44.7726822, 44.5875053
31: -23.9686699, 24.2403278, -23.8021069, 24.2341805, -48.2028503, 48.0424347
32: -47.7563438, -1.7771959, -47.7250137, -1.8878198, -40.7873993, 40.8905258
33: -70.7192230, 5.6408100, -70.6765976, 5.5235147, -67.1873627, 67.2719574
34: -63.6759453, -5.5083184, -63.6484680, -5.6123033, -46.2772064, 46.3750687
35: -41.9851494, 14.4080343, -41.9558411, 14.3348427, -53.8161621, 53.8742218
36: -42.4258041, 16.2269402, -42.4033508, 16.1297112, -58.0291290, 58.1054993
37: -71.0341339, -1.1886482, -70.9755554, -1.2272282, -66.5152740, 66.5021667
38: -59.5912094, 8.8890429, -59.5667000, 8.7761898, -68.3674011, 68.4557419
39: -74.2827225, 1.2207437, -74.2415009, 1.0963640, -74.3763580, 74.4601746
40: -70.9754105, -18.6186371, -70.9497833, -18.7323341, -41.8339767, 41.9235458
41: -47.8809052, 3.8178878, -47.8488922, 3.7551451, -46.5317307, 46.5677109
42: -46.9579239, -3.7964296, -46.9323654, -3.8291526, -39.1124954, 39.0857544

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1698
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
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1404
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
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
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
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 911
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
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 836
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
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 894
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
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1533
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1335

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0214443, upper bound: 18.0161564
time: 38.57 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0324140, upper bound: 18.0164504
time: 44.38 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -38.4466515, 25.1841278, -38.7031250, 25.2232533, -63.0466614, 63.2690582
1: -21.2760067, 21.2539940, -21.4695339, 21.2841263, -42.0156479, 42.1795883
2: -15.3251638, 23.5531006, -15.4804516, 23.5904961, -37.4642639, 37.5833817
3: -28.4355869, 18.8152046, -28.6416512, 18.8398037, -47.2753906, 47.4568558
4: -26.1955147, 22.6532288, -26.4228554, 22.6673775, -48.8628922, 49.0760841
5: -24.1251431, 24.2043076, -24.3140030, 24.2328606, -48.3580017, 48.5183105
6: -53.6621284, -5.1843243, -53.7055817, -5.1185999, -41.7717514, 41.8208542
7: -34.4289474, 15.2317696, -34.5908241, 15.2776690, -48.1462860, 48.2687988
8: -39.9764938, 17.0896416, -40.2098465, 17.1550903, -57.1036987, 57.2703857
9: -23.1502151, 25.8331375, -23.3645191, 25.8273029, -48.9775162, 49.1976547
10: -36.0698662, 24.9302979, -36.1307907, 24.9813538, -61.0512199, 61.0610886
11: -27.2612076, 14.0423164, -27.1534309, 14.1966505, -41.4578590, 41.1957474
12: -45.3758430, 18.7978821, -45.4250565, 18.9490585, -62.1116028, 61.9899750
13: -37.7821960, 26.0336819, -37.9052277, 26.0190296, -63.8012238, 63.9389114
14: -52.2143250, 16.0208359, -52.2307663, 16.0922565, -68.1615448, 68.0535889
15: -21.5941238, 29.1179962, -21.7940998, 29.1008148, -50.6949387, 50.9120941
16: -35.7387657, 20.7569885, -35.8491211, 20.7894001, -55.4806519, 55.5549850
17: -48.8072586, 12.4637756, -48.7258263, 12.5877380, -61.3949966, 61.1896019
18: -21.8841763, 30.0599365, -21.9486065, 30.2341881, -52.1183624, 52.0085449
19: -17.6048298, 19.9340782, -17.5620785, 20.0871181, -37.6919479, 37.4961548
20: -24.8459854, 14.0047112, -24.8556633, 14.1736145, -39.0195999, 38.8603745
21: -22.7800369, 21.8863983, -22.7328815, 22.0305233, -44.8105621, 44.6192780
22: -13.8990202, 27.3970585, -13.9184666, 27.5296936, -37.3365326, 37.2350769
23: -16.3264561, 22.8661919, -16.2792206, 23.0441971, -39.3706512, 39.1454124
24: -16.4854374, 23.8413506, -16.4574451, 23.9867744, -40.4674072, 40.2895737
25: -16.7522850, 27.9813194, -16.7381535, 28.1834393, -43.4605713, 43.2818222
26: -27.1065235, 35.3799286, -27.1374016, 35.5796852, -62.6862106, 62.5173302
27: -22.0070610, 19.5858231, -22.0104218, 19.6457500, -41.6528091, 41.5962448
28: -17.1954556, 25.0672150, -17.1753998, 25.2466965, -42.2086029, 42.0005836
29: -9.9598103, 22.7121773, -9.9132690, 22.8141136, -30.8187141, 30.6837502
30: -29.0515633, 15.5697832, -28.9784145, 15.7302656, -44.7818298, 44.5481987
31: -23.8991070, 24.1948757, -23.8732605, 24.3863621, -48.2854691, 48.0681381
32: -47.6950836, -1.8274894, -47.7363434, -1.8292947, -40.8187714, 40.8692551
33: -70.6674957, 5.6173553, -70.6885223, 5.6053753, -67.2198792, 67.2626343
34: -63.5877113, -5.5615344, -63.6304817, -5.5201287, -46.2756577, 46.3132248
35: -41.9206734, 14.3776798, -41.9495392, 14.4114790, -53.7729950, 53.8497009
36: -42.3872528, 16.1989136, -42.4191475, 16.2209358, -58.0894318, 58.1012878
37: -70.9840240, -1.2227907, -71.0298080, -1.1236715, -66.4969025, 66.4960938
38: -59.5228043, 8.8458939, -59.6051369, 8.8762302, -68.3990326, 68.4510345
39: -74.2611084, 1.1920996, -74.2930984, 1.1550546, -74.4369049, 74.4808502
40: -70.9280243, -18.6423969, -70.9642410, -18.6851387, -41.8459244, 41.9097214
41: -47.8166885, 3.7850127, -47.8437881, 3.7942710, -46.5180817, 46.5293427
42: -46.9209862, -3.8313789, -46.9495926, -3.7669578, -39.2162933, 39.1054916

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 906
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
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1404
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
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 792
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
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 945
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
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 894
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
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 897

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 1335

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9965380, upper bound: 18.0224946
time: 35.41 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0074334, upper bound: 18.0227456
time: 42.64 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -38.5716400, 25.3121071, -38.7254639, 25.2963772, -63.2475739, 63.4037628
1: -21.3510036, 21.3522949, -21.4801559, 21.3386745, -42.1443863, 42.2933807
2: -15.3847866, 23.6500206, -15.4859295, 23.6449776, -37.5710144, 37.6765289
3: -28.4886684, 18.9061127, -28.6465034, 18.8906364, -47.3793030, 47.5526161
4: -26.2813492, 22.7655563, -26.4304810, 22.7314682, -49.0128174, 49.1960373
5: -24.1915359, 24.2917595, -24.3199615, 24.2827072, -48.4742432, 48.6117210
6: -53.7517166, -5.1327486, -53.7555275, -5.1103592, -41.8296585, 41.7940216
7: -34.4907608, 15.3163013, -34.5983467, 15.3253145, -48.2637482, 48.3660965
8: -40.0621338, 17.2431946, -40.2149048, 17.2426300, -57.2790833, 57.4315033
9: -23.2361603, 25.9326954, -23.3739281, 25.8834705, -49.1196289, 49.3066254
10: -36.1121750, 24.9962845, -36.1435547, 25.0126038, -61.1247787, 61.1398392
11: -27.3849144, 14.1105976, -27.2218227, 14.2030315, -41.5879440, 41.3324203
12: -45.4758759, 18.8784447, -45.4810867, 18.9603539, -62.2017670, 62.1261292
13: -37.8436775, 26.1666660, -37.9178085, 26.0885963, -63.9322739, 64.0844727
14: -52.2754173, 16.0762520, -52.2472305, 16.1205692, -68.2529907, 68.1312408
15: -21.6799145, 29.2276878, -21.8043251, 29.1619358, -50.8418503, 51.0320129
16: -35.8151703, 20.8168659, -35.8692398, 20.8216896, -55.5910416, 55.6353378
17: -48.8845711, 12.4816904, -48.7580261, 12.5937490, -61.4783211, 61.2397156
18: -22.0465107, 30.1259613, -22.0356445, 30.2385998, -52.2851105, 52.1616058
19: -17.6917267, 19.9694195, -17.6073418, 20.0879021, -37.7796288, 37.5767593
20: -24.9327507, 14.0727348, -24.9013805, 14.1827850, -39.1155357, 38.9741135
21: -22.8964996, 21.9449501, -22.7927227, 22.0337276, -44.9302292, 44.7376709
22: -13.9812298, 27.4292240, -13.9605665, 27.5317097, -37.4177399, 37.2944565
23: -16.4407673, 22.9354458, -16.3407764, 23.0513153, -39.4920807, 39.2762222
24: -16.6022892, 23.8823223, -16.5214691, 23.9908276, -40.5931168, 40.4037933
25: -16.8433380, 28.0355721, -16.7854595, 28.1889935, -43.5557175, 43.3431015
26: -27.2395439, 35.4579849, -27.2084160, 35.5859070, -62.8254509, 62.6664009
27: -22.0944805, 19.6067123, -22.0572567, 19.6539116, -41.7483902, 41.6639709
28: -17.3068161, 25.1305103, -17.2357235, 25.2546711, -42.3253098, 42.1265526
29: -10.0423374, 22.7386131, -9.9558811, 22.8169098, -30.9048538, 30.7340622
30: -29.1867008, 15.6359463, -29.0531960, 15.7405472, -44.9272461, 44.6891403
31: -24.0283718, 24.2437248, -23.9422455, 24.3895035, -48.4178772, 48.1859703
32: -47.7802315, -1.7726989, -47.7833900, -1.8133602, -40.8960419, 40.9528885
33: -70.7332230, 5.6476717, -70.7211151, 5.6149502, -67.3018188, 67.3281097
34: -63.6876793, -5.5032735, -63.6846275, -5.5115495, -46.4050980, 46.4259262
35: -41.9934845, 14.4130182, -41.9877396, 14.4174452, -53.9043732, 53.9210281
36: -42.4448738, 16.2309875, -42.4505920, 16.2263107, -58.1501770, 58.1559296
37: -71.0723877, -1.1828756, -71.0765762, -1.1171513, -66.6709442, 66.6013947
38: -59.6224022, 8.8966522, -59.6565475, 8.8896790, -68.5120850, 68.5531998
39: -74.3030853, 1.2242584, -74.3131256, 1.1697783, -74.4714966, 74.5323334
40: -70.9875183, -18.6116982, -70.9936905, -18.6722679, -41.9170837, 41.9732895
41: -47.8941574, 3.8231225, -47.8875122, 3.8043723, -46.6158676, 46.6129379
42: -46.9740448, -3.7921376, -46.9786644, -3.7562642, -39.2552643, 39.1312714

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1698
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
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1404
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
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 792
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
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 820
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
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 894
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
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1533
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1335

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0214443, upper bound: 18.0321310
time: 57.81 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0324140, upper bound: 18.0324140
time: 45.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 105.28 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -17.9965380, upper bound: 17.9886508
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -18.0074334, upper bound: 17.9889105
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -18.0214443, upper bound: 17.9983100
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -18.0324140, upper bound: 17.9985973
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -17.9965380, upper bound: 18.0045598
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -18.0074334, upper bound: 18.0048071
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -18.0214443, upper bound: 18.0142684
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -18.0324140, upper bound: 18.0145462
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -17.9965380, upper bound: 18.0065851
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -18.0074334, upper bound: 18.0068438
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -18.0214443, upper bound: 18.0161564
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -18.0324140, upper bound: 18.0164504
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -17.9965380, upper bound: 18.0224946
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -18.0074334, upper bound: 18.0227456
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -18.0214443, upper bound: 18.0321310
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 105.28
Output dim: 29, lower bound: -18.0324140, upper bound: 18.0324140

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -38.3891563, 25.0859718, -38.4827957, 25.0971909, -62.8651123, 62.9469452
1: -21.2401257, 21.1666107, -21.3026047, 21.1714363, -41.8691940, 41.9208145
2: -15.3071156, 23.4796829, -15.3577805, 23.4889145, -37.3448792, 37.3805771
3: -28.4175701, 18.6985359, -28.4636612, 18.6869736, -47.1045456, 47.1621971
4: -26.1729279, 22.5278854, -26.2478161, 22.5277920, -48.7007217, 48.7757034
5: -24.1069069, 24.1173935, -24.1669350, 24.1170769, -48.2239838, 48.2843285
6: -53.5898781, -5.2339420, -53.5970001, -5.2108145, -41.5600891, 41.6732826
7: -34.3797073, 15.1781492, -34.4322395, 15.1827879, -48.0030823, 48.0506744
8: -39.9423370, 16.9849892, -40.0079269, 17.0055542, -56.9183044, 56.9608307
9: -23.0863724, 25.7227077, -23.1699333, 25.7292118, -48.8155823, 48.8926392
10: -35.9749756, 24.8863106, -35.9895782, 24.9125938, -60.8875694, 60.8758888
11: -26.9962063, 13.9843102, -27.0204067, 14.0590067, -41.0552139, 41.0047150
12: -45.2722626, 18.7336273, -45.2684097, 18.7825203, -61.8382721, 61.7804565
13: -37.6872864, 25.8948936, -37.7245407, 25.9170074, -63.6042938, 63.6194344
14: -52.0591202, 15.9803019, -52.0691376, 16.0090523, -67.8927307, 67.8618469
15: -21.5458927, 28.9692345, -21.6205807, 28.9822903, -50.5281830, 50.5898132
16: -35.6109695, 20.7199249, -35.6565437, 20.7274132, -55.2882233, 55.3217239
17: -48.5538597, 12.4004822, -48.5687141, 12.4270306, -60.9808884, 60.9691963
18: -21.7858067, 30.0143471, -21.8243656, 30.0621643, -51.8479691, 51.8387146
19: -17.4324608, 19.8844833, -17.4339714, 19.9301682, -37.3626289, 37.3184547
20: -24.7379513, 13.9726353, -24.7383022, 14.0251675, -38.7631187, 38.7109375
21: -22.5788078, 21.8346901, -22.6005077, 21.8981686, -44.4769745, 44.4351959
22: -13.8070784, 27.3649673, -13.8092937, 27.3802490, -37.0944290, 37.1008148
23: -16.1427269, 22.8138847, -16.1482162, 22.8834667, -39.0261917, 38.9621010
24: -16.3181610, 23.7987251, -16.3196049, 23.8338337, -40.1367798, 40.1112213
25: -16.5900345, 27.9256840, -16.5760345, 27.9708786, -43.0807114, 43.0662460
26: -26.9789505, 35.3339500, -26.9920063, 35.3908195, -62.3697701, 62.3259583
27: -21.9188461, 19.5522003, -21.9241886, 19.5524616, -41.4713058, 41.4763870
28: -17.0460472, 25.0227394, -17.0469761, 25.0703773, -41.8820724, 41.8339043
29: -9.7961807, 22.6739197, -9.8036079, 22.6977844, -30.5360603, 30.5393867
30: -28.8306179, 15.5261335, -28.8445568, 15.5904932, -44.4211121, 44.3706894
31: -23.6983585, 24.1260815, -23.7020187, 24.1788177, -47.8771744, 47.8281021
32: -47.5958290, -1.9522567, -47.6113777, -1.9366322, -40.6025848, 40.6199646
33: -70.5525208, 5.4924688, -70.5761414, 5.4926653, -66.9866638, 67.0205231
34: -63.5076447, -5.6785107, -63.5340958, -5.6469655, -46.0509796, 46.1035461
35: -41.8460121, 14.2980118, -41.8658180, 14.3110647, -53.6039429, 53.6728210
36: -42.3024063, 16.0907669, -42.3084335, 16.0980873, -57.8756409, 57.8834991
37: -70.8662567, -1.2676640, -70.8617477, -1.2493563, -66.2423706, 66.2903900
38: -59.4425812, 8.7166796, -59.4572105, 8.7292957, -68.1718750, 68.1738892
39: -74.1240997, 1.0573487, -74.1222000, 1.0477562, -74.1873779, 74.1733704
40: -70.8484039, -18.7652283, -70.8643875, -18.7672729, -41.6755676, 41.6927261
41: -47.7343597, 3.7116861, -47.7412262, 3.7149677, -46.3323364, 46.3537979
42: -46.8601112, -3.8739719, -46.8562202, -3.8655577, -38.9719315, 38.9927979

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
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
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
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
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
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
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 782
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
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1766
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9965199, upper bound: 17.9756467
time: 62.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9965200, upper bound: 17.9886326
time: 50.27 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -38.3918915, 25.0927620, -38.4884186, 25.1089954, -62.8808441, 62.9590912
1: -21.2411938, 21.1721115, -21.3053703, 21.1827869, -41.8815613, 41.9289474
2: -15.3078289, 23.4838181, -15.3597040, 23.4986095, -37.3573990, 37.3861618
3: -28.4210320, 18.7028580, -28.4703922, 18.7005424, -47.1215744, 47.1732483
4: -26.1744003, 22.5389252, -26.2562752, 22.5494308, -48.7238312, 48.7952003
5: -24.1082134, 24.1210213, -24.1663742, 24.1271420, -48.2353554, 48.2873955
6: -53.6084747, -5.2320442, -53.6293068, -5.1919022, -41.6001816, 41.6944885
7: -34.3809853, 15.1820307, -34.4347954, 15.1923428, -48.0180969, 48.0588989
8: -39.9439774, 16.9971485, -40.0224113, 17.0287743, -56.9428101, 56.9923248
9: -23.0923309, 25.7267704, -23.1822701, 25.7452106, -48.8375397, 48.9090424
10: -35.9831657, 24.8891869, -36.0059357, 24.9286461, -60.9118118, 60.8951225
11: -27.0062828, 13.9854450, -27.0412235, 14.0688591, -41.0751419, 41.0266685
12: -45.2924881, 18.7365532, -45.3030281, 18.8049088, -61.8681793, 61.8014069
13: -37.7139587, 25.8988724, -37.7721024, 25.9516850, -63.6656418, 63.6709747
14: -52.0633583, 15.9889288, -52.0904198, 16.0252991, -67.9108429, 67.8953857
15: -21.5488377, 28.9783897, -21.6359711, 29.0007133, -50.5495529, 50.6143608
16: -35.6181488, 20.7247963, -35.6721802, 20.7466927, -55.3161163, 55.3385391
17: -48.5587387, 12.4058743, -48.5860634, 12.4368439, -60.9955826, 60.9919357
18: -21.7893085, 30.0288773, -21.8492126, 30.0890980, -51.8784065, 51.8780899
19: -17.4362984, 19.8932476, -17.4508896, 19.9452248, -37.3815231, 37.3441391
20: -24.7413940, 13.9822769, -24.7555771, 14.0417385, -38.7831345, 38.7378540
21: -22.5838737, 21.8452435, -22.6228199, 21.9164276, -44.5003014, 44.4680634
22: -13.8116865, 27.3798370, -13.8339205, 27.4061489, -37.1148224, 37.1404877
23: -16.1455154, 22.8208237, -16.1671562, 22.8965702, -39.0420837, 38.9879799
24: -16.3233624, 23.8107834, -16.3462410, 23.8546429, -40.1547394, 40.1516113
25: -16.5935516, 27.9421692, -16.6023693, 27.9998550, -43.1050415, 43.1095657
26: -26.9859447, 35.3464241, -27.0253963, 35.4139099, -62.3998566, 62.3718185
27: -21.9230709, 19.5660458, -21.9500504, 19.5764313, -41.4995041, 41.5160980
28: -17.0491619, 25.0354519, -17.0693665, 25.0936089, -41.9046021, 41.8701172
29: -9.8012447, 22.6826782, -9.8251286, 22.7134018, -30.5530777, 30.5726776
30: -28.8370171, 15.5296326, -28.8727341, 15.5982456, -44.4352646, 44.4023666
31: -23.7032032, 24.1404114, -23.7246609, 24.2036095, -47.9068146, 47.8650742
32: -47.6189613, -1.9492950, -47.6524506, -1.9089932, -40.6558990, 40.6485672
33: -70.5729980, 5.4939899, -70.6141891, 5.5118265, -67.0298920, 67.0462799
34: -63.5290604, -5.6764045, -63.5723801, -5.6255445, -46.1034393, 46.1169434
35: -41.8639679, 14.2988873, -41.8984604, 14.3264151, -53.6392822, 53.6977921
36: -42.3258820, 16.0922585, -42.3497391, 16.1195259, -57.9220123, 57.9241638
37: -70.8890457, -1.2665586, -70.9031830, -1.2362566, -66.2793427, 66.3279114
38: -59.4661827, 8.7189426, -59.4984436, 8.7562637, -68.2224426, 68.2173843
39: -74.1560287, 1.0587687, -74.1801224, 1.0784817, -74.2524567, 74.2287598
40: -70.8647919, -18.7638664, -70.8941574, -18.7487011, -41.7099762, 41.7117386
41: -47.7596436, 3.7140512, -47.7847290, 3.7406425, -46.3863220, 46.3848877
42: -46.8802071, -3.8712988, -46.8904877, -3.8439946, -39.0150909, 39.0070877

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
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
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
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
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 836
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
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 848
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
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 782
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
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1766
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0074155, upper bound: 17.9759013
time: 48.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0074155, upper bound: 17.9888924
time: 65.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -38.5141830, 25.2139473, -38.5051498, 25.1703472, -63.0659943, 63.0816803
1: -21.3152065, 21.2649422, -21.3132381, 21.2259293, -41.9980164, 42.0345840
2: -15.3667841, 23.5765915, -15.3632908, 23.5433388, -37.4515839, 37.4737549
3: -28.4706268, 18.7894306, -28.4685593, 18.7377987, -47.2084274, 47.2579880
4: -26.2587776, 22.6401482, -26.2554722, 22.5918732, -48.8506508, 48.8956223
5: -24.1733322, 24.2047539, -24.1729355, 24.1669006, -48.3402328, 48.3776894
6: -53.6794128, -5.1822920, -53.6469498, -5.2026443, -41.6179276, 41.6464119
7: -34.4415817, 15.2626381, -34.4398499, 15.2304125, -48.1205292, 48.1480713
8: -40.0280037, 17.1384869, -40.0130539, 17.0930367, -57.0936890, 57.1219177
9: -23.1724510, 25.8222675, -23.1794624, 25.7853622, -48.9578133, 49.0017319
10: -36.0173492, 24.9522400, -36.0023575, 24.9438629, -60.9612122, 60.9545975
11: -27.1198559, 14.0526028, -27.0888214, 14.0654106, -41.1852646, 41.1414261
12: -45.3722916, 18.8142319, -45.3244781, 18.7938938, -61.9283752, 61.9166412
13: -37.7489204, 26.0278549, -37.7371445, 25.9865570, -63.7354774, 63.7649994
14: -52.1201248, 16.0356522, -52.0854225, 16.0373859, -67.9840851, 67.9394836
15: -21.6316433, 29.0789261, -21.6307716, 29.0434475, -50.6750908, 50.7096977
16: -35.6874924, 20.7797947, -35.6766891, 20.7596111, -55.3987427, 55.4020081
17: -48.6311607, 12.4183636, -48.6008720, 12.4331150, -61.0642776, 61.0192337
18: -21.9481239, 30.0804043, -21.9113884, 30.0665703, -52.0146942, 51.9917908
19: -17.5193195, 19.9198418, -17.4792480, 19.9310017, -37.4503212, 37.3990898
20: -24.8246994, 14.0406904, -24.7840099, 14.0343914, -38.8590927, 38.8246994
21: -22.6952057, 21.8932076, -22.6604023, 21.9013729, -44.5965805, 44.5536118
22: -13.8892574, 27.3971100, -13.8513317, 27.3822479, -37.1755524, 37.1601791
23: -16.2569828, 22.8831615, -16.2097874, 22.8905830, -39.1475677, 39.0929489
24: -16.4349804, 23.8397503, -16.3835869, 23.8379192, -40.2728996, 40.2233353
25: -16.6811333, 27.9800148, -16.6233597, 27.9764023, -43.1758652, 43.1275864
26: -27.1120186, 35.4120598, -27.0629368, 35.3971481, -62.5091667, 62.4749985
27: -22.0061607, 19.5730171, -21.9709797, 19.5605850, -41.5667458, 41.5439987
28: -17.1573772, 25.0860615, -17.1072903, 25.0783501, -41.9987946, 41.9599075
29: -9.8785791, 22.7003822, -9.8461952, 22.7005939, -30.6221123, 30.5896988
30: -28.9657440, 15.5923052, -28.9193192, 15.6007919, -44.5665359, 44.5116234
31: -23.8275814, 24.1749935, -23.7710228, 24.1819553, -48.0095367, 47.9460144
32: -47.6809387, -1.8973875, -47.6583748, -1.9206982, -40.6798096, 40.7036362
33: -70.6181946, 5.5228291, -70.6087952, 5.5023308, -67.0683289, 67.0859985
34: -63.6075897, -5.6202145, -63.5881958, -5.6384335, -46.1803818, 46.2164688
35: -41.9187546, 14.3333168, -41.9040260, 14.3169718, -53.7352295, 53.7442169
36: -42.3600464, 16.1228561, -42.3398972, 16.1034546, -57.9363861, 57.9381638
37: -70.9543533, -1.2277117, -70.9085007, -1.2429018, -66.4161377, 66.3956299
38: -59.5421600, 8.7677031, -59.5086594, 8.7427959, -68.2849579, 68.2763596
39: -74.1661530, 1.0894203, -74.1421280, 1.0623617, -74.2217712, 74.2248077
40: -70.9077606, -18.7345467, -70.8937836, -18.7543087, -41.7466125, 41.7562180
41: -47.8118210, 3.7498302, -47.7849083, 3.7250934, -46.4301376, 46.4373932
42: -46.9131241, -3.8347144, -46.8852768, -3.8549423, -39.0108795, 39.0185318

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
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
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1404
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
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
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
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 836
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
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1766
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0214291, upper bound: 17.9853360
time: 57.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0214291, upper bound: 17.9982931
time: 43.46 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -38.5169601, 25.2207279, -38.5108414, 25.1821251, -63.0817108, 63.0937347
1: -21.3162785, 21.2704697, -21.3159790, 21.2373104, -42.0104218, 42.0427704
2: -15.3674765, 23.5807381, -15.3651676, 23.5530586, -37.4641190, 37.4793320
3: -28.4740906, 18.7937412, -28.4752922, 18.7514153, -47.2255058, 47.2690353
4: -26.2602959, 22.6511841, -26.2638760, 22.6135178, -48.8738136, 48.9150620
5: -24.1746254, 24.2083855, -24.1723442, 24.1769524, -48.3515778, 48.3807297
6: -53.6979866, -5.1803999, -53.6791916, -5.1836557, -41.6580200, 41.6676407
7: -34.4428787, 15.2664909, -34.4423866, 15.2400055, -48.1356506, 48.1562653
8: -40.0296402, 17.1506996, -40.0274734, 17.1162834, -57.1181488, 57.1533813
9: -23.1783829, 25.8263702, -23.1917324, 25.8013496, -48.9797325, 49.0181046
10: -36.0255470, 24.9551182, -36.0186691, 24.9598789, -60.9854279, 60.9737854
11: -27.1299152, 14.0537548, -27.1095943, 14.0752525, -41.2051697, 41.1633492
12: -45.3925095, 18.8170891, -45.3590164, 18.8163185, -61.9583740, 61.9376068
13: -37.7755852, 26.0317612, -37.7846832, 26.0212212, -63.7968063, 63.8164444
14: -52.1244698, 16.0443687, -52.1066971, 16.0536003, -68.0021973, 67.9729309
15: -21.6345558, 29.0880909, -21.6461868, 29.0617676, -50.6963234, 50.7342758
16: -35.6946259, 20.7846718, -35.6923676, 20.7788887, -55.4266357, 55.4188538
17: -48.6360359, 12.4237843, -48.6181946, 12.4429264, -61.0789642, 61.0419769
18: -21.9515915, 30.0949326, -21.9361858, 30.0935173, -52.0451088, 52.0311203
19: -17.5232067, 19.9285698, -17.4961376, 19.9460564, -37.4692612, 37.4247055
20: -24.8281593, 14.0503139, -24.8012657, 14.0509453, -38.8791046, 38.8515778
21: -22.7002926, 21.9037590, -22.6826839, 21.9196167, -44.6199112, 44.5864410
22: -13.8938780, 27.4119625, -13.8759899, 27.4081879, -37.1959686, 37.1998596
23: -16.2597961, 22.8900871, -16.2287483, 22.9037170, -39.1635132, 39.1188354
24: -16.4401684, 23.8517990, -16.4102669, 23.8587189, -40.2914810, 40.2620659
25: -16.6846046, 27.9964218, -16.6496964, 28.0053825, -43.2002563, 43.1708908
26: -27.1189137, 35.4245262, -27.0963230, 35.4201927, -62.5391083, 62.5208511
27: -22.0103836, 19.5869007, -21.9968300, 19.5846100, -41.5949936, 41.5837326
28: -17.1605015, 25.0987911, -17.1296806, 25.1015816, -42.0213623, 41.9960747
29: -9.8836460, 22.7091351, -9.8677340, 22.7162056, -30.6390877, 30.6229782
30: -28.9721203, 15.5957947, -28.9474297, 15.6085176, -44.5806389, 44.5432243
31: -23.8324337, 24.1893234, -23.7936096, 24.2066879, -48.0391235, 47.9829330
32: -47.7040634, -1.8944392, -47.6993866, -1.8931112, -40.7331619, 40.7322464
33: -70.6387024, 5.5243101, -70.6467743, 5.5213537, -67.1115570, 67.1118011
34: -63.6290207, -5.6181340, -63.6264229, -5.6170740, -46.2328720, 46.2297974
35: -41.9367561, 14.3341990, -41.9366798, 14.3324490, -53.7705536, 53.7692261
36: -42.3835220, 16.1243382, -42.3811989, 16.1249084, -57.9828033, 57.9787445
37: -70.9772263, -1.2266750, -70.9499283, -1.2298422, -66.4532318, 66.4331207
38: -59.5658493, 8.7699070, -59.5498657, 8.7697048, -68.3355560, 68.3197708
39: -74.1980972, 1.0908484, -74.2000732, 1.0930872, -74.2868347, 74.2801208
40: -70.9241333, -18.7332096, -70.9235764, -18.7358170, -41.7810364, 41.7752457
41: -47.8370590, 3.7521873, -47.8283691, 3.7507524, -46.4840698, 46.4685364
42: -46.9332275, -3.8320365, -46.9195251, -3.8333716, -39.0540695, 39.0328026

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
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
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
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
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 836
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
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1766
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0323986, upper bound: 17.9856141
time: 85.40 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0323986, upper bound: 17.9985805
time: 52.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -38.4026413, 25.1411839, -38.6922531, 25.2067070, -62.9840698, 63.2138824
1: -21.2449150, 21.2137108, -21.4665833, 21.2683525, -41.9673157, 42.1329041
2: -15.3119154, 23.5208206, -15.4786530, 23.5769806, -37.4342194, 37.5437698
3: -28.4198647, 18.7595520, -28.6315117, 18.8217602, -47.2416229, 47.3910637
4: -26.1778526, 22.5812092, -26.4124489, 22.6375618, -48.8154144, 48.9936600
5: -24.1102543, 24.1635857, -24.3126888, 24.2174149, -48.3276672, 48.4762726
6: -53.6140823, -5.2282248, -53.6587677, -5.1408806, -41.6898575, 41.7391434
7: -34.3829308, 15.2182312, -34.5854797, 15.2670259, -48.0836182, 48.2457275
8: -39.9468307, 17.0402946, -40.1920776, 17.1245022, -57.0359802, 57.2010880
9: -23.0950966, 25.7590256, -23.3398018, 25.8080330, -48.9031296, 49.0988274
10: -35.9889183, 24.8936081, -36.1049271, 24.9605350, -60.9494553, 60.9985352
11: -27.0304928, 13.9894390, -27.1228218, 14.1685438, -41.1990356, 41.1122589
12: -45.3185577, 18.7385597, -45.3769836, 18.9208279, -62.0361938, 61.8927155
13: -37.6950874, 25.9101868, -37.8273697, 25.9774227, -63.6725082, 63.7375565
14: -52.0907898, 15.9833746, -52.2014694, 16.0628166, -68.0068665, 67.9851074
15: -21.5512772, 29.0088348, -21.7657814, 29.0738049, -50.6250839, 50.7746162
16: -35.6246872, 20.7328682, -35.8219452, 20.7669888, -55.3390808, 55.5010834
17: -48.5918999, 12.4060888, -48.6996040, 12.5578632, -61.1497650, 61.1056938
18: -21.8160534, 30.0198898, -21.9186077, 30.1954918, -52.0115433, 51.9384995
19: -17.4740753, 19.8854980, -17.5389309, 20.0507164, -37.5247917, 37.4244308
20: -24.7782974, 13.9771366, -24.8330879, 14.1497517, -38.9280472, 38.8102264
21: -22.6137867, 21.8372974, -22.7009830, 21.9919853, -44.6057739, 44.5382805
22: -13.8364172, 27.3675461, -13.8878450, 27.4922333, -37.2395859, 37.1698303
23: -16.1892681, 22.8174305, -16.2541981, 23.0123730, -39.2016411, 39.0716286
24: -16.3626823, 23.8017349, -16.4242516, 23.9517403, -40.3017578, 40.2130280
25: -16.6435127, 27.9291744, -16.7056999, 28.1361485, -43.3025208, 43.1946793
26: -27.0179062, 35.3407478, -27.0973301, 35.5459290, -62.5638351, 62.4380798
27: -21.9409790, 19.5563087, -21.9788799, 19.6130753, -41.5540543, 41.5351868
28: -17.0907364, 25.0274601, -17.1479454, 25.2092438, -42.0672150, 41.9349060
29: -9.8248081, 22.6764908, -9.8855267, 22.7856731, -30.6530685, 30.6172409
30: -28.8704872, 15.5303459, -28.9419193, 15.7052135, -44.5756989, 44.4722672
31: -23.7579937, 24.1295395, -23.8421459, 24.3341637, -48.0921555, 47.9716873
32: -47.6197014, -1.9477091, -47.6697159, -1.8621454, -40.7111969, 40.6822739
33: -70.5665588, 5.4992924, -70.6206055, 5.5842323, -67.1010284, 67.0766449
34: -63.5193863, -5.6734757, -63.5702705, -5.5462513, -46.1788483, 46.1544876
35: -41.8543167, 14.3030052, -41.8976936, 14.3935919, -53.6921387, 53.7195663
36: -42.3215179, 16.0948257, -42.3556519, 16.1947155, -57.9967041, 57.9338989
37: -70.9043808, -1.2618666, -70.9627457, -1.1394215, -66.3979034, 66.3895721
38: -59.4737740, 8.7243099, -59.5470734, 8.8428020, -68.3165741, 68.2713852
39: -74.1444855, 1.0608320, -74.1937866, 1.1211119, -74.2822571, 74.2454376
40: -70.8604431, -18.7582741, -70.9082184, -18.7071171, -41.7586212, 41.7424011
41: -47.7476692, 3.7169037, -47.7798080, 3.7642403, -46.4164581, 46.3990097
42: -46.8762550, -3.8696947, -46.9025040, -3.7927389, -39.1147003, 39.0382881

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
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
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 836
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
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1766
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9965199, upper bound: 17.9915270
time: 48.99 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9965200, upper bound: 18.0045420
time: 45.29 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -38.4053879, 25.1479435, -38.6978912, 25.2184753, -62.9998627, 63.2258911
1: -21.2460480, 21.2192268, -21.4694443, 21.2797165, -41.9796600, 42.1410370
2: -15.3126545, 23.5249100, -15.4805012, 23.5867233, -37.4467468, 37.5493240
3: -28.4233456, 18.7638321, -28.6382828, 18.8354244, -47.2587700, 47.4021149
4: -26.1793537, 22.5922337, -26.4208755, 22.6591778, -48.8385315, 49.0131073
5: -24.1115417, 24.1672249, -24.3121071, 24.2275047, -48.3390465, 48.4793320
6: -53.6326866, -5.2263250, -53.6911049, -5.1218967, -41.7299347, 41.7603455
7: -34.3842278, 15.2221308, -34.5880089, 15.2765846, -48.0986633, 48.2538681
8: -39.9484482, 17.0524330, -40.2065544, 17.1477280, -57.0604095, 57.2325897
9: -23.1010704, 25.7630844, -23.3520908, 25.8240414, -48.9251099, 49.1151733
10: -35.9970932, 24.8964672, -36.1212196, 24.9765682, -60.9736633, 61.0176849
11: -27.0405331, 13.9905624, -27.1436100, 14.1784058, -41.2189407, 41.1341705
12: -45.3387794, 18.7413464, -45.4115753, 18.9432240, -62.0660400, 61.9137115
13: -37.7217178, 25.9141197, -37.8749199, 26.0120888, -63.7338066, 63.7890396
14: -52.0950432, 15.9920712, -52.2227821, 16.0790558, -68.0249786, 68.0187073
15: -21.5541744, 29.0179825, -21.7811852, 29.0921555, -50.6463318, 50.7991676
16: -35.6318130, 20.7377281, -35.8375473, 20.7862663, -55.3670044, 55.5179291
17: -48.5968246, 12.4115143, -48.7169571, 12.5676785, -61.1645050, 61.1284714
18: -21.8195343, 30.0343971, -21.9434319, 30.2224407, -52.0419769, 51.9778290
19: -17.4779472, 19.8942299, -17.5558205, 20.0657711, -37.5437164, 37.4500504
20: -24.7817974, 13.9867496, -24.8503799, 14.1663046, -38.9481010, 38.8371277
21: -22.6188774, 21.8478813, -22.7233086, 22.0101967, -44.6290741, 44.5711899
22: -13.8410358, 27.3824158, -13.9125004, 27.5181694, -37.2599716, 37.2094727
23: -16.1920967, 22.8243217, -16.2731609, 23.0254936, -39.2175903, 39.0974808
24: -16.3678703, 23.8138065, -16.4508896, 23.9725342, -40.3197556, 40.2534256
25: -16.6469707, 27.9456673, -16.7320366, 28.1651459, -43.3269043, 43.2379837
26: -27.0248737, 35.3531647, -27.1306763, 35.5690117, -62.5938873, 62.4838409
27: -21.9452133, 19.5701637, -22.0047417, 19.6370678, -41.5822830, 41.5749054
28: -17.0938797, 25.0401649, -17.1703129, 25.2324619, -42.0897141, 41.9710732
29: -9.8298702, 22.6852646, -9.9070616, 22.8012810, -30.6700363, 30.6505661
30: -28.8768501, 15.5338621, -28.9701462, 15.7129622, -44.5898132, 44.5040092
31: -23.7629185, 24.1438274, -23.8648033, 24.3589172, -48.1218338, 48.0086288
32: -47.6428680, -1.9448109, -47.7106972, -1.8345103, -40.7645187, 40.7108841
33: -70.5870209, 5.5008011, -70.6586304, 5.6032982, -67.1442261, 67.1024323
34: -63.5407867, -5.6713715, -63.6084976, -5.5248480, -46.2313080, 46.1679153
35: -41.8723259, 14.3038950, -41.9303665, 14.4090652, -53.7274323, 53.7445679
36: -42.3449707, 16.0963058, -42.3969727, 16.2161980, -58.0431213, 57.9745941
37: -70.9272003, -1.2607603, -71.0041275, -1.1262445, -66.4350128, 66.4270477
38: -59.4974365, 8.7265854, -59.5883446, 8.8697071, -68.3671417, 68.3149261
39: -74.1764679, 1.0622096, -74.2517395, 1.1518869, -74.3472900, 74.3007050
40: -70.8768387, -18.7569542, -70.9380417, -18.6886063, -41.7930298, 41.7614365
41: -47.7729111, 3.7192998, -47.8233109, 3.7898331, -46.4704590, 46.4300995
42: -46.8963432, -3.8670087, -46.9367790, -3.7711062, -39.1578674, 39.0525513

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
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
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 836
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
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1766
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0074155, upper bound: 17.9917707
time: 85.41 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0074155, upper bound: 18.0047893
time: 58.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -38.5276794, 25.2691574, -38.7146072, 25.2798119, -63.1849823, 63.3485260
1: -21.3200188, 21.3120327, -21.4772568, 21.3228645, -42.0961380, 42.2467041
2: -15.3715572, 23.6177330, -15.4841471, 23.6314182, -37.5409393, 37.6368256
3: -28.4729195, 18.8504486, -28.6363811, 18.8726101, -47.3455276, 47.4868317
4: -26.2637253, 22.6934738, -26.4200459, 22.7016335, -48.9653587, 49.1135178
5: -24.1766415, 24.2509727, -24.3186321, 24.2672749, -48.4439163, 48.5696030
6: -53.7036285, -5.1765728, -53.7087555, -5.1326656, -41.7476807, 41.7122955
7: -34.4447975, 15.3027439, -34.5930824, 15.3146429, -48.2010803, 48.3430023
8: -40.0324402, 17.1938515, -40.1971436, 17.2119274, -57.2113953, 57.3622131
9: -23.1811314, 25.8585987, -23.3492413, 25.8641586, -49.0452881, 49.2078400
10: -36.0312233, 24.9595222, -36.1176910, 24.9917774, -61.0230026, 61.0772133
11: -27.1540947, 14.0577259, -27.1912003, 14.1749144, -41.3290100, 41.2489243
12: -45.4186363, 18.8190346, -45.4330215, 18.9322395, -62.1262665, 62.0288849
13: -37.7566833, 26.0430851, -37.8399887, 26.0469418, -63.8036270, 63.8830719
14: -52.1518288, 16.0388145, -52.2178955, 16.0911331, -68.0982819, 68.0627136
15: -21.6369972, 29.1185780, -21.7760201, 29.1348839, -50.7718811, 50.8945999
16: -35.7011299, 20.7927437, -35.8421021, 20.7992268, -55.4494476, 55.5813599
17: -48.6692047, 12.4240103, -48.7318039, 12.5639172, -61.2331238, 61.1558151
18: -21.9783573, 30.0859566, -22.0055199, 30.1999016, -52.1782608, 52.0914764
19: -17.5609818, 19.9208546, -17.5841904, 20.0515270, -37.6125107, 37.5050430
20: -24.8650684, 14.0452099, -24.8788319, 14.1589441, -39.0240135, 38.9240417
21: -22.7302647, 21.8958549, -22.7608566, 21.9951496, -44.7254143, 44.6567116
22: -13.9185905, 27.3996773, -13.9299126, 27.4942665, -37.3207169, 37.2291946
23: -16.3035488, 22.8866653, -16.3157997, 23.0194988, -39.3230476, 39.2024651
24: -16.4795475, 23.8427258, -16.4882488, 23.9558029, -40.4353485, 40.3309746
25: -16.7345276, 27.9834881, -16.7530117, 28.1417084, -43.3977203, 43.2560196
26: -27.1509171, 35.4187241, -27.1682720, 35.5522461, -62.7031631, 62.5869980
27: -22.0283909, 19.5771332, -22.0256786, 19.6212311, -41.6496201, 41.6028137
28: -17.2020473, 25.0907612, -17.2082253, 25.2172089, -42.1839066, 42.0608597
29: -9.9072380, 22.7029648, -9.9281120, 22.7884941, -30.7391281, 30.6675682
30: -29.0055561, 15.5965357, -29.0167027, 15.7155457, -44.7210999, 44.6132393
31: -23.8872509, 24.1784286, -23.9111309, 24.3372288, -48.2244797, 48.0895615
32: -47.7048378, -1.8929038, -47.7167244, -1.8462443, -40.7884521, 40.7659912
33: -70.6322784, 5.5296879, -70.6532593, 5.5937958, -67.1827393, 67.1421661
34: -63.6193466, -5.6152248, -63.6243362, -5.5377431, -46.3082657, 46.2672729
35: -41.9270554, 14.3383770, -41.9359512, 14.3996038, -53.8234558, 53.7909851
36: -42.3791428, 16.1269302, -42.3871078, 16.2000771, -58.0575104, 57.9885635
37: -70.9926300, -1.2219381, -71.0095673, -1.1329165, -66.5717773, 66.4947968
38: -59.5734253, 8.7752819, -59.5984955, 8.8561859, -68.4296112, 68.3737793
39: -74.1865082, 1.0929766, -74.2137985, 1.1357927, -74.3167419, 74.2967682
40: -70.9198380, -18.7275734, -70.9376526, -18.6942673, -41.8296814, 41.8059235
41: -47.8251343, 3.7550335, -47.8234901, 3.7743382, -46.5142517, 46.4826355
42: -46.9292793, -3.8304286, -46.9316025, -3.7820878, -39.1536789, 39.0640526

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
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
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 823
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
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 866
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0012727
time: 44.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0142529
time: 42.62 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -38.5304451, 25.2759552, -38.7202072, 25.2916145, -63.2006226, 63.3605652
1: -21.3211079, 21.3175411, -21.4799957, 21.3342533, -42.1084900, 42.2547989
2: -15.3722744, 23.6218376, -15.4860163, 23.6411552, -37.5535126, 37.6423798
3: -28.4764366, 18.8547630, -28.6431503, 18.8862190, -47.3626556, 47.4979134
4: -26.2652264, 22.7045479, -26.4284630, 22.7232800, -48.9885063, 49.1330109
5: -24.1779690, 24.2546501, -24.3180389, 24.2773533, -48.4553223, 48.5726891
6: -53.7222672, -5.1746731, -53.7409821, -5.1137152, -41.7877960, 41.7335129
7: -34.4460907, 15.3066254, -34.5955696, 15.3242130, -48.2161865, 48.3512344
8: -40.0340805, 17.2059937, -40.2116241, 17.2351856, -57.2357635, 57.3935852
9: -23.1870766, 25.8627090, -23.3615456, 25.8801765, -49.0672531, 49.2242546
10: -36.0394211, 24.9623833, -36.1339569, 25.0078316, -61.0472527, 61.0963402
11: -27.1642036, 14.0588818, -27.2119961, 14.1848068, -41.3490105, 41.2708778
12: -45.4388351, 18.8219109, -45.4676094, 18.9546165, -62.1561890, 62.0498962
13: -37.7833176, 26.0470333, -37.8875160, 26.0815868, -63.8649063, 63.9345474
14: -52.1560822, 16.0475159, -52.2392235, 16.1074200, -68.1163635, 68.0963440
15: -21.6399536, 29.1277161, -21.7914543, 29.1532116, -50.7931671, 50.9191704
16: -35.7082443, 20.7976875, -35.8577347, 20.8185177, -55.4774933, 55.5981369
17: -48.6740952, 12.4294710, -48.7491226, 12.5736923, -61.2477875, 61.1785927
18: -21.9818535, 30.1004810, -22.0304031, 30.2268600, -52.2087135, 52.1308823
19: -17.5648441, 19.9295750, -17.6011105, 20.0665932, -37.6314392, 37.5306854
20: -24.8685169, 14.0548096, -24.8960915, 14.1754913, -39.0440063, 38.9509010
21: -22.7353497, 21.9064560, -22.7831535, 22.0133934, -44.7487411, 44.6896095
22: -13.9231958, 27.4145355, -13.9545727, 27.5201721, -37.3411179, 37.2688828
23: -16.3063412, 22.8936176, -16.3347702, 23.0326061, -39.3389473, 39.2283859
24: -16.4847298, 23.8548107, -16.5149097, 23.9765968, -40.4565277, 40.3697205
25: -16.7380295, 27.9998932, -16.7793579, 28.1706486, -43.4220352, 43.2992859
26: -27.1578236, 35.4311752, -27.2016582, 35.5752869, -62.7331085, 62.6328354
27: -22.0326138, 19.5910301, -22.0515347, 19.6452408, -41.6778564, 41.6425629
28: -17.2051849, 25.1034966, -17.2305851, 25.2404251, -42.2064362, 42.0970306
29: -9.9123220, 22.7117310, -9.9496555, 22.8040638, -30.7561264, 30.7008820
30: -29.0119572, 15.6000099, -29.0448551, 15.7232227, -44.7351799, 44.6448669
31: -23.8921509, 24.1926899, -23.9337597, 24.3619843, -48.2541351, 48.1264496
32: -47.7279739, -1.8899150, -47.7577362, -1.8186145, -40.8417664, 40.7946091
33: -70.6526871, 5.5312033, -70.6912766, 5.6128635, -67.2259674, 67.1679840
34: -63.6407356, -5.6130581, -63.6625862, -5.5163088, -46.3607559, 46.2806473
35: -41.9450455, 14.3392324, -41.9685783, 14.4150724, -53.8587952, 53.8160095
36: -42.4026184, 16.1283875, -42.4284744, 16.2215500, -58.1038971, 58.0291977
37: -71.0154114, -1.2208824, -71.0509796, -1.1197920, -66.6088562, 66.5323029
38: -59.5970497, 8.7775898, -59.6397171, 8.8831768, -68.4802246, 68.4173050
39: -74.2184906, 1.0943518, -74.2716827, 1.1665573, -74.3817749, 74.3521118
40: -70.9361725, -18.7262402, -70.9674377, -18.6757698, -41.8641052, 41.8249664
41: -47.8503609, 3.7574415, -47.8670197, 3.7999792, -46.5682602, 46.5137329
42: -46.9493828, -3.8277292, -46.9658279, -3.7605009, -39.1968689, 39.0783157

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
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
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 823
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
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 866
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

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0015505
time: 50.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0145307
time: 56.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -38.4265938, 25.1203918, -38.4816742, 25.0992050, -62.9049225, 62.9813995
1: -21.2691994, 21.1999187, -21.3012924, 21.1734314, -41.9008789, 41.9559250
2: -15.3185883, 23.5063438, -15.3559904, 23.4905109, -37.3589783, 37.4105530
3: -28.4280663, 18.7481041, -28.4640121, 18.6887169, -47.1167831, 47.2121162
4: -26.1877251, 22.5854435, -26.2475471, 22.5301075, -48.7178345, 48.8329926
5: -24.1195297, 24.1515808, -24.1671906, 24.1177826, -48.2373123, 48.3187714
6: -53.6123619, -5.1934891, -53.5996590, -5.2099228, -41.5930710, 41.7067032
7: -34.4234581, 15.1867065, -34.4334831, 15.1822224, -48.0472717, 48.0620651
8: -39.9690475, 17.0193005, -40.0090790, 17.0081539, -56.9523926, 56.9950714
9: -23.1332817, 25.7912731, -23.1785831, 25.7303867, -48.8636703, 48.9698563
10: -36.0441666, 24.9184589, -35.9930916, 24.9146843, -60.9588509, 60.9115524
11: -27.2133923, 14.0351934, -27.0244980, 14.0758591, -41.2892532, 41.0596924
12: -45.3015213, 18.7881622, -45.2687492, 18.7850094, -61.8711090, 61.8317413
13: -37.7365570, 26.0113716, -37.7358742, 25.9198532, -63.6564102, 63.7472458
14: -52.1755219, 16.0064831, -52.0729218, 16.0178566, -68.0177002, 67.8897095
15: -21.5841599, 29.0658379, -21.6308060, 28.9855251, -50.5696869, 50.6966438
16: -35.7145386, 20.7373619, -35.6623917, 20.7278061, -55.3955307, 55.3463669
17: -48.7612457, 12.4507618, -48.5731392, 12.4438114, -61.2050552, 61.0239029
18: -21.8485374, 30.0343456, -21.8268414, 30.0644760, -51.9130135, 51.8611870
19: -17.5570660, 19.9209309, -17.4369011, 19.9456902, -37.5027542, 37.3578339
20: -24.8003960, 13.9863167, -24.7409515, 14.0252600, -38.8256569, 38.7272682
21: -22.7371082, 21.8687820, -22.6057835, 21.9110088, -44.6481171, 44.4745636
22: -13.8626146, 27.3736877, -13.8116493, 27.3815784, -37.1483154, 37.1169357
23: -16.2747307, 22.8527851, -16.1509457, 22.8971424, -39.1718750, 39.0037308
24: -16.4329109, 23.8213539, -16.3219414, 23.8396301, -40.2649231, 40.1382675
25: -16.6932316, 27.9552917, -16.5789833, 27.9792137, -43.1928787, 43.1005936
26: -27.0579967, 35.3560867, -26.9948597, 35.3935432, -62.4515381, 62.3509445
27: -21.9786358, 19.5629807, -21.9268055, 19.5528793, -41.5315170, 41.4897842
28: -17.1455021, 25.0444031, -17.0491314, 25.0754681, -41.9845581, 41.8548126
29: -9.9235802, 22.6971340, -9.8060637, 22.7043304, -30.6704559, 30.5648155
30: -29.0025082, 15.5564232, -28.8486862, 15.5984888, -44.6009979, 44.4051094
31: -23.8316021, 24.1716843, -23.7059574, 24.1970520, -48.0286560, 47.8776398
32: -47.6396255, -1.8367624, -47.6232376, -1.9340730, -40.6461105, 40.7471695
33: -70.6257477, 5.6079664, -70.5938110, 5.4932127, -67.0530548, 67.1529388
34: -63.5464249, -5.5698471, -63.5423546, -5.6440601, -46.0867615, 46.2059402
35: -41.8870201, 14.3708124, -41.8724823, 14.3121367, -53.6404572, 53.7541504
36: -42.3357162, 16.1915245, -42.3153076, 16.1005135, -57.9104156, 57.9907990
37: -70.9094238, -1.2309742, -70.8644333, -1.2487011, -66.2891846, 66.3282166
38: -59.4583435, 8.8338490, -59.4578247, 8.7330303, -68.1913757, 68.2916718
39: -74.1950760, 1.1853776, -74.1405258, 1.0485835, -74.2604065, 74.3228607
40: -70.8884888, -18.6521988, -70.8722382, -18.7660789, -41.7147980, 41.8098831
41: -47.7685623, 3.7758131, -47.7453957, 3.7169490, -46.3683167, 46.4198837
42: -46.8779297, -3.8401008, -46.8577652, -3.8641868, -39.0211487, 39.0097313

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 897

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9965199, upper bound: 17.9935817
time: 44.25 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9965200, upper bound: 18.0065669
time: 75.48 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -38.4293442, 25.1271362, -38.4872971, 25.1110268, -62.9206390, 62.9935760
1: -21.2702942, 21.2053986, -21.3040390, 21.1848450, -41.9132233, 41.9640579
2: -15.3192940, 23.5104580, -15.3579292, 23.5002403, -37.3715515, 37.4161224
3: -28.4315376, 18.7524700, -28.4707832, 18.7023315, -47.1338692, 47.2232513
4: -26.1892548, 22.5963707, -26.2559662, 22.5517941, -48.7410507, 48.8523369
5: -24.1208305, 24.1551476, -24.1666565, 24.1278744, -48.2487030, 48.3218040
6: -53.6309624, -5.1915703, -53.6319962, -5.1909332, -41.6331253, 41.7279282
7: -34.4247627, 15.1906090, -34.4360466, 15.1917648, -48.0623627, 48.0702591
8: -39.9706306, 17.0314445, -40.0235786, 17.0313759, -56.9767761, 57.0265503
9: -23.1392250, 25.7953758, -23.1908913, 25.7463970, -48.8856201, 48.9862671
10: -36.0523148, 24.9213200, -36.0094223, 24.9306602, -60.9829750, 60.9307404
11: -27.2234344, 14.0363541, -27.0452881, 14.0857286, -41.3091621, 41.0816422
12: -45.3217773, 18.7910309, -45.3033447, 18.8073921, -61.9009705, 61.8526917
13: -37.7632065, 26.0153160, -37.7834435, 25.9545345, -63.7177429, 63.7987595
14: -52.1798058, 16.0151596, -52.0941505, 16.0341225, -68.0357666, 67.9232483
15: -21.5870991, 29.0750427, -21.6462402, 29.0038872, -50.5909882, 50.7212830
16: -35.7216110, 20.7422485, -35.6780014, 20.7470360, -55.4234085, 55.3632202
17: -48.7661285, 12.4562025, -48.5904541, 12.4536095, -61.2197380, 61.0466576
18: -21.8520298, 30.0488605, -21.8516769, 30.0914116, -51.9434433, 51.9005356
19: -17.5609360, 19.9296322, -17.4538193, 19.9607258, -37.5216599, 37.3834534
20: -24.8038540, 13.9959526, -24.7582397, 14.0418129, -38.8456650, 38.7541924
21: -22.7421589, 21.8793449, -22.6280575, 21.9292393, -44.6713982, 44.5074005
22: -13.8672161, 27.3885498, -13.8362961, 27.4075089, -37.1687469, 37.1565704
23: -16.2775459, 22.8597260, -16.1698875, 22.9102612, -39.1878052, 39.0296135
24: -16.4381084, 23.8334160, -16.3485813, 23.8604584, -40.2829056, 40.1786423
25: -16.6967258, 27.9717770, -16.6053276, 28.0082054, -43.2172318, 43.1438828
26: -27.0649433, 35.3684540, -27.0282593, 35.4166298, -62.4815750, 62.3967133
27: -21.9828720, 19.5768757, -21.9526501, 19.5768814, -41.5597534, 41.5295258
28: -17.1486111, 25.0571213, -17.0715084, 25.0987034, -42.0070648, 41.8909988
29: -9.9286671, 22.7058773, -9.8276024, 22.7199783, -30.6874084, 30.5980835
30: -29.0088310, 15.5599222, -28.8768673, 15.6062431, -44.6150742, 44.4367905
31: -23.8364811, 24.1859646, -23.7286091, 24.2218246, -48.0583038, 47.9145737
32: -47.6627731, -1.8338351, -47.6642647, -1.9064856, -40.6994553, 40.7758255
33: -70.6462021, 5.6094913, -70.6318588, 5.5123596, -67.0962830, 67.1787415
34: -63.5678177, -5.5676932, -63.5806351, -5.6226597, -46.1392899, 46.2193222
35: -41.9050293, 14.3716717, -41.9051781, 14.3275757, -53.6757812, 53.7791748
36: -42.3591385, 16.1929722, -42.3566589, 16.1219826, -57.9568787, 58.0314560
37: -70.9322662, -1.2298708, -70.9059296, -1.2356844, -66.3262024, 66.3657074
38: -59.4820023, 8.8361015, -59.4990540, 8.7599506, -68.2419510, 68.3351593
39: -74.2270813, 1.1867743, -74.1985016, 1.0793238, -74.3254395, 74.3781433
40: -70.9048462, -18.6508331, -70.9020691, -18.7475777, -41.7492218, 41.8289185
41: -47.7938080, 3.7781806, -47.7888641, 3.7425785, -46.4222717, 46.4509583
42: -46.8980560, -3.8373795, -46.8920403, -3.8426223, -39.0643387, 39.0240250

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1507
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
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1281
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
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 897

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0074155, upper bound: 17.9938350
time: 39.11 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9965200, upper bound: 18.0068257
time: 45.72 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -38.5515976, 25.2484207, -38.5040436, 25.1723633, -63.1057434, 63.1161499
1: -21.3442268, 21.2982311, -21.3119316, 21.2279663, -42.0296936, 42.0697403
2: -15.3782473, 23.6033325, -15.3615313, 23.5449867, -37.4657440, 37.5037003
3: -28.4811211, 18.8390560, -28.4689331, 18.7395248, -47.2206459, 47.3079910
4: -26.2735329, 22.6977348, -26.2552338, 22.5941887, -48.8677216, 48.9529686
5: -24.1859360, 24.2389526, -24.1731777, 24.1676216, -48.3535576, 48.4121323
6: -53.7019272, -5.1418571, -53.6496048, -5.2016802, -41.6509628, 41.6798325
7: -34.4852715, 15.2712173, -34.4410706, 15.2298136, -48.1647491, 48.1594086
8: -40.0546761, 17.1728477, -40.0141830, 17.0956345, -57.1277618, 57.1561203
9: -23.2192307, 25.8908691, -23.1880417, 25.7865200, -49.0057526, 49.0789108
10: -36.0864944, 24.9844246, -36.0058441, 24.9459000, -61.0323944, 60.9902687
11: -27.3370914, 14.1035194, -27.0928669, 14.0822525, -41.4193420, 41.1963882
12: -45.4016266, 18.8686562, -45.3247604, 18.7963772, -61.9612579, 61.9678192
13: -37.7981567, 26.1442986, -37.7484512, 25.9893532, -63.7875099, 63.8927498
14: -52.2365532, 16.0619373, -52.0892143, 16.0461693, -68.1090546, 67.9673920
15: -21.6699753, 29.1756001, -21.6410408, 29.0466309, -50.7166061, 50.8166428
16: -35.7909164, 20.7972736, -35.6825638, 20.7600098, -55.5059586, 55.4266205
17: -48.8385849, 12.4686832, -48.6052933, 12.4498415, -61.2884254, 61.0739746
18: -22.0108395, 30.1003895, -21.9138336, 30.0689201, -52.0797577, 52.0142212
19: -17.6439571, 19.9562263, -17.4822006, 19.9465008, -37.5904579, 37.4384270
20: -24.8871498, 14.0543442, -24.7866859, 14.0344534, -38.9216042, 38.8410301
21: -22.8535843, 21.9272976, -22.6656551, 21.9141979, -44.7677841, 44.5929527
22: -13.9448500, 27.4058819, -13.8537035, 27.3836040, -37.2295685, 37.1762848
23: -16.3890476, 22.9220219, -16.2125530, 22.9042969, -39.2933426, 39.1345749
24: -16.5497513, 23.8623581, -16.3859520, 23.8437061, -40.3934555, 40.2483101
25: -16.7842884, 28.0095310, -16.6263142, 27.9847565, -43.2880707, 43.1618729
26: -27.1910076, 35.4340973, -27.0658588, 35.3998718, -62.5908813, 62.4999542
27: -22.0660686, 19.5838470, -21.9735928, 19.5610657, -41.6271362, 41.5574417
28: -17.2568512, 25.1076870, -17.1094379, 25.0834389, -42.1012878, 41.9807739
29: -10.0061035, 22.7235851, -9.8486404, 22.7071533, -30.7565765, 30.6150818
30: -29.1376247, 15.6225758, -28.9234486, 15.6088057, -44.7464294, 44.5460243
31: -23.9608250, 24.2205696, -23.7749233, 24.2001324, -48.1609573, 47.9954910
32: -47.7247696, -1.7820034, -47.6702499, -1.9181204, -40.7233429, 40.8308334
33: -70.6915359, 5.6382952, -70.6264191, 5.5027485, -67.1349030, 67.2183838
34: -63.6464157, -5.5115628, -63.5964355, -5.6355085, -46.2162018, 46.3187714
35: -41.9598160, 14.4061689, -41.9107323, 14.3181562, -53.7718201, 53.8255692
36: -42.3933372, 16.2235909, -42.3467827, 16.1058846, -57.9712524, 58.0453644
37: -70.9977264, -1.1910753, -70.9111557, -1.2421885, -66.4631195, 66.4334717
38: -59.5579376, 8.8846817, -59.5092773, 8.7465191, -68.3044586, 68.3939590
39: -74.2371368, 1.2174821, -74.1605148, 1.0631647, -74.2948914, 74.3742218
40: -70.9478531, -18.6215057, -70.9016571, -18.7532005, -41.7859268, 41.8733902
41: -47.8460617, 3.8139253, -47.7890244, 3.7270417, -46.4661255, 46.5034637
42: -46.9310226, -3.8008537, -46.8868332, -3.8535647, -39.0601578, 39.0354843

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1762
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
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 897

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0031895
time: 49.25 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0161403
time: 44.73 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -38.5543747, 25.2551594, -38.5096779, 25.1841850, -63.1215515, 63.1282043
1: -21.3452873, 21.3036976, -21.3147049, 21.2393341, -42.0420914, 42.0778885
2: -15.3789558, 23.6074486, -15.3634062, 23.5547180, -37.4783325, 37.5092697
3: -28.4845963, 18.8433819, -28.4757156, 18.7531433, -47.2377396, 47.3190994
4: -26.2750473, 22.7087002, -26.2636108, 22.6158161, -48.8908615, 48.9723129
5: -24.1872215, 24.2426033, -24.1725731, 24.1776924, -48.3649139, 48.4151764
6: -53.7205200, -5.1399355, -53.6818390, -5.1827097, -41.6910172, 41.7010422
7: -34.4866257, 15.2750759, -34.4436150, 15.2393990, -48.1798248, 48.1676559
8: -40.0563278, 17.1849442, -40.0286369, 17.1188698, -57.1521912, 57.1876602
9: -23.2251720, 25.8949471, -23.2003403, 25.8024712, -49.0276413, 49.0952873
10: -36.0946922, 24.9872932, -36.0222054, 24.9619427, -61.0566330, 61.0094986
11: -27.3471775, 14.1046658, -27.1137028, 14.0921392, -41.4393158, 41.2183685
12: -45.4218826, 18.8715229, -45.3593292, 18.8188629, -61.9912109, 61.9888306
13: -37.8247986, 26.1483116, -37.7959976, 26.0240860, -63.8488846, 63.9443092
14: -52.2408104, 16.0705853, -52.1104317, 16.0624580, -68.1271362, 68.0008698
15: -21.6729240, 29.1847591, -21.6564560, 29.0649719, -50.7378960, 50.8412170
16: -35.7981186, 20.8021240, -35.6982498, 20.7792816, -55.5338745, 55.4435120
17: -48.8434715, 12.4740753, -48.6226654, 12.4596176, -61.3030891, 61.0967407
18: -22.0143318, 30.1149254, -21.9386864, 30.0958748, -52.1102066, 52.0536118
19: -17.6478176, 19.9649792, -17.4990921, 19.9615669, -37.6093826, 37.4640732
20: -24.8906136, 14.0639677, -24.8039589, 14.0510244, -38.9416389, 38.8679276
21: -22.8586464, 21.9378967, -22.6879349, 21.9324551, -44.7910995, 44.6258316
22: -13.9494858, 27.4207268, -13.8783817, 27.4095459, -37.2499466, 37.2159653
23: -16.3918514, 22.9289818, -16.2314835, 22.9173813, -39.3092346, 39.1604652
24: -16.5549736, 23.8744011, -16.4126110, 23.8645401, -40.4195137, 40.2870102
25: -16.7877712, 28.0259857, -16.6526566, 28.0137348, -43.3124390, 43.2051849
26: -27.1980152, 35.4465446, -27.0991936, 35.4229164, -62.6209335, 62.5457382
27: -22.0702610, 19.5977249, -21.9994583, 19.5850716, -41.6553345, 41.5971832
28: -17.2599468, 25.1204414, -17.1317978, 25.1066628, -42.1237946, 42.0169525
29: -10.0111961, 22.7323532, -9.8701859, 22.7227688, -30.7735634, 30.6483917
30: -29.1440067, 15.6260757, -28.9516525, 15.6165504, -44.7605591, 44.5777283
31: -23.9657364, 24.2348804, -23.7975788, 24.2248707, -48.1906052, 48.0324593
32: -47.7478943, -1.7790084, -47.7112427, -1.8905282, -40.7766876, 40.8594360
33: -70.7119293, 5.6398401, -70.6644745, 5.5218897, -67.1780701, 67.2441559
34: -63.6678200, -5.5094175, -63.6347542, -5.6141176, -46.2687225, 46.3321457
35: -41.9778328, 14.4069901, -41.9434052, 14.3335733, -53.8070679, 53.8505783
36: -42.4168091, 16.2250900, -42.3881378, 16.1273613, -58.0176544, 58.0860519
37: -71.0205460, -1.1899414, -70.9526978, -1.2291737, -66.5001831, 66.4709167
38: -59.5816116, 8.8869276, -59.5504913, 8.7734737, -68.3550873, 68.4374161
39: -74.2691345, 1.2189760, -74.2184601, 1.0940466, -74.3599091, 74.4294739
40: -70.9642792, -18.6201897, -70.9314651, -18.7346458, -41.8203125, 41.8923874
41: -47.8712692, 3.8162780, -47.8325195, 3.7527218, -46.5200653, 46.5345459
42: -46.9511337, -3.7981377, -46.9210663, -3.8319421, -39.1033249, 39.0497589

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1281
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
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 897

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0034786
time: 41.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0164335
time: 45.03 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -38.4401665, 25.1756172, -38.6911354, 25.2087555, -63.0239105, 63.2483673
1: -21.2740364, 21.2470417, -21.4652920, 21.2704163, -41.9989548, 42.1680145
2: -15.3234138, 23.5474987, -15.4768934, 23.5785904, -37.4483261, 37.5736465
3: -28.4303722, 18.8091621, -28.6318951, 18.8235435, -47.2539139, 47.4410553
4: -26.1926422, 22.6387787, -26.4121380, 22.6398888, -48.8325310, 49.0509186
5: -24.1228542, 24.1978168, -24.3129578, 24.2181282, -48.3409805, 48.5107727
6: -53.6365738, -5.1878166, -53.6614342, -5.1399555, -41.7228622, 41.7725677
7: -34.4266510, 15.2268028, -34.5867233, 15.2664337, -48.1278992, 48.2570648
8: -39.9735184, 17.0745449, -40.1932259, 17.1270485, -57.0700378, 57.2352753
9: -23.1420212, 25.8276253, -23.3484764, 25.8091736, -48.9511948, 49.1761017
10: -36.0580444, 24.9257011, -36.1084595, 24.9626560, -61.0206985, 61.0341606
11: -27.2476959, 14.0403070, -27.1268768, 14.1854057, -41.4331017, 41.1671829
12: -45.3478813, 18.7930069, -45.3773727, 18.9233379, -62.0690002, 61.9439240
13: -37.7443733, 26.0266533, -37.8386955, 25.9802170, -63.7245903, 63.8653488
14: -52.2071838, 16.0095520, -52.2052307, 16.0716934, -68.1319122, 68.0129547
15: -21.5895500, 29.1054478, -21.7760887, 29.0769997, -50.6665497, 50.8815384
16: -35.7282028, 20.7503281, -35.8278503, 20.7674141, -55.4463959, 55.5257263
17: -48.7993546, 12.4564247, -48.7040634, 12.5746155, -61.3739700, 61.1604881
18: -21.8788147, 30.0398846, -21.9210472, 30.1978760, -52.0766907, 51.9609299
19: -17.5987244, 19.9219284, -17.5418510, 20.0662251, -37.6649475, 37.4637794
20: -24.8407745, 13.9908209, -24.8357525, 14.1498318, -38.9906082, 38.8265724
21: -22.7721500, 21.8714123, -22.7062626, 22.0047989, -44.7769470, 44.5776749
22: -13.8919678, 27.3762646, -13.8901949, 27.4936390, -37.2934952, 37.1859207
23: -16.3212852, 22.8562946, -16.2569618, 23.0261135, -39.3473969, 39.1132584
24: -16.4774704, 23.8243484, -16.4265594, 23.9575138, -40.4299850, 40.2401199
25: -16.7466908, 27.9587975, -16.7086525, 28.1444721, -43.4147110, 43.2290192
26: -27.0969028, 35.3628273, -27.1001549, 35.5486641, -62.6455688, 62.4629822
27: -22.0008316, 19.5671024, -21.9814854, 19.6135178, -41.6143494, 41.5485878
28: -17.1902046, 25.0491028, -17.1500492, 25.2143593, -42.1696701, 41.9557571
29: -9.9522552, 22.6997261, -9.8880186, 22.7922382, -30.7874908, 30.6426735
30: -29.0423679, 15.5606670, -28.9460869, 15.7132292, -44.7555962, 44.5067520
31: -23.8913250, 24.1751251, -23.8460922, 24.3523903, -48.2437134, 48.0212173
32: -47.6634903, -1.8322387, -47.6815567, -1.8596025, -40.7547379, 40.8095856
33: -70.6397552, 5.6148005, -70.6382675, 5.5847311, -67.1674500, 67.2090759
34: -63.5582047, -5.5647678, -63.5784988, -5.5433016, -46.2146683, 46.2568741
35: -41.8953056, 14.3758535, -41.9044266, 14.3946924, -53.7286224, 53.8009567
36: -42.3547745, 16.1955528, -42.3625412, 16.1971149, -58.0315399, 58.0411835
37: -70.9476013, -1.2251396, -70.9653931, -1.1386909, -66.4447479, 66.4273376
38: -59.4895096, 8.8414669, -59.5477219, 8.8464775, -68.3359833, 68.3891907
39: -74.2155380, 1.1888108, -74.2121353, 1.1220312, -74.3554382, 74.3948517
40: -70.9005051, -18.6452484, -70.9161072, -18.7059937, -41.7978516, 41.8595734
41: -47.7818375, 3.7810645, -47.7839737, 3.7661715, -46.4524765, 46.4650955
42: -46.8940926, -3.8357925, -46.9040375, -3.7913470, -39.1639862, 39.0551987

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1594
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
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1507
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
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1281
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
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1306
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
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 897

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9965199, upper bound: 18.0094729
time: 51.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9965200, upper bound: 18.0224766
time: 42.21 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -38.4428940, 25.1823692, -38.6968079, 25.2205505, -63.0395966, 63.2604523
1: -21.2751179, 21.2525101, -21.4680843, 21.2817497, -42.0113144, 42.1761780
2: -15.3241253, 23.5516205, -15.4787998, 23.5882988, -37.4608994, 37.5792694
3: -28.4338531, 18.8134766, -28.6386490, 18.8371334, -47.2709885, 47.4521255
4: -26.1941605, 22.6497231, -26.4205856, 22.6614552, -48.8556137, 49.0703087
5: -24.1241589, 24.2014198, -24.3123703, 24.2282181, -48.3523788, 48.5137901
6: -53.6551819, -5.1858931, -53.6937485, -5.1209803, -41.7629242, 41.7938309
7: -34.4279709, 15.2306862, -34.5892296, 15.2759943, -48.1428986, 48.2652435
8: -39.9751129, 17.0867348, -40.2076988, 17.1503048, -57.0944672, 57.2667847
9: -23.1479034, 25.8316727, -23.3607445, 25.8251801, -48.9730835, 49.1924171
10: -36.0662308, 24.9286194, -36.1247635, 24.9787025, -61.0449333, 61.0533829
11: -27.2577019, 14.0414801, -27.1477127, 14.1952543, -41.4529572, 41.1891937
12: -45.3680496, 18.7958546, -45.4119034, 18.9457436, -62.0988617, 61.9649048
13: -37.7709541, 26.0305748, -37.8862610, 26.0148945, -63.7858505, 63.9168358
14: -52.2114601, 16.0182724, -52.2265511, 16.0879402, -68.1499939, 68.0466003
15: -21.5925140, 29.1146584, -21.7914829, 29.0953388, -50.6878510, 50.9061432
16: -35.7353134, 20.7551918, -35.8434715, 20.7866325, -55.4742584, 55.5425339
17: -48.8041992, 12.4618111, -48.7213974, 12.5844336, -61.3886337, 61.1832085
18: -21.8822346, 30.0544243, -21.9458790, 30.2248058, -52.1070404, 52.0003052
19: -17.6025848, 19.9306736, -17.5587673, 20.0812550, -37.6838379, 37.4894409
20: -24.8442440, 14.0004358, -24.8530197, 14.1664057, -39.0106506, 38.8534546
21: -22.7772007, 21.8819771, -22.7285385, 22.0230408, -44.8002396, 44.6105156
22: -13.8965740, 27.3911057, -13.9148273, 27.5195580, -37.3138962, 37.2255630
23: -16.3240948, 22.8632393, -16.2759190, 23.0391884, -39.3632812, 39.1391602
24: -16.4826393, 23.8363762, -16.4532032, 23.9783440, -40.4479370, 40.2804565
25: -16.7501373, 27.9752197, -16.7349968, 28.1734676, -43.4390564, 43.2722855
26: -27.1038837, 35.3752365, -27.1335278, 35.5717506, -62.6756363, 62.5087662
27: -22.0050449, 19.5809898, -22.0073681, 19.6375122, -41.6425552, 41.5883560
28: -17.1933250, 25.0618057, -17.1724396, 25.2375908, -42.1922226, 41.9919624
29: -9.9573135, 22.7084732, -9.9095221, 22.8078575, -30.8044167, 30.6759720
30: -29.0487270, 15.5641413, -28.9742355, 15.7209911, -44.7697182, 44.5383759
31: -23.8961601, 24.1894302, -23.8687286, 24.3771324, -48.2732925, 48.0581589
32: -47.6866531, -1.8293314, -47.7225800, -1.8319969, -40.8080597, 40.8381882
33: -70.6602020, 5.6163158, -70.6762848, 5.6037865, -67.2106323, 67.2348480
34: -63.5795593, -5.5626564, -63.6167870, -5.5219159, -46.2671967, 46.2702484
35: -41.9133415, 14.3766899, -41.9371262, 14.4101629, -53.7639313, 53.8259659
36: -42.3782463, 16.1970291, -42.4039078, 16.2185745, -58.0779724, 58.0818405
37: -70.9704132, -1.2240591, -71.0069046, -1.1256256, -66.4819183, 66.4648590
38: -59.5132294, 8.8437986, -59.5889091, 8.8735256, -68.3867569, 68.4327087
39: -74.2474976, 1.1902280, -74.2700806, 1.1527686, -74.4205017, 74.4501343
40: -70.9168472, -18.6439400, -70.9459076, -18.6875229, -41.8322678, 41.8785782
41: -47.8070984, 3.7834458, -47.8274918, 3.7918429, -46.5063934, 46.4961777
42: -46.9141922, -3.8331337, -46.9382973, -3.7697463, -39.2071304, 39.0694618

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
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
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1594
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
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1507
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
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1281
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
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 894
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
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1306
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
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 897

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9965199, upper bound: 18.0097208
time: 60.35 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0074155, upper bound: 18.0227276
time: 176.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -38.5651093, 25.3035851, -38.7134705, 25.2818604, -63.2247162, 63.3830414
1: -21.3490486, 21.3453484, -21.4759254, 21.3249283, -42.1277924, 42.2818298
2: -15.3830023, 23.6444511, -15.4823704, 23.6330452, -37.5550613, 37.6667938
3: -28.4834595, 18.9000511, -28.6368084, 18.8743820, -47.3578415, 47.5368576
4: -26.2784462, 22.7510777, -26.4197540, 22.7039566, -48.9824028, 49.1708298
5: -24.1892586, 24.2852287, -24.3189163, 24.2679558, -48.4572144, 48.6041451
6: -53.7261429, -5.1361513, -53.7114029, -5.1317368, -41.7807388, 41.7457123
7: -34.4884987, 15.3112869, -34.5943031, 15.3140631, -48.2453308, 48.3543930
8: -40.0591125, 17.2281075, -40.1983185, 17.2145958, -57.2453918, 57.3964539
9: -23.2279282, 25.9271889, -23.3578815, 25.8653221, -49.0932503, 49.2850723
10: -36.1003799, 24.9916821, -36.1212006, 24.9938717, -61.0942535, 61.1128845
11: -27.3713760, 14.1085949, -27.1952648, 14.1917706, -41.5631485, 41.3038597
12: -45.4479332, 18.8734550, -45.4333344, 18.9347076, -62.1590881, 62.0800476
13: -37.8059006, 26.1596203, -37.8512917, 26.0497532, -63.8556519, 64.0109100
14: -52.2683029, 16.0650253, -52.2216301, 16.0999985, -68.2232819, 68.0906372
15: -21.6753597, 29.2152176, -21.7863159, 29.1380424, -50.8134003, 51.0015335
16: -35.8046188, 20.8101883, -35.8480072, 20.7996330, -55.5567703, 55.6060257
17: -48.8766708, 12.4742928, -48.7362518, 12.5806503, -61.4573212, 61.2105446
18: -22.0411224, 30.1059361, -22.0080147, 30.2022629, -52.2433853, 52.1139526
19: -17.6856308, 19.9572487, -17.5871181, 20.0670357, -37.7526665, 37.5443649
20: -24.9275379, 14.0588474, -24.8814564, 14.1590080, -39.0865479, 38.9403038
21: -22.8886318, 21.9299507, -22.7661095, 22.0080032, -44.8966370, 44.6960602
22: -13.9742060, 27.4083824, -13.9322491, 27.4956284, -37.3747177, 37.2453003
23: -16.4356060, 22.9255447, -16.3185482, 23.0332203, -39.4688263, 39.2440948
24: -16.5943413, 23.8653107, -16.4905739, 23.9615898, -40.5559311, 40.3558846
25: -16.8377380, 28.0130291, -16.7559662, 28.1500244, -43.5099716, 43.2903061
26: -27.2299919, 35.4408760, -27.1711502, 35.5549393, -62.7849312, 62.6120262
27: -22.0882874, 19.5879784, -22.0283051, 19.6216888, -41.7099762, 41.6162834
28: -17.3015251, 25.1124039, -17.2103500, 25.2223206, -42.2864304, 42.0817108
29: -10.0347939, 22.7261734, -9.9305677, 22.7950363, -30.8736420, 30.6929855
30: -29.1775150, 15.6268253, -29.0208378, 15.7235518, -44.9010658, 44.6476631
31: -24.0205498, 24.2239895, -23.9150620, 24.3554478, -48.3759995, 48.1390533
32: -47.7486458, -1.7774892, -47.7285805, -1.8436608, -40.8319931, 40.8931885
33: -70.7056122, 5.6451502, -70.6708832, 5.5943680, -67.2492676, 67.2745514
34: -63.6581841, -5.5065389, -63.6325912, -5.5347996, -46.3441010, 46.3695831
35: -41.9681168, 14.4111013, -41.9426308, 14.4007435, -53.8600006, 53.8723373
36: -42.4124451, 16.2276154, -42.3940086, 16.2024765, -58.0923309, 58.0957489
37: -71.0359650, -1.1852770, -71.0121613, -1.1322556, -66.6188202, 66.5325775
38: -59.5891876, 8.8922672, -59.5991287, 8.8599329, -68.4491196, 68.4913940
39: -74.2575226, 1.2209125, -74.2321320, 1.1366477, -74.3899841, 74.4462280
40: -70.9599762, -18.6145592, -70.9455261, -18.6931305, -41.8690338, 41.9231110
41: -47.8593521, 3.8192058, -47.8276558, 3.7763186, -46.5502472, 46.5486832
42: -46.9471474, -3.7965488, -46.9330978, -3.7806792, -39.2029572, 39.0809669

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1502
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
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1594
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
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1507
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
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1281
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
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1667
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
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1540
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
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 897

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0191376
time: 41.51 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0321156
time: 61.95 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -38.5678940, 25.3103504, -38.7191544, 25.2936287, -63.2404327, 63.3950958
1: -21.3501129, 21.3508034, -21.4786816, 21.3362751, -42.1400909, 42.2899170
2: -15.3837318, 23.6485691, -15.4843044, 23.6427269, -37.5676270, 37.6723557
3: -28.4868870, 18.9043770, -28.6435509, 18.8879471, -47.3748322, 47.5479279
4: -26.2799911, 22.7620449, -26.4282150, 22.7255936, -49.0055847, 49.1902618
5: -24.1905594, 24.2888908, -24.3183098, 24.2780800, -48.4686394, 48.6072006
6: -53.7447433, -5.1342459, -53.7436752, -5.1127377, -41.8208237, 41.7669601
7: -34.4898186, 15.3151731, -34.5968399, 15.3236046, -48.2603912, 48.3626251
8: -40.0607491, 17.2402763, -40.2127686, 17.2378082, -57.2698517, 57.4279480
9: -23.2338734, 25.9312725, -23.3701477, 25.8813343, -49.1152077, 49.3014221
10: -36.1085548, 24.9946175, -36.1375122, 25.0099144, -61.1184692, 61.1321297
11: -27.3814182, 14.1097488, -27.2160721, 14.2016468, -41.5830650, 41.3258209
12: -45.4681854, 18.8763485, -45.4679451, 18.9570923, -62.1890411, 62.1010284
13: -37.8325310, 26.1635551, -37.8988266, 26.0844421, -63.9169731, 64.0623779
14: -52.2725677, 16.0736313, -52.2429504, 16.1162720, -68.2413788, 68.1241913
15: -21.6782932, 29.2243671, -21.8017406, 29.1564674, -50.8347626, 51.0261078
16: -35.8117676, 20.8151035, -35.8636246, 20.8189049, -55.5847321, 55.6228333
17: -48.8815727, 12.4797344, -48.7535744, 12.5903950, -61.4719696, 61.2333069
18: -22.0446339, 30.1204453, -22.0328236, 30.2292175, -52.2738495, 52.1532669
19: -17.6895008, 19.9659996, -17.6040230, 20.0820560, -37.7715569, 37.5700226
20: -24.9309902, 14.0684633, -24.8987389, 14.1755800, -39.1065712, 38.9672012
21: -22.8936844, 21.9405518, -22.7884369, 22.0262375, -44.9199219, 44.7289886
22: -13.9788342, 27.4232540, -13.9569273, 27.5215778, -37.3951035, 37.2849197
23: -16.4384346, 22.9324837, -16.3374691, 23.0463181, -39.4847527, 39.2699509
24: -16.5995541, 23.8773937, -16.5172653, 23.9824181, -40.5819702, 40.3946609
25: -16.8412247, 28.0294609, -16.7823315, 28.1789589, -43.5342941, 43.3335648
26: -27.2369003, 35.4533119, -27.2044621, 35.5780029, -62.8149033, 62.6577759
27: -22.0924625, 19.6018715, -22.0541821, 19.6456966, -41.7381592, 41.6560516
28: -17.3046703, 25.1251183, -17.2327137, 25.2455349, -42.3089600, 42.1179428
29: -10.0398788, 22.7349396, -9.9521303, 22.8106499, -30.8905716, 30.7262802
30: -29.1838512, 15.6302767, -29.0490437, 15.7312489, -44.9151001, 44.6793213
31: -24.0254822, 24.2383270, -23.9377308, 24.3801861, -48.4056702, 48.1760559
32: -47.7717781, -1.7745366, -47.7695694, -1.8160472, -40.8853149, 40.9217987
33: -70.7260056, 5.6466417, -70.7089081, 5.6133423, -67.2925262, 67.3003540
34: -63.6795387, -5.5043969, -63.6708603, -5.5134287, -46.3966064, 46.3829956
35: -41.9860992, 14.4120064, -41.9753036, 14.4162102, -53.8953400, 53.8973312
36: -42.4358902, 16.2291069, -42.4353714, 16.2239380, -58.1387787, 58.1364136
37: -71.0587311, -1.1841583, -71.0536575, -1.1191730, -66.6558990, 66.5701447
38: -59.6128197, 8.8945742, -59.6403198, 8.8869171, -68.4997406, 68.5348969
39: -74.2894897, 1.2224226, -74.2900925, 1.1674471, -74.4550476, 74.5015411
40: -70.9763794, -18.6131897, -70.9753265, -18.6745834, -41.9034576, 41.9421539
41: -47.8845444, 3.8215513, -47.8711166, 3.8019199, -46.6042023, 46.5797577
42: -46.9672623, -3.7938704, -46.9673653, -3.7591066, -39.2461243, 39.0952530

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
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
type: A, layer: 1, pos: 1662
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
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1502
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
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1594
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
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 911
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
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1507
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
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1281
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
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1667
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
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1540
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
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 897

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0194189
time: 46.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0323986
time: 48.40 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 97.02 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -17.9965199, upper bound: 17.9756467
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -17.9965200, upper bound: 17.9886326
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0074155, upper bound: 17.9759013
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0074155, upper bound: 17.9888924
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0214291, upper bound: 17.9853360
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0214291, upper bound: 17.9982931
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0323986, upper bound: 17.9856141
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0323986, upper bound: 17.9985805
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -17.9965199, upper bound: 17.9915270
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -17.9965200, upper bound: 18.0045420
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0074155, upper bound: 17.9917707
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0074155, upper bound: 18.0047893
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0012727
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0142529
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0015505
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0145307
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -17.9965199, upper bound: 17.9935817
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -17.9965200, upper bound: 18.0065669
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0074155, upper bound: 17.9938350
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -17.9965200, upper bound: 18.0068257
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0031895
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0161403
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0034786
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0164335
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -17.9965199, upper bound: 18.0094729
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -17.9965200, upper bound: 18.0224766
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -17.9965199, upper bound: 18.0097208
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0074155, upper bound: 18.0227276
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0191376
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0321156
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0194189
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 97.02
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0323986

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -38.2753220, 25.0381718, -38.4217758, 25.0917759, -62.7480469, 62.8378296
1: -21.1703186, 21.1315880, -21.2659740, 21.1654415, -41.7887878, 41.8397675
2: -15.2640581, 23.4565105, -15.3353739, 23.4835548, -37.2914047, 37.3240585
3: -28.3355846, 18.6455822, -28.4178925, 18.6749115, -47.0104980, 47.0634766
4: -26.0827217, 22.4807854, -26.1988220, 22.5203362, -48.6030579, 48.6796074
5: -24.0712280, 24.0859985, -24.1485901, 24.1092110, -48.1804390, 48.2345886
6: -53.5753708, -5.2966213, -53.5894623, -5.2429752, -41.4837341, 41.5871429
7: -34.3544464, 15.1603241, -34.4223976, 15.1753674, -47.9551392, 47.9870834
8: -39.8741722, 16.9429474, -39.9704208, 16.9964867, -56.8331604, 56.8683853
9: -23.0045166, 25.6866169, -23.1250725, 25.7192097, -48.7237244, 48.8116913
10: -35.9059753, 24.8558617, -35.9572372, 24.8985386, -60.8045120, 60.8130989
11: -26.9261913, 13.8786488, -27.0111046, 13.9997921, -40.9259834, 40.8897552
12: -45.2170219, 18.6273956, -45.2597923, 18.7241325, -61.7180481, 61.6593323
13: -37.6568756, 25.8351059, -37.7129936, 25.8850670, -63.5419426, 63.5480995
14: -51.9879074, 15.9026279, -52.0440865, 15.9662991, -67.7831116, 67.7607574
15: -21.4044380, 28.9234695, -21.5438766, 28.9747677, -50.3792038, 50.4673462
16: -35.5653267, 20.6938477, -35.6386223, 20.7167282, -55.2271118, 55.2743149
17: -48.4565887, 12.2650337, -48.5512733, 12.3511353, -60.8077240, 60.8163071
18: -21.7420940, 29.9828720, -21.8022594, 30.0462570, -51.7883530, 51.7851334
19: -17.3730068, 19.8188362, -17.4196301, 19.8929615, -37.2659683, 37.2384644
20: -24.6987190, 13.9011154, -24.7292747, 13.9853477, -38.6840668, 38.6303902
21: -22.5161591, 21.7498798, -22.5856686, 21.8508148, -44.3669739, 44.3355484
22: -13.7739410, 27.3476238, -13.7950630, 27.3714828, -37.0453949, 37.0643311
23: -16.1095695, 22.7748814, -16.1390381, 22.8622990, -38.9718704, 38.9139175
24: -16.2902775, 23.7804642, -16.3082733, 23.8242073, -40.1019974, 40.0832443
25: -16.5480824, 27.8821526, -16.5628204, 27.9467621, -43.0099640, 43.0071945
26: -26.9366989, 35.2713432, -26.9756889, 35.3561096, -62.2928085, 62.2470322
27: -21.8996544, 19.5380344, -21.9151440, 19.5457649, -41.4454193, 41.4531784
28: -17.0094700, 24.9797649, -17.0373001, 25.0467815, -41.8223877, 41.7813416
29: -9.7380533, 22.6200256, -9.7907648, 22.6671257, -30.4475327, 30.4727707
30: -28.7921696, 15.4397745, -28.8371735, 15.5432596, -44.3354301, 44.2769470
31: -23.6359425, 24.0512962, -23.6830444, 24.1374340, -47.7733765, 47.7343407
32: -47.5864182, -1.9833007, -47.6067886, -1.9521594, -40.5759888, 40.5834808
33: -70.4891968, 5.4639416, -70.5425568, 5.4844980, -66.9193115, 66.9601746
34: -63.4075623, -5.7382321, -63.4780312, -5.6574283, -45.9356689, 45.9838638
35: -41.8107338, 14.2837114, -41.8471985, 14.3076210, -53.5570221, 53.6280975
36: -42.2772369, 16.0529480, -42.2975388, 16.0810375, -57.8300781, 57.8321609
37: -70.8214111, -1.3020926, -70.8405457, -1.2648239, -66.1321564, 66.2094574
38: -59.4102974, 8.6884470, -59.4419403, 8.7192383, -68.1295319, 68.1303864
39: -74.0852509, 1.0431342, -74.1018753, 1.0446954, -74.1307526, 74.1277161
40: -70.8121490, -18.7897301, -70.8457184, -18.7738037, -41.6295013, 41.6486893
41: -47.7110863, 3.6961913, -47.7299080, 3.7080350, -46.2949066, 46.3218994
42: -46.8448563, -3.8985724, -46.8500786, -3.8774633, -38.9264069, 38.9521332

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1404
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
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 911
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
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 960
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
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
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
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 848
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
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1766
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9776127, upper bound: 17.9685467
time: 51.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9873062, upper bound: 17.9682152
time: 39.31 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -38.3840485, 25.0853024, -38.4802094, 25.0968418, -62.8416290, 62.9437256
1: -21.2377529, 21.1660347, -21.3013191, 21.1710930, -41.8639603, 41.9244614
2: -15.3037777, 23.4789391, -15.3560572, 23.4885216, -37.3412857, 37.3812790
3: -28.4138069, 18.6973076, -28.4617214, 18.6863289, -47.1001358, 47.1590271
4: -26.1687984, 22.5271530, -26.2456894, 22.5274448, -48.6962433, 48.7728424
5: -24.1038284, 24.1164532, -24.1654034, 24.1166191, -48.2204475, 48.2818565
6: -53.5889206, -5.2435675, -53.5965233, -5.2158384, -41.5570526, 41.6547470
7: -34.3754311, 15.1775808, -34.4299927, 15.1824951, -47.9947052, 48.0645676
8: -39.9385681, 16.9838486, -40.0057335, 17.0049953, -56.9125977, 56.9620438
9: -23.0808334, 25.7216187, -23.1670113, 25.7287140, -48.8095474, 48.8886299
10: -35.9708862, 24.8845596, -35.9874496, 24.9117451, -60.8826294, 60.8720093
11: -26.9950447, 13.9800529, -27.0197964, 14.0568161, -41.0518608, 40.9998474
12: -45.2712936, 18.7280521, -45.2679520, 18.7796497, -61.8361969, 61.7741852
13: -37.6856651, 25.8894348, -37.7237053, 25.9142189, -63.5998840, 63.6131401
14: -52.0567093, 15.9747276, -52.0679169, 16.0060349, -67.8874817, 67.8484344
15: -21.5406208, 28.9682121, -21.6178131, 28.9817963, -50.5224152, 50.5860252
16: -35.6051254, 20.7185249, -35.6534500, 20.7266769, -55.2797241, 55.3150330
17: -48.5520554, 12.3946886, -48.5678101, 12.4241085, -60.9761658, 60.9624977
18: -21.7826099, 30.0122986, -21.8227100, 30.0610313, -51.8436432, 51.8350067
19: -17.4308548, 19.8817844, -17.4331322, 19.9287872, -37.3596420, 37.3149185
20: -24.7370319, 13.9693756, -24.7378025, 14.0235119, -38.7605438, 38.7071762
21: -22.5769444, 21.8312492, -22.5995636, 21.8964291, -44.4733734, 44.4308128
22: -13.8054104, 27.3616486, -13.8083668, 27.3784504, -37.0914078, 37.0944214
23: -16.1416359, 22.8119926, -16.1476631, 22.8824749, -39.0241089, 38.9596558
24: -16.3166103, 23.7967072, -16.3188171, 23.8327980, -40.1336365, 40.1062622
25: -16.5883064, 27.9228649, -16.5751419, 27.9692497, -43.0808411, 43.0610809
26: -26.9773655, 35.3279152, -26.9911633, 35.3875504, -62.3649139, 62.3190765
27: -21.9175949, 19.5513725, -21.9235706, 19.5519981, -41.4695930, 41.4749451
28: -17.0449066, 25.0205288, -17.0464153, 25.0692177, -41.8798523, 41.8259430
29: -9.7947865, 22.6727619, -9.8028898, 22.6971893, -30.5350571, 30.5361938
30: -28.8295307, 15.5223656, -28.8439579, 15.5885468, -44.4180756, 44.3663254
31: -23.6962967, 24.1230507, -23.7009659, 24.1772594, -47.8735580, 47.8240166
32: -47.5949211, -1.9547949, -47.6109352, -1.9379239, -40.6000671, 40.6164627
33: -70.5471115, 5.4916248, -70.5733490, 5.4923229, -66.9653473, 67.0139618
34: -63.5031891, -5.6793938, -63.5318298, -5.6474600, -46.0107956, 46.1006012
35: -41.8435516, 14.2976809, -41.8645439, 14.3108358, -53.5922852, 53.6620407
36: -42.3012085, 16.0843925, -42.3077583, 16.0948257, -57.8721924, 57.8760910
37: -70.8638458, -1.2733526, -70.8604965, -1.2522993, -66.2647247, 66.2751923
38: -59.4408798, 8.7155991, -59.4563446, 8.7287035, -68.1695862, 68.1719437
39: -74.1211319, 1.0570273, -74.1207275, 1.0475154, -74.1803589, 74.1625366
40: -70.8444748, -18.7664070, -70.8622665, -18.7678452, -41.6699982, 41.6891251
41: -47.7327805, 3.7105665, -47.7403793, 3.7144041, -46.3285980, 46.3486023
42: -46.8594666, -3.8786712, -46.8559036, -3.8679838, -38.9791718, 38.9857025

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1698
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
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1404
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
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 911
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
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 960
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
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
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
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 894
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
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1766
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9776127, upper bound: 17.9815423
time: 41.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9873062, upper bound: 17.9812120
time: 42.52 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -38.2780647, 25.0449047, -38.4274445, 25.1035786, -62.7638702, 62.8499298
1: -21.1714153, 21.1370926, -21.2687683, 21.1768398, -41.8011246, 41.8479233
2: -15.2647820, 23.4606667, -15.3372650, 23.4932442, -37.3040161, 37.3296280
3: -28.3390713, 18.6498795, -28.4246769, 18.6884880, -47.0275574, 47.0745544
4: -26.0842857, 22.4918289, -26.2072468, 22.5419865, -48.6262741, 48.6990738
5: -24.0725365, 24.0896263, -24.1479969, 24.1192589, -48.1917953, 48.2376251
6: -53.5939484, -5.2947388, -53.6217422, -5.2240162, -41.5238495, 41.6083679
7: -34.3557625, 15.1641970, -34.4249344, 15.1848602, -47.9701996, 47.9953079
8: -39.8758240, 16.9550934, -39.9848976, 17.0197315, -56.8575745, 56.8999176
9: -23.0104790, 25.6907139, -23.1373787, 25.7352142, -48.7456932, 48.8280945
10: -35.9142036, 24.8587189, -35.9735680, 24.9145279, -60.8287315, 60.8322868
11: -26.9362392, 13.8797607, -27.0318928, 14.0096684, -40.9459076, 40.9116516
12: -45.2372322, 18.6302299, -45.2944565, 18.7465820, -61.7478943, 61.6802216
13: -37.6834793, 25.8390350, -37.7605629, 25.9197731, -63.6032524, 63.5995979
14: -51.9921875, 15.9113188, -52.0653305, 15.9825830, -67.8011322, 67.7942352
15: -21.4073544, 28.9326439, -21.5592918, 28.9931087, -50.4004631, 50.4919357
16: -35.5724640, 20.6986904, -35.6542358, 20.7359905, -55.2550659, 55.2911987
17: -48.4614944, 12.2704926, -48.5685043, 12.3609495, -60.8224449, 60.8389969
18: -21.7455559, 29.9974670, -21.8270569, 30.0732040, -51.8187599, 51.8245239
19: -17.3768520, 19.8275890, -17.4365196, 19.9080219, -37.2848740, 37.2641068
20: -24.7021713, 13.9107227, -24.7465496, 14.0019588, -38.7041321, 38.6572723
21: -22.5212059, 21.7604580, -22.6080208, 21.8690357, -44.3902435, 44.3684769
22: -13.7786036, 27.3624973, -13.8197308, 27.3974171, -37.0657883, 37.1039734
23: -16.1124020, 22.7818184, -16.1579895, 22.8754139, -38.9878159, 38.9398079
24: -16.2954731, 23.7925377, -16.3348980, 23.8450127, -40.1199341, 40.1236115
25: -16.5515575, 27.8986244, -16.5891361, 27.9757385, -43.0343246, 43.0504761
26: -26.9436550, 35.2837791, -27.0090637, 35.3792419, -62.3228989, 62.2928429
27: -21.9038677, 19.5519276, -21.9410076, 19.5697556, -41.4736252, 41.4929352
28: -17.0125980, 24.9924545, -17.0596619, 25.0700111, -41.8449020, 41.8175507
29: -9.7431507, 22.6288052, -9.8122921, 22.6827431, -30.4644814, 30.5060654
30: -28.7985516, 15.4432793, -28.8653374, 15.5510283, -44.3495789, 44.3086166
31: -23.6408024, 24.0656414, -23.7056656, 24.1621761, -47.8029785, 47.7713089
32: -47.6095581, -1.9803491, -47.6477814, -1.9245563, -40.6293869, 40.6121445
33: -70.5096130, 5.4654493, -70.5806274, 5.5036077, -66.9625092, 66.9859009
34: -63.4289665, -5.7361741, -63.5162849, -5.6360378, -45.9881439, 45.9972153
35: -41.8287430, 14.2845392, -41.8798981, 14.3230410, -53.5923004, 53.6531067
36: -42.3006935, 16.0544701, -42.3389359, 16.1025181, -57.8765259, 57.8728790
37: -70.8442459, -1.3009825, -70.8820724, -1.2517424, -66.1692657, 66.2469177
38: -59.4339981, 8.6906443, -59.4831505, 8.7461367, -68.1801376, 68.1737976
39: -74.1172791, 1.0445561, -74.1597748, 1.0755124, -74.1958466, 74.1830292
40: -70.8285370, -18.7883873, -70.8755646, -18.7553024, -41.6639099, 41.6676636
41: -47.7363281, 3.6986022, -47.7734451, 3.7336469, -46.3488846, 46.3529816
42: -46.8649445, -3.8958540, -46.8843460, -3.8559217, -38.9695435, 38.9664040

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1404
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
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 911
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
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
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
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 848
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
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1766
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

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9818289, upper bound: 17.9725222
time: 43.11 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0040200, upper bound: 17.9725222
time: 40.05 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -38.3868561, 25.0920868, -38.4858284, 25.1086349, -62.8573303, 62.9557800
1: -21.2388363, 21.1715202, -21.3040905, 21.1824780, -41.8763199, 41.9326706
2: -15.3044720, 23.4830742, -15.3579826, 23.4982338, -37.3538513, 37.3869400
3: -28.4172688, 18.7016296, -28.4684868, 18.6999397, -47.1172104, 47.1701164
4: -26.1702976, 22.5382156, -26.2541580, 22.5490494, -48.7193451, 48.7923737
5: -24.1051159, 24.1200809, -24.1648197, 24.1266556, -48.2317734, 48.2849007
6: -53.6075096, -5.2416687, -53.6288071, -5.1968727, -41.5971451, 41.6759872
7: -34.3767281, 15.1814470, -34.4325714, 15.1920071, -48.0097961, 48.0727692
8: -39.9401703, 16.9959946, -40.0202255, 17.0282059, -56.9370270, 56.9935150
9: -23.0867920, 25.7257442, -23.1793365, 25.7446804, -48.8314743, 48.9050827
10: -35.9791260, 24.8874874, -36.0038185, 24.9277306, -60.9068565, 60.8913040
11: -27.0050831, 13.9811382, -27.0405922, 14.0666962, -41.0717773, 41.0217285
12: -45.2915268, 18.7309284, -45.3025284, 18.8020248, -61.8661346, 61.7951355
13: -37.7123528, 25.8934288, -37.7713165, 25.9489021, -63.6612549, 63.6647453
14: -52.0610313, 15.9833431, -52.0891953, 16.0223026, -67.9056091, 67.8820038
15: -21.5435219, 28.9773655, -21.6332378, 29.0001621, -50.5436859, 50.6106033
16: -35.6122208, 20.7233505, -35.6691093, 20.7459431, -55.3076782, 55.3318481
17: -48.5569534, 12.4000950, -48.5850754, 12.4339142, -60.9908676, 60.9851685
18: -21.7861290, 30.0267792, -21.8475838, 30.0880013, -51.8741302, 51.8743629
19: -17.4347210, 19.8904972, -17.4500732, 19.9438400, -37.3785629, 37.3405685
20: -24.7404938, 13.9789848, -24.7550850, 14.0401278, -38.7806206, 38.7340698
21: -22.5820408, 21.8418579, -22.6218529, 21.9146538, -44.4966965, 44.4637108
22: -13.8099976, 27.3764763, -13.8330498, 27.4043770, -37.1117859, 37.1340866
23: -16.1444550, 22.8189468, -16.1666088, 22.8956146, -39.0400696, 38.9855576
24: -16.3217945, 23.8087845, -16.3454552, 23.8536148, -40.1515961, 40.1466522
25: -16.5917969, 27.9393082, -16.6014614, 27.9982376, -43.1052399, 43.1043930
26: -26.9842987, 35.3403320, -27.0245533, 35.4106636, -62.3949623, 62.3648834
27: -21.9217777, 19.5652351, -21.9493942, 19.5759983, -41.4977760, 41.5146294
28: -17.0480480, 25.0332642, -17.0687790, 25.0924625, -41.9023514, 41.8621521
29: -9.7998800, 22.6815510, -9.8244190, 22.7128029, -30.5520706, 30.5694885
30: -28.8358707, 15.5258942, -28.8721294, 15.5963221, -44.4321938, 44.3980255
31: -23.7011547, 24.1373711, -23.7235966, 24.2020149, -47.9031677, 47.8609695
32: -47.6180725, -1.9518666, -47.6519318, -1.9103446, -40.6533966, 40.6450882
33: -70.5675354, 5.4931774, -70.6113739, 5.5113668, -67.0086060, 67.0397034
34: -63.5245934, -5.6773028, -63.5700836, -5.6260343, -46.0632477, 46.1140289
35: -41.8615799, 14.2985439, -41.8972397, 14.3262796, -53.6275940, 53.6869888
36: -42.3246460, 16.0858574, -42.3491364, 16.1162968, -57.9186096, 57.9167404
37: -70.8866119, -1.2721729, -70.9019089, -1.2391663, -66.3018799, 66.3126984
38: -59.4645157, 8.7178307, -59.4976196, 8.7556877, -68.2201996, 68.2154541
39: -74.1530609, 1.0583577, -74.1785583, 1.0783033, -74.2454834, 74.2179260
40: -70.8608398, -18.7650566, -70.8920822, -18.7492981, -41.7043762, 41.7081451
41: -47.7580109, 3.7128868, -47.7838898, 3.7400322, -46.3826523, 46.3796921
42: -46.8795624, -3.8759985, -46.8901443, -3.8464026, -39.0223618, 38.9999733

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
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
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 1698
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
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1404
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
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 911
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
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 836
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
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1766
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

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -17.9818289, upper bound: 17.9855091
time: 50.47 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -18.0040200, upper bound: 17.9855091
time: 46.01 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 98.75 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 98.75
Output dim: 29, lower bound: -17.9776127, upper bound: 17.9685467
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 98.75
Output dim: 29, lower bound: -17.9873062, upper bound: 17.9682152
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 98.75
Output dim: 29, lower bound: -17.9776127, upper bound: 17.9815423
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 98.75
Output dim: 29, lower bound: -17.9873062, upper bound: 17.9812120
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 98.75
Output dim: 29, lower bound: -17.9818289, upper bound: 17.9725222
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 98.75
Output dim: 29, lower bound: -18.0040200, upper bound: 17.9725222
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 98.75
Output dim: 29, lower bound: -17.9818289, upper bound: 17.9855091
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 98.75
Output dim: 29, lower bound: -18.0040200, upper bound: 17.9855091
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0214291, upper bound: 17.9853360
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0214291, upper bound: 17.9982931
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0323986, upper bound: 17.9856141
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0323986, upper bound: 17.9985805
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -17.9965199, upper bound: 17.9915270
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -17.9965200, upper bound: 18.0045420
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0074155, upper bound: 17.9917707
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0074155, upper bound: 18.0047893
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0012727
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0142529
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0015505
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0145307
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -17.9965199, upper bound: 17.9935817
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -17.9965200, upper bound: 18.0065669
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0074155, upper bound: 17.9938350
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -17.9965200, upper bound: 18.0068257
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0031895
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0161403
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0034786
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0164335
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -17.9965199, upper bound: 18.0094729
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -17.9965200, upper bound: 18.0224766
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -17.9965199, upper bound: 18.0097208
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0074155, upper bound: 18.0227276
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0191376
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0214291, upper bound: 18.0321156
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0194189
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 98.75
Output dim: 29, lower bound: -18.0323986, upper bound: 18.0323986

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 55.62 + 3596.88 = 3652.50 seconds
