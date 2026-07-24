## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 45.034503
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-67.4471893, 43.9867935, -67.4471893, 43.9867935, -111.4339828, 111.4339828)
1: (-38.0577011, 35.2121086, -38.0577011, 35.2121086, -73.2698059, 73.2698059)
2: (-29.4427834, 37.8078308, -29.4427834, 37.8078308, -67.2506104, 67.2506104)
3: (-43.6639252, 37.4548798, -43.6639252, 37.4548798, -81.1188049, 81.1188049)
4: (-44.5875244, 39.5688057, -44.5875244, 39.5688057, -84.1563263, 84.1563263)
5: (-40.8208656, 41.8740158, -40.8208656, 41.8740158, -82.6948853, 82.6948853)
6: (-72.3137741, 13.1571198, -72.3137741, 13.1571198, -85.4708939, 85.4708939)
7: (-53.2639313, 32.0415268, -53.2639313, 32.0415268, -85.3054581, 85.3054581)
8: (-57.7919350, 39.3559418, -57.7919350, 39.3559418, -97.1478729, 97.1478729)
9: (-41.8387680, 42.6367416, -41.8387680, 42.6367416, -84.4755096, 84.4755096)
10: (-58.5155983, 48.8589668, -58.5155983, 48.8589668, -107.3745651, 107.3745651)
11: (-48.7890358, 27.8059158, -48.7890358, 27.8059158, -76.5949554, 76.5949554)
12: (-66.3382111, 41.5691986, -66.3382111, 41.5691986, -107.9074097, 107.9074097)
13: (-60.5002441, 50.0187111, -60.5002441, 50.0187111, -110.5189514, 110.5189514)
14: (-86.0621872, 36.1742096, -86.0621872, 36.1742096, -122.2363968, 122.2363968)
15: (-41.5096474, 44.9642563, -41.5096474, 44.9642563, -86.4739075, 86.4739075)
16: (-61.3557091, 39.3986778, -61.3557091, 39.3986778, -100.7543869, 100.7543869)
17: (-80.3320160, 33.0731964, -80.3320160, 33.0731964, -113.4052124, 113.4052124)
18: (-45.8324699, 45.7068939, -45.8324699, 45.7068939, -91.5393677, 91.5393677)
19: (-35.4370384, 30.1147995, -35.4370384, 30.1147995, -65.5518341, 65.5518341)
20: (-40.6649933, 26.7752533, -40.6649933, 26.7752533, -67.4402466, 67.4402466)
21: (-45.3317757, 33.9482727, -45.3317757, 33.9482727, -79.2800446, 79.2800446)
22: (-36.4696274, 39.4194336, -36.4696274, 39.4194336, -75.8890610, 75.8890610)
23: (-34.1263351, 34.8214836, -34.1263351, 34.8214836, -68.9478149, 68.9478149)
24: (-38.9773102, 35.2675400, -38.9773102, 35.2675400, -74.2448502, 74.2448502)
25: (-36.5572929, 42.6335678, -36.5572929, 42.6335678, -79.1908569, 79.1908569)
26: (-51.8876228, 54.8559837, -51.8876228, 54.8559837, -106.7436066, 106.7436066)
27: (-43.1078339, 31.4249229, -43.1078339, 31.4249229, -74.5327606, 74.5327606)
28: (-35.0236740, 38.0913277, -35.0236740, 38.0913277, -73.1150055, 73.1150055)
29: (-33.8305511, 32.2959061, -33.8305511, 32.2959061, -66.1264572, 66.1264572)
30: (-49.5392380, 30.3828545, -49.5392380, 30.3828545, -79.9220886, 79.9220886)
31: (-47.0840111, 37.2110939, -47.0840111, 37.2110939, -84.2951050, 84.2951050)
32: (-67.0599289, 15.8917522, -67.0599289, 15.8917522, -82.9516830, 82.9516830)
33: (-96.4942551, 32.1869125, -96.4942551, 32.1869125, -128.3750000, 128.3750000)
34: (-83.6580658, 15.7571354, -83.6580658, 15.7571354, -97.8325272, 97.8325272)
35: (-63.4767685, 33.3785858, -63.4767685, 33.3785858, -96.8553543, 96.8553543)
36: (-64.7697601, 34.9338455, -64.7697601, 34.9338455, -99.7036057, 99.7036057)
37: (-100.9295654, 21.9632225, -100.9295654, 21.9632225, -122.8927917, 122.8927917)
38: (-86.1509171, 33.3935394, -86.1509171, 33.3935394, -119.5444565, 119.5444565)
39: (-104.1541748, 26.6670990, -104.1541748, 26.6670990, -130.8212738, 130.8212738)
40: (-91.5092926, 3.2036600, -91.5092926, 3.2036600, -93.4074402, 93.4074326)
41: (-67.6376953, 22.2651768, -67.6376953, 22.2651768, -89.2062073, 89.2062073)
42: (-60.6191864, 15.0053864, -60.6191864, 15.0053864, -75.6245728, 75.6245728)

## BASE Result
execution time: IAR + LP analysis = 2.91 + 65.57 = 68.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 29, lower bound: -53.4893768, upper bound: 53.4893768


# Binary Search by BASE starts (time budget: 17931.52 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=66.12645721435547
rel_dist={29: [-48.278793808377486, 48.27879381030013]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=65.85055541992188
rel_dist={29: [-45.07674778233984, 45.07674777663058]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=65.57858276367188
rel_dist={29: [-42.58028969104383, 42.58028968925122]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=65.71456909179688
rel_dist={29: [-43.86410211785519, 43.86410211877009]}

## Binary Search Result
Binary search time: 540.48 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 17391.04 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.2186509, upper bound: 49.1518900
time: 69.83 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.2026015, upper bound: 49.2026014
time: 76.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 146.85 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 146.85
Output dim: 29, lower bound: -49.2186509, upper bound: 49.1518900
IS_A2, status: Status.UNKNOWN, split count: 1, time: 146.85
Output dim: 29, lower bound: -49.2026015, upper bound: 49.2026014

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -67.4241028, 43.9691849, -67.4449463, 43.9851303, -111.4092331, 111.4141312
1: -38.0452576, 35.1950378, -38.0565186, 35.2104721, -73.2557297, 73.2515564
2: -29.4349117, 37.7924118, -29.4419994, 37.8063431, -67.2412567, 67.2344131
3: -43.6466942, 37.4368362, -43.6622543, 37.4531937, -81.0998840, 81.0990906
4: -44.5588341, 39.5477715, -44.5847855, 39.5668373, -84.1256714, 84.1325531
5: -40.8166656, 41.8543777, -40.8204460, 41.8721848, -82.6888504, 82.6748199
6: -72.2819138, 13.1482906, -72.3107300, 13.1562805, -83.1973267, 83.2214355
7: -53.2542725, 32.0250435, -53.2629776, 32.0398712, -85.2941437, 85.2880249
8: -57.7837524, 39.3276825, -57.7911530, 39.3531761, -97.1369324, 97.1188354
9: -41.7618484, 42.6260910, -41.8314133, 42.6357498, -84.3975983, 84.4575043
10: -58.4933395, 48.8165512, -58.5134964, 48.8549423, -107.3482819, 107.3300476
11: -48.7610054, 27.7154675, -48.7862930, 27.7970181, -76.5580215, 76.5017624
12: -66.3109055, 41.5532341, -66.3355408, 41.5676651, -107.5510864, 107.5590820
13: -60.4191322, 49.9962234, -60.4924660, 50.0166054, -110.4357376, 110.4886932
14: -86.0290222, 36.1202774, -86.0589981, 36.1690750, -122.1980972, 122.1792755
15: -41.4489746, 44.9460793, -41.5036240, 44.9625244, -86.4114990, 86.4497070
16: -61.3155556, 39.3793259, -61.3517761, 39.3967972, -100.7123566, 100.7311020
17: -80.2972641, 32.9762955, -80.3286133, 33.0639725, -113.3612366, 113.3049088
18: -45.8070526, 45.6773338, -45.8300514, 45.7040100, -91.5110626, 91.5073853
19: -35.4148064, 30.0358467, -35.4349098, 30.1073093, -65.5221176, 65.4707565
20: -40.6463013, 26.7496262, -40.6632309, 26.7727890, -67.4190903, 67.4128571
21: -45.3004723, 33.8702164, -45.3287735, 33.9406281, -79.2411041, 79.1989899
22: -36.4442444, 39.4071846, -36.4671783, 39.4182434, -75.8624878, 75.8743591
23: -34.1080399, 34.7547913, -34.1245880, 34.8149147, -68.9229584, 68.8793793
24: -38.9556770, 35.2179832, -38.9752274, 35.2628212, -74.2184982, 74.1932068
25: -36.5341148, 42.5710716, -36.5550461, 42.6276474, -79.1617584, 79.1261139
26: -51.8562584, 54.8376122, -51.8846283, 54.8542023, -106.7104645, 106.7222443
27: -43.0847702, 31.4085827, -43.1056442, 31.4233589, -74.5081329, 74.5142288
28: -35.0060806, 38.0555801, -35.0220032, 38.0879059, -73.0939865, 73.0775833
29: -33.8021812, 32.2545853, -33.8278275, 32.2919159, -66.0941010, 66.0824127
30: -49.5133018, 30.3148060, -49.5367699, 30.3763962, -79.8896942, 79.8515778
31: -47.0561867, 37.1188965, -47.0813751, 37.2023621, -84.2585449, 84.2002716
32: -66.9891434, 15.8724270, -67.0531998, 15.8899269, -80.6235733, 80.6703568
33: -96.4113007, 32.1759300, -96.4864502, 32.1858521, -124.7005310, 124.7625046
34: -83.6097717, 15.7388000, -83.6534576, 15.7553139, -92.4151764, 92.4375381
35: -63.4354172, 33.3702469, -63.4728165, 33.3778343, -96.8132477, 96.8430634
36: -64.6954803, 34.9204254, -64.7625885, 34.9325562, -99.6280365, 99.6830139
37: -100.8906860, 21.9551201, -100.9258575, 21.9624481, -122.8531342, 122.8809814
38: -86.0990906, 33.3707848, -86.1459808, 33.3913422, -119.4904327, 119.5167694
39: -104.0569763, 26.6585693, -104.1448746, 26.6662102, -130.7231903, 130.8034363
40: -91.4443054, 3.1938534, -91.5031128, 3.2027330, -90.2122345, 90.2636795
41: -67.5957184, 22.2521381, -67.6336670, 22.2638817, -87.4188232, 87.4450684
42: -60.5945625, 14.9930954, -60.6168137, 15.0042095, -73.7905273, 73.7908249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1639

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1518900, upper bound: 49.1518900
time: 77.23 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1518900, upper bound: 49.1518900
time: 61.01 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -67.4959259, 44.0195923, -67.4450226, 43.9849243, -111.4808502, 111.4646149
1: -38.0917435, 35.2418556, -38.0563087, 35.2105865, -73.3023300, 73.2981644
2: -29.4580460, 37.8343887, -29.4415989, 37.8064117, -67.2644577, 67.2759857
3: -43.6686363, 37.5141373, -43.6612854, 37.4531021, -81.1217346, 81.1754227
4: -44.5932007, 39.6324539, -44.5828857, 39.5673370, -84.1605377, 84.2153397
5: -40.8353195, 41.9103394, -40.8204002, 41.8716049, -82.7069244, 82.7307434
6: -72.3270569, 13.2128582, -72.3111267, 13.1564159, -83.2745667, 83.2777863
7: -53.3150978, 32.0465050, -53.2628555, 32.0383148, -85.3534088, 85.3093567
8: -57.8236465, 39.3821983, -57.7910919, 39.3526535, -97.1763000, 97.1732941
9: -41.8505096, 42.7237167, -41.8307915, 42.6355171, -84.4860229, 84.5545044
10: -58.5922127, 48.8829765, -58.5141068, 48.8534546, -107.4456635, 107.3970795
11: -49.0592079, 27.8105030, -48.7865143, 27.8014107, -76.8606186, 76.5970154
12: -66.3652802, 41.6292725, -66.3349609, 41.5682068, -107.6102753, 107.6323700
13: -60.5091972, 50.1594810, -60.4942551, 50.0163574, -110.5255585, 110.6537323
14: -86.2087250, 36.1774368, -86.0591507, 36.1719780, -122.3807068, 122.2365875
15: -41.5217171, 45.0794029, -41.5060425, 44.9628868, -86.4846039, 86.5854492
16: -61.4743958, 39.4169655, -61.3532410, 39.3959084, -100.8703003, 100.7702026
17: -80.6034775, 33.0777664, -80.3293304, 33.0682831, -113.6717606, 113.4070969
18: -45.9073524, 45.7207909, -45.8296509, 45.7040787, -91.6114349, 91.5504456
19: -35.5884514, 30.1116009, -35.4353447, 30.1117630, -65.7002106, 65.5469437
20: -40.7368851, 26.7777939, -40.6632118, 26.7713356, -67.5082245, 67.4410095
21: -45.5208397, 33.9427834, -45.3296471, 33.9446640, -79.4654999, 79.2724304
22: -36.5489273, 39.4252052, -36.4671516, 39.4183464, -75.9672699, 75.8923569
23: -34.2918701, 34.8287659, -34.1251030, 34.8195992, -69.1114655, 68.9538727
24: -39.1308250, 35.2664108, -38.9754066, 35.2631836, -74.3940125, 74.2418213
25: -36.6780777, 42.6343117, -36.5553970, 42.6291084, -79.3071899, 79.1897125
26: -51.9784889, 54.8755112, -51.8842316, 54.8550110, -106.8334961, 106.7597427
27: -43.1833572, 31.4301414, -43.1056252, 31.4228497, -74.6062088, 74.5357666
28: -35.1460381, 38.0970230, -35.0222588, 38.0888596, -73.2348938, 73.1192780
29: -34.0077400, 32.2971497, -33.8278961, 32.2931786, -66.3009186, 66.1250458
30: -49.7524719, 30.3781452, -49.5371895, 30.3770294, -80.1295013, 79.9153366
31: -47.2395172, 37.2110825, -47.0819016, 37.2084084, -84.4479218, 84.2929840
32: -67.0737762, 16.0306511, -67.0556259, 15.8902283, -80.7098465, 80.8307648
33: -96.5323563, 32.3340988, -96.4917068, 32.1850433, -124.8122559, 124.9237213
34: -83.6744080, 15.8904438, -83.6553802, 15.7559166, -92.4720230, 92.5738907
35: -63.5047493, 33.4696655, -63.4743996, 33.3777618, -96.8825073, 96.9440613
36: -64.7727356, 35.0557442, -64.7636261, 34.9330940, -99.7058258, 99.8193665
37: -100.9780731, 22.0094032, -100.9257202, 21.9621658, -122.9402390, 122.9351196
38: -86.1432571, 33.5402031, -86.1434631, 33.3924141, -119.5356750, 119.6836700
39: -104.1855621, 26.8324165, -104.1495209, 26.6659813, -130.8515472, 130.9819336
40: -91.5264664, 3.3505287, -91.5037460, 3.2029743, -90.2979584, 90.4155350
41: -67.6576767, 22.3406715, -67.6344299, 22.2642899, -87.4824677, 87.5315552
42: -60.6337433, 15.0519371, -60.6169586, 15.0043449, -73.8723450, 73.8323517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1975568, upper bound: 49.1381865
time: 84.21 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1975568, upper bound: 49.1975566
time: 79.82 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 166.49 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 166.49
Output dim: 29, lower bound: -49.1518900, upper bound: 49.1518900
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 166.49
Output dim: 29, lower bound: -49.1518900, upper bound: 49.1518900
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 166.49
Output dim: 29, lower bound: -49.1975568, upper bound: 49.1381865
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 166.49
Output dim: 29, lower bound: -49.1975568, upper bound: 49.1975566

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -67.4241028, 43.9691849, -67.4241028, 43.9691849, -111.3932877, 111.3932877
1: -38.0452576, 35.1950378, -38.0452576, 35.1950378, -73.2402954, 73.2402954
2: -29.4349117, 37.7924118, -29.4349117, 37.7924118, -67.2273254, 67.2273254
3: -43.6466942, 37.4368362, -43.6466942, 37.4368362, -81.0835266, 81.0835266
4: -44.5588341, 39.5477715, -44.5588341, 39.5477715, -84.1066055, 84.1066055
5: -40.8166656, 41.8543777, -40.8166656, 41.8543777, -82.6710434, 82.6710434
6: -72.2819138, 13.1482906, -72.2819138, 13.1482906, -83.1819229, 83.1819305
7: -53.2542725, 32.0250435, -53.2542725, 32.0250435, -85.2793121, 85.2793121
8: -57.7837524, 39.3276825, -57.7837524, 39.3276825, -97.1114349, 97.1114349
9: -41.7618484, 42.6260910, -41.7618484, 42.6260910, -84.3879395, 84.3879395
10: -58.4933395, 48.8165512, -58.4933395, 48.8165512, -107.3098907, 107.3098907
11: -48.7610054, 27.7154675, -48.7610054, 27.7154675, -76.4764709, 76.4764709
12: -66.3109055, 41.5532341, -66.3109055, 41.5532341, -107.5357361, 107.5357361
13: -60.4191322, 49.9962234, -60.4191322, 49.9962234, -110.4153595, 110.4153595
14: -86.0290222, 36.1202774, -86.0290222, 36.1202774, -122.1492996, 122.1492996
15: -41.4489746, 44.9460793, -41.4489746, 44.9460793, -86.3950500, 86.3950500
16: -61.3155556, 39.3793259, -61.3155556, 39.3793259, -100.6948853, 100.6948853
17: -80.2972641, 32.9762955, -80.2972641, 32.9762955, -113.2735596, 113.2735596
18: -45.8070526, 45.6773338, -45.8070526, 45.6773338, -91.4843903, 91.4843903
19: -35.4148064, 30.0358467, -35.4148064, 30.0358467, -65.4506531, 65.4506531
20: -40.6463013, 26.7496262, -40.6463013, 26.7496262, -67.3959274, 67.3959274
21: -45.3004723, 33.8702164, -45.3004723, 33.8702164, -79.1706848, 79.1706848
22: -36.4442444, 39.4071846, -36.4442444, 39.4071846, -75.8514252, 75.8514252
23: -34.1080399, 34.7547913, -34.1080399, 34.7547913, -68.8628311, 68.8628311
24: -38.9556770, 35.2179832, -38.9556770, 35.2179832, -74.1736603, 74.1736603
25: -36.5341148, 42.5710716, -36.5341148, 42.5710716, -79.1051865, 79.1051865
26: -51.8562584, 54.8376122, -51.8562584, 54.8376122, -106.6938705, 106.6938705
27: -43.0847702, 31.4085827, -43.0847702, 31.4085827, -74.4933548, 74.4933548
28: -35.0060806, 38.0555801, -35.0060806, 38.0555801, -73.0616608, 73.0616608
29: -33.8021812, 32.2545853, -33.8021812, 32.2545853, -66.0567627, 66.0567627
30: -49.5133018, 30.3148060, -49.5133018, 30.3148060, -79.8281097, 79.8281097
31: -47.0561867, 37.1188965, -47.0561867, 37.1188965, -84.1750793, 84.1750793
32: -66.9891434, 15.8724270, -66.9891434, 15.8724270, -80.6055908, 80.6055984
33: -96.4113007, 32.1759300, -96.4113007, 32.1759300, -124.6904602, 124.6904449
34: -83.6097717, 15.7388000, -83.6097717, 15.7388000, -92.3991318, 92.3991394
35: -63.4354172, 33.3702469, -63.4354172, 33.3702469, -96.8056641, 96.8056641
36: -64.6954803, 34.9204254, -64.6954803, 34.9204254, -99.6159058, 99.6159058
37: -100.8906860, 21.9551201, -100.8906860, 21.9551201, -122.8458099, 122.8458099
38: -86.0990906, 33.3707848, -86.0990906, 33.3707848, -119.4698792, 119.4698792
39: -104.0569763, 26.6585693, -104.0569763, 26.6585693, -130.7155457, 130.7155457
40: -91.4443054, 3.1938534, -91.4443054, 3.1938534, -90.2022705, 90.2022858
41: -67.5957184, 22.2521381, -67.5957184, 22.2521381, -87.4067841, 87.4067764
42: -60.5945625, 14.9930954, -60.5945625, 14.9930954, -73.7656097, 73.7656097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1662

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1521139, upper bound: 49.1466852
time: 81.22 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.2112018, upper bound: 49.1466852
time: 58.06 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -67.4241028, 43.9691849, -67.4959259, 44.0195923, -111.4436951, 111.4651108
1: -38.0452576, 35.1950378, -38.0917435, 35.2418556, -73.2871094, 73.2867813
2: -29.4349117, 37.7924118, -29.4580460, 37.8343887, -67.2693024, 67.2504578
3: -43.6466942, 37.4368362, -43.6686363, 37.5141373, -81.1608276, 81.1054688
4: -44.5588341, 39.5477715, -44.5932007, 39.6324539, -84.1912842, 84.1409760
5: -40.8166656, 41.8543777, -40.8353195, 41.9103394, -82.7270050, 82.6896973
6: -72.2819138, 13.1482906, -72.3270569, 13.2128582, -83.2398682, 83.2284012
7: -53.2542725, 32.0250435, -53.3150978, 32.0465050, -85.3007812, 85.3401413
8: -57.7837524, 39.3276825, -57.8236465, 39.3821983, -97.1659546, 97.1513290
9: -41.7618484, 42.6260910, -41.8505096, 42.7237167, -84.4855652, 84.4766006
10: -58.4933395, 48.8165512, -58.5922127, 48.8829765, -107.3763123, 107.4087677
11: -48.7610054, 27.7154675, -49.0592079, 27.8105030, -76.5715103, 76.7746735
12: -66.3109055, 41.5532341, -66.3652802, 41.6292725, -107.6097107, 107.5917587
13: -60.4191322, 49.9962234, -60.5091972, 50.1594810, -110.5786133, 110.5054169
14: -86.0290222, 36.1202774, -86.2087250, 36.1774368, -122.2064590, 122.3290024
15: -41.4489746, 44.9460793, -41.5217171, 45.0794029, -86.5283813, 86.4677963
16: -61.3155556, 39.3793259, -61.4743958, 39.4169655, -100.7325211, 100.8537216
17: -80.2972641, 32.9762955, -80.6034775, 33.0777664, -113.3750305, 113.5797729
18: -45.8070526, 45.6773338, -45.9073524, 45.7207909, -91.5278473, 91.5846863
19: -35.4148064, 30.0358467, -35.5884514, 30.1116009, -65.5264053, 65.6242981
20: -40.6463013, 26.7496262, -40.7368851, 26.7777939, -67.4240952, 67.4865112
21: -45.3004723, 33.8702164, -45.5208397, 33.9427834, -79.2432556, 79.3910522
22: -36.4442444, 39.4071846, -36.5489273, 39.4252052, -75.8694458, 75.9561157
23: -34.1080399, 34.7547913, -34.2918701, 34.8287659, -68.9368057, 69.0466614
24: -38.9556770, 35.2179832, -39.1308250, 35.2664108, -74.2220917, 74.3488083
25: -36.5341148, 42.5710716, -36.6780777, 42.6343117, -79.1684265, 79.2491455
26: -51.8562584, 54.8376122, -51.9784889, 54.8755112, -106.7317657, 106.8161011
27: -43.0847702, 31.4085827, -43.1833572, 31.4301414, -74.5149078, 74.5919418
28: -35.0060806, 38.0555801, -35.1460381, 38.0970230, -73.1031036, 73.2016144
29: -33.8021812, 32.2545853, -34.0077400, 32.2971497, -66.0993347, 66.2623291
30: -49.5133018, 30.3148060, -49.7524719, 30.3781452, -79.8914490, 80.0672760
31: -47.0561867, 37.1188965, -47.2395172, 37.2110825, -84.2672729, 84.3584137
32: -66.9891434, 15.8724270, -67.0737762, 16.0306511, -80.7637329, 80.6918640
33: -96.4113007, 32.1759300, -96.5323563, 32.3340988, -124.8474121, 124.8060684
34: -83.6097717, 15.7388000, -83.6744080, 15.8904438, -92.5362701, 92.4548340
35: -63.4354172, 33.3702469, -63.5047493, 33.4696655, -96.9050827, 96.8750000
36: -64.6954803, 34.9204254, -64.7727356, 35.0557442, -99.7512207, 99.6931610
37: -100.8906860, 21.9551201, -100.9780731, 22.0094032, -122.9000854, 122.9331970
38: -86.0990906, 33.3707848, -86.1432571, 33.5402031, -119.6392975, 119.5140381
39: -104.0569763, 26.6585693, -104.1855621, 26.8324165, -130.8893890, 130.8441315
40: -91.4443054, 3.1938534, -91.5264664, 3.3505287, -90.3544464, 90.2823334
41: -67.5957184, 22.2521381, -67.6576767, 22.3406715, -87.4931259, 87.4670868
42: -60.5945625, 14.9930954, -60.6337433, 15.0519371, -73.8147736, 73.8036270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=366, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1662

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1521139, upper bound: 49.1466852
time: 62.58 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.2112018, upper bound: 49.1466852
time: 93.13 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -67.4812164, 43.9778366, -67.3847351, 43.8146973, -111.2959137, 111.3625717
1: -38.0874290, 35.2076454, -38.0385666, 35.0694351, -73.1568604, 73.2462158
2: -29.4531174, 37.8049164, -29.4214649, 37.6849365, -67.1380539, 67.2263794
3: -43.6660538, 37.4700127, -43.6507912, 37.2710495, -80.9371033, 81.1208038
4: -44.5890121, 39.5921860, -44.5657997, 39.4013596, -83.9903717, 84.1579895
5: -40.8322563, 41.8751907, -40.8078575, 41.7272797, -82.5595398, 82.6830444
6: -72.3062363, 13.2067719, -72.2262115, 13.1315727, -83.2280121, 83.1797333
7: -53.3111496, 32.0149078, -53.2469025, 31.9079227, -85.2190704, 85.2618103
8: -57.8190918, 39.3414536, -57.7724533, 39.1842194, -97.0033112, 97.1139069
9: -41.8432159, 42.6939316, -41.8007507, 42.5149231, -84.3581390, 84.4946823
10: -58.5797615, 48.8716278, -58.4627190, 48.8073120, -107.3870697, 107.3343506
11: -49.0317268, 27.8061733, -48.6757736, 27.7837029, -76.8154297, 76.4819489
12: -66.3311081, 41.6244392, -66.1949158, 41.5487213, -107.5563812, 107.4877930
13: -60.5020370, 50.1408195, -60.4647102, 49.9399300, -110.4419708, 110.6055298
14: -86.1800537, 36.1738853, -85.9435883, 36.1576920, -122.3377457, 122.1174774
15: -41.5168381, 45.0484695, -41.4861069, 44.8370667, -86.3539047, 86.5345764
16: -61.4622040, 39.4024658, -61.3030472, 39.3371964, -100.7994003, 100.7055130
17: -80.5766296, 33.0690880, -80.2204666, 33.0326691, -113.6092987, 113.2895508
18: -45.8847275, 45.7151642, -45.7377625, 45.6809158, -91.5656433, 91.4529266
19: -35.5586014, 30.1089821, -35.3124809, 30.1010914, -65.6596909, 65.4214630
20: -40.7051392, 26.7741127, -40.5333099, 26.7561703, -67.4613113, 67.3074188
21: -45.4927711, 33.9402657, -45.2152824, 33.9343605, -79.4271317, 79.1555481
22: -36.5186844, 39.4219360, -36.3432693, 39.4049149, -75.9235992, 75.7652054
23: -34.2579193, 34.8256569, -33.9854813, 34.8069191, -69.0648346, 68.8111420
24: -39.0898209, 35.2637405, -38.8073540, 35.2522659, -74.3420868, 74.0710907
25: -36.6378555, 42.6306915, -36.3899307, 42.6142807, -79.2521362, 79.0206223
26: -51.9488487, 54.8697739, -51.7634544, 54.8312073, -106.7800598, 106.6332245
27: -43.1571846, 31.4267979, -42.9988403, 31.4090385, -74.5662231, 74.4256363
28: -35.1129150, 38.0931473, -34.8860664, 38.0729904, -73.1859055, 72.9792175
29: -33.9728661, 32.2943993, -33.6854439, 32.2817764, -66.2546387, 65.9798431
30: -49.7169762, 30.3743534, -49.3929405, 30.3614807, -80.0784607, 79.7672958
31: -47.1962357, 37.2074242, -46.9037018, 37.1933327, -84.3895721, 84.1111298
32: -67.0485611, 16.0261707, -66.9544983, 15.8720379, -80.6653748, 80.7239151
33: -96.5185089, 32.3270950, -96.4350739, 32.1560783, -124.7672424, 124.8583755
34: -83.6642532, 15.8852005, -83.6138458, 15.7345982, -92.4357452, 92.5133667
35: -63.4965134, 33.4639091, -63.4405556, 33.3540192, -96.8505325, 96.9044647
36: -64.7544556, 35.0518608, -64.6907349, 34.9173355, -99.6717911, 99.7425995
37: -100.9439392, 22.0044174, -100.7891846, 21.9419556, -122.8858948, 122.7936020
38: -86.1201019, 33.5331192, -86.0504074, 33.3634987, -119.4835968, 119.5835266
39: -104.1636353, 26.8283539, -104.0602112, 26.6495056, -130.8131409, 130.8885651
40: -91.5075912, 3.3456116, -91.4266739, 3.1827202, -90.2582855, 90.3341064
41: -67.6454010, 22.3355560, -67.5839081, 22.2434959, -87.4477844, 87.4738388
42: -60.6208687, 15.0478611, -60.5648651, 14.9878883, -73.8376312, 73.7556076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1591

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1827025, upper bound: 49.0586594
time: 182.71 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1798740, upper bound: 49.1204229
time: 74.18 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -67.4936905, 44.0140953, -67.7265854, 43.9924774, -111.4861679, 111.7406769
1: -38.0908775, 35.2378998, -38.3035393, 35.2216873, -73.3125610, 73.5414429
2: -29.4570370, 37.8310394, -29.6132832, 37.8208008, -67.2778397, 67.4443207
3: -43.6678238, 37.5090485, -43.9177361, 37.4814606, -81.1492844, 81.4267883
4: -44.5921478, 39.6279373, -44.8249130, 39.5813293, -84.1734772, 84.4528503
5: -40.8343506, 41.9062119, -41.0365601, 41.8934708, -82.7278214, 82.9427719
6: -72.3229599, 13.2119064, -72.3316727, 13.2456245, -83.3779984, 83.2918396
7: -53.3139954, 32.0428543, -53.4976616, 32.0507507, -85.3647461, 85.5405121
8: -57.8224220, 39.3774796, -58.0663834, 39.3742294, -97.1966553, 97.4438629
9: -41.8494301, 42.7195854, -42.0670090, 42.6494331, -84.4988632, 84.7865906
10: -58.5904770, 48.8805199, -58.6609192, 48.9089165, -107.4993896, 107.5414429
11: -49.0553970, 27.8097916, -48.8478012, 27.9594040, -77.0148010, 76.6575928
12: -66.3612518, 41.6279907, -66.3681183, 41.7666626, -107.8238525, 107.6634140
13: -60.5078163, 50.1526566, -60.6282196, 50.0473633, -110.5551758, 110.7808762
14: -86.2044220, 36.1763687, -86.1803436, 36.2532349, -122.4576569, 122.3567123
15: -41.5206909, 45.0752678, -41.7111664, 44.9866982, -86.5073853, 86.7864380
16: -61.4723625, 39.4140778, -61.5844498, 39.4181671, -100.8905334, 100.9985275
17: -80.6000214, 33.0760002, -80.4426880, 33.2521362, -113.8521576, 113.5186920
18: -45.9039650, 45.7198715, -45.8823013, 45.8964386, -91.8003998, 91.6021729
19: -35.5846786, 30.1110191, -35.4760704, 30.2981281, -65.8828049, 65.5870895
20: -40.7328186, 26.7772903, -40.6928902, 26.9560833, -67.6889038, 67.4701843
21: -45.5170670, 33.9423294, -45.3881302, 34.0905228, -79.6075897, 79.3304596
22: -36.5449066, 39.4245491, -36.5047607, 39.5912933, -76.1362000, 75.9293060
23: -34.2877655, 34.8280563, -34.1535416, 35.0125809, -69.3003464, 68.9815979
24: -39.1255531, 35.2659492, -39.0066223, 35.4429131, -74.5684662, 74.2725677
25: -36.6732864, 42.6334763, -36.6006508, 42.8806458, -79.5539322, 79.2341309
26: -51.9739532, 54.8745003, -51.9331398, 55.0903397, -107.0642929, 106.8076401
27: -43.1786842, 31.4296360, -43.1239738, 31.5021286, -74.6808167, 74.5536118
28: -35.1419067, 38.0961533, -35.0472260, 38.2925568, -73.4344635, 73.1433792
29: -34.0032120, 32.2965050, -33.8718758, 32.4286652, -66.4318771, 66.1683807
30: -49.7478065, 30.3773041, -49.5729408, 30.5417976, -80.2896042, 79.9502411
31: -47.2341766, 37.2105331, -47.1251678, 37.4400368, -84.6742096, 84.3357010
32: -67.0702820, 16.0298672, -67.0776901, 15.9915543, -80.8116302, 80.8502884
33: -96.5290833, 32.3329391, -96.5221710, 32.3024292, -124.9404907, 124.9559479
34: -83.6718369, 15.8895569, -83.6735001, 15.8938599, -92.6176834, 92.5873489
35: -63.5033112, 33.4684753, -63.5009041, 33.4946518, -96.9979630, 96.9693756
36: -64.7694244, 35.0549698, -64.7789459, 35.0710793, -99.8404999, 99.8339157
37: -100.9732971, 22.0083618, -100.9748688, 22.1057320, -123.0790253, 122.9832306
38: -86.1395721, 33.5389824, -86.1868591, 33.5433121, -119.6828842, 119.7258453
39: -104.1810303, 26.8314590, -104.2022247, 26.7647781, -130.9458008, 131.0336914
40: -91.5220184, 3.3497057, -91.5358658, 3.2708654, -90.3723602, 90.4468460
41: -67.6550064, 22.3398285, -67.6551971, 22.3268967, -87.5522003, 87.5510788
42: -60.6314316, 15.0511341, -60.6425552, 15.1005325, -74.0221634, 73.8380661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1591

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1827025, upper bound: 49.1181412
time: 78.20 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1798740, upper bound: 49.1798736
time: 266.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 347.48 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 347.48
Output dim: 29, lower bound: -49.1521139, upper bound: 49.1466852
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 347.48
Output dim: 29, lower bound: -49.2112018, upper bound: 49.1466852
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 347.48
Output dim: 29, lower bound: -49.1521139, upper bound: 49.1466852
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 347.48
Output dim: 29, lower bound: -49.2112018, upper bound: 49.1466852
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 347.48
Output dim: 29, lower bound: -49.1827025, upper bound: 49.0586594
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 347.48
Output dim: 29, lower bound: -49.1798740, upper bound: 49.1204229
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 347.48
Output dim: 29, lower bound: -49.1827025, upper bound: 49.1181412
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 347.48
Output dim: 29, lower bound: -49.1798740, upper bound: 49.1798736

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -67.3638458, 43.7989197, -67.4093475, 43.9274673, -111.2913132, 111.2082672
1: -38.0275726, 35.0538559, -38.0409927, 35.1608429, -73.1884155, 73.0948486
2: -29.4147797, 37.6708984, -29.4299889, 37.7629128, -67.1776886, 67.1008911
3: -43.6362000, 37.2547913, -43.6440887, 37.3927307, -81.0289307, 80.8988800
4: -44.5417404, 39.3818321, -44.5546379, 39.5074844, -84.0492249, 83.9364700
5: -40.8040848, 41.7100143, -40.8135529, 41.8192673, -82.6233521, 82.5235672
6: -72.1969757, 13.1233864, -72.2610855, 13.1421852, -83.0838928, 83.1353302
7: -53.2382927, 31.8946629, -53.2503242, 31.9934387, -85.2317352, 85.1449890
8: -57.7651100, 39.1593323, -57.7792282, 39.2870026, -97.0521088, 96.9385605
9: -41.7317657, 42.5054932, -41.7545242, 42.5962830, -84.3280487, 84.2600174
10: -58.4420395, 48.7703819, -58.4808693, 48.8052139, -107.2472534, 107.2512512
11: -48.6502342, 27.6977558, -48.7335052, 27.7111588, -76.3613892, 76.4312592
12: -66.1708832, 41.5337524, -66.2767258, 41.5484238, -107.3911896, 107.4819183
13: -60.3895988, 49.9198265, -60.4119797, 49.9774857, -110.3670807, 110.3318024
14: -85.9133835, 36.1060181, -86.0003815, 36.1167412, -122.0301208, 122.1063995
15: -41.4290428, 44.8202667, -41.4440956, 44.9151382, -86.3441772, 86.2643585
16: -61.2654495, 39.3205833, -61.3033791, 39.3648224, -100.6302719, 100.6239624
17: -80.1883774, 32.9406586, -80.2704468, 32.9675484, -113.1559296, 113.2111053
18: -45.7151566, 45.6542282, -45.7844582, 45.6717186, -91.3868713, 91.4386902
19: -35.2919617, 30.0251617, -35.3849640, 30.0332222, -65.3251801, 65.4101257
20: -40.5164146, 26.7343979, -40.6145630, 26.7459164, -67.2623291, 67.3489609
21: -45.1861687, 33.8599472, -45.2724533, 33.8677521, -79.0539246, 79.1324005
22: -36.3203621, 39.3938484, -36.4140015, 39.4039612, -75.7243195, 75.8078461
23: -33.9684563, 34.7421112, -34.0741348, 34.7517014, -68.7201538, 68.8162460
24: -38.7875938, 35.2070770, -38.9146690, 35.2153168, -74.0029144, 74.1217499
25: -36.3686867, 42.5562363, -36.4939232, 42.5674133, -78.9360962, 79.0501556
26: -51.7355881, 54.8137970, -51.8266716, 54.8317719, -106.5673599, 106.6404724
27: -42.9780350, 31.3948441, -43.0586014, 31.4052467, -74.3832855, 74.4534454
28: -34.8698578, 38.0397377, -34.9729843, 38.0517273, -72.9215851, 73.0127258
29: -33.6597252, 32.2431297, -33.7672806, 32.2517929, -65.9115143, 66.0104065
30: -49.3690910, 30.2992897, -49.4778214, 30.3109818, -79.6800690, 79.7771149
31: -46.8779678, 37.1038208, -47.0129356, 37.1151924, -83.9931641, 84.1167603
32: -66.8879547, 15.8542328, -66.9639130, 15.8679619, -80.4987183, 80.5610962
33: -96.3545532, 32.1469765, -96.3974609, 32.1688919, -124.6250458, 124.6453705
34: -83.5681152, 15.7175636, -83.5995941, 15.7336330, -92.3385773, 92.3628464
35: -63.4015465, 33.3464622, -63.4271317, 33.3644753, -96.7660217, 96.7735901
36: -64.6225586, 34.9046173, -64.6772461, 34.9165878, -99.5391464, 99.5818634
37: -100.7541504, 21.9347897, -100.8565598, 21.9501572, -122.7043076, 122.7913513
38: -86.0060043, 33.3418007, -86.0758820, 33.3636246, -119.3696289, 119.4176788
39: -103.9675903, 26.6420383, -104.0350342, 26.6545372, -130.6221313, 130.6770782
40: -91.3671646, 3.1736250, -91.4253845, 3.1889105, -90.1207581, 90.1625977
41: -67.5452118, 22.2313042, -67.5834198, 22.2469940, -87.3489685, 87.3720551
42: -60.5424652, 14.9766426, -60.5817032, 14.9890213, -73.6888275, 73.7308807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.0773161, upper bound: 49.2010725
time: 84.28 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1400608, upper bound: 49.1991301
time: 95.30 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -67.7056274, 43.9768219, -67.4218597, 43.9636765, -111.6693039, 111.3986816
1: -38.2925339, 35.2061424, -38.0444336, 35.1910896, -73.4836273, 73.2505798
2: -29.6065636, 37.8067856, -29.4339523, 37.7890091, -67.3955688, 67.2407379
3: -43.9030838, 37.4652367, -43.6458664, 37.4317780, -81.3348618, 81.1110992
4: -44.8008041, 39.5618019, -44.5577240, 39.5432434, -84.3440475, 84.1195221
5: -41.0328140, 41.8762245, -40.8157005, 41.8502426, -82.8830566, 82.6919250
6: -72.3024597, 13.2374477, -72.2778091, 13.1473312, -83.1959534, 83.2853470
7: -53.4890671, 32.0374985, -53.2531586, 32.0213432, -85.5104065, 85.2906570
8: -58.0590630, 39.3493080, -57.7825699, 39.3229866, -97.3820496, 97.1318817
9: -41.9980621, 42.6400681, -41.7607422, 42.6219368, -84.6199951, 84.4008102
10: -58.6401443, 48.8719482, -58.4915581, 48.8141251, -107.4542694, 107.3635101
11: -48.8223381, 27.8734245, -48.7572098, 27.7147198, -76.5370560, 76.6306305
12: -66.3440170, 41.7517204, -66.3068848, 41.5519829, -107.5668793, 107.7493515
13: -60.5530968, 50.0271873, -60.4177475, 49.9893875, -110.5424805, 110.4449310
14: -86.1501389, 36.2015533, -86.0247650, 36.1192017, -122.2693405, 122.2263184
15: -41.6541557, 44.9699326, -41.4479599, 44.9419479, -86.5960999, 86.4178925
16: -61.5467720, 39.4015694, -61.3135643, 39.3764305, -100.9232025, 100.7151337
17: -80.4106522, 33.1601067, -80.2938004, 32.9745178, -113.3851700, 113.4539032
18: -45.8597603, 45.8697205, -45.8036766, 45.6764221, -91.5361786, 91.6734009
19: -35.4555664, 30.2221756, -35.4110527, 30.0352631, -65.4908295, 65.6332245
20: -40.6760330, 26.9342957, -40.6422348, 26.7490845, -67.4251175, 67.5765305
21: -45.3590164, 34.0161209, -45.2967949, 33.8697624, -79.2287750, 79.3129120
22: -36.4818420, 39.5801773, -36.4402313, 39.4065361, -75.8883820, 76.0204086
23: -34.1364594, 34.9478073, -34.1039581, 34.7540817, -68.8905411, 69.0517654
24: -38.9868584, 35.3977013, -38.9503555, 35.2175484, -74.2044067, 74.3480530
25: -36.5794601, 42.8226089, -36.5292816, 42.5702515, -79.1497116, 79.3518906
26: -51.9052963, 55.0729904, -51.8517647, 54.8365479, -106.7418442, 106.9247589
27: -43.1031342, 31.4878788, -43.0801010, 31.4080963, -74.5112305, 74.5679779
28: -35.0310822, 38.2592888, -35.0019188, 38.0546951, -73.0857773, 73.2612076
29: -33.8461723, 32.3900490, -33.7975883, 32.2539177, -66.1000900, 66.1876373
30: -49.5490875, 30.4795780, -49.5086670, 30.3139629, -79.8630524, 79.9882431
31: -47.0994720, 37.3504906, -47.0508156, 37.1183395, -84.2178116, 84.4013062
32: -67.0111771, 15.9737320, -66.9855881, 15.8716412, -80.6251068, 80.7074127
33: -96.4416656, 32.2932892, -96.4080353, 32.1747169, -124.7225952, 124.8186569
34: -83.6278534, 15.8767662, -83.6072006, 15.7379398, -92.4125061, 92.5448074
35: -63.4618759, 33.4871750, -63.4339218, 33.3691139, -96.8309937, 96.9210968
36: -64.7107468, 35.0584412, -64.6922073, 34.9196396, -99.6303864, 99.7506485
37: -100.9399033, 22.0985870, -100.8859634, 21.9541245, -122.8940277, 122.9845505
38: -86.1424332, 33.5216904, -86.0953674, 33.3695564, -119.5119934, 119.6170578
39: -104.1096954, 26.7573299, -104.0524368, 26.6576538, -130.7673492, 130.8097687
40: -91.4764099, 3.2617064, -91.4398727, 3.1929979, -90.2335815, 90.2766495
41: -67.6165314, 22.3147087, -67.5930481, 22.2512932, -87.4263153, 87.4764557
42: -60.6201439, 15.0893135, -60.5922623, 14.9922905, -73.7712936, 73.9154358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.0773161, upper bound: 49.2010725
time: 78.62 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1991298, upper bound: 49.1991301
time: 76.31 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -67.3638458, 43.7989197, -67.4812164, 43.9778366, -111.3416824, 111.2801361
1: -38.0275726, 35.0538559, -38.0874290, 35.2076454, -73.2352142, 73.1412811
2: -29.4147797, 37.6708984, -29.4531174, 37.8049164, -67.2196960, 67.1240158
3: -43.6362000, 37.2547913, -43.6660538, 37.4700127, -81.1062164, 80.9208450
4: -44.5417404, 39.3818321, -44.5890121, 39.5921860, -84.1339264, 83.9708405
5: -40.8040848, 41.7100143, -40.8322563, 41.8751907, -82.6792755, 82.5422668
6: -72.1969757, 13.1233864, -72.3062363, 13.2067719, -83.1418152, 83.1817551
7: -53.2382927, 31.8946629, -53.3111496, 32.0149078, -85.2532043, 85.2058105
8: -57.7651100, 39.1593323, -57.8190918, 39.3414536, -97.1065674, 96.9784241
9: -41.7317657, 42.5054932, -41.8432159, 42.6939316, -84.4256973, 84.3487091
10: -58.4420395, 48.7703819, -58.5797615, 48.8716278, -107.3136673, 107.3501434
11: -48.6502342, 27.6977558, -49.0317268, 27.8061733, -76.4564056, 76.7294846
12: -66.1708832, 41.5337524, -66.3311081, 41.6244392, -107.4651642, 107.5379105
13: -60.3895988, 49.9198265, -60.5020370, 50.1408195, -110.5304184, 110.4218597
14: -85.9133835, 36.1060181, -86.1800537, 36.1738853, -122.0872650, 122.2860718
15: -41.4290428, 44.8202667, -41.5168381, 45.0484695, -86.4775085, 86.3371048
16: -61.2654495, 39.3205833, -61.4622040, 39.4024658, -100.6679153, 100.7827911
17: -80.1883774, 32.9406586, -80.5766296, 33.0690880, -113.2574615, 113.5172882
18: -45.7151566, 45.6542282, -45.8847275, 45.7151642, -91.4303207, 91.5389557
19: -35.2919617, 30.0251617, -35.5586014, 30.1089821, -65.4009399, 65.5837631
20: -40.5164146, 26.7343979, -40.7051392, 26.7741127, -67.2905273, 67.4395370
21: -45.1861687, 33.8599472, -45.4927711, 33.9402657, -79.1264343, 79.3527222
22: -36.3203621, 39.3938484, -36.5186844, 39.4219360, -75.7422943, 75.9125366
23: -33.9684563, 34.7421112, -34.2579193, 34.8256569, -68.7941132, 69.0000305
24: -38.7875938, 35.2070770, -39.0898209, 35.2637405, -74.0513306, 74.2968979
25: -36.3686867, 42.5562363, -36.6378555, 42.6306915, -78.9993744, 79.1940918
26: -51.7355881, 54.8137970, -51.9488487, 54.8697739, -106.6053619, 106.7626495
27: -42.9780350, 31.3948441, -43.1571846, 31.4267979, -74.4048309, 74.5520325
28: -34.8698578, 38.0397377, -35.1129150, 38.0931473, -72.9630051, 73.1526489
29: -33.6597252, 32.2431297, -33.9728661, 32.2943993, -65.9541245, 66.2159958
30: -49.3690910, 30.2992897, -49.7169762, 30.3743534, -79.7434464, 80.0162659
31: -46.8779678, 37.1038208, -47.1962357, 37.2074242, -84.0853882, 84.3000565
32: -66.8879547, 15.8542328, -67.0485611, 16.0261707, -80.6568146, 80.6473923
33: -96.3545532, 32.1469765, -96.5185089, 32.3270950, -124.7819824, 124.7610245
34: -83.5681152, 15.7175636, -83.6642532, 15.8852005, -92.4757156, 92.4185715
35: -63.4015465, 33.3464622, -63.4965134, 33.4639091, -96.8654556, 96.8429718
36: -64.6225586, 34.9046173, -64.7544556, 35.0518608, -99.6744232, 99.6590729
37: -100.7541504, 21.9347897, -100.9439392, 22.0044174, -122.7585678, 122.8787308
38: -86.0060043, 33.3418007, -86.1201019, 33.5331192, -119.5391235, 119.4618988
39: -103.9675903, 26.6420383, -104.1636353, 26.8283539, -130.7959442, 130.8056793
40: -91.3671646, 3.1736250, -91.5075912, 3.3456116, -90.2729340, 90.2426529
41: -67.5452118, 22.2313042, -67.6454010, 22.3355560, -87.4353333, 87.4324036
42: -60.5424652, 14.9766426, -60.6208687, 15.0478611, -73.7379608, 73.7689514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.0748348, upper bound: 49.1321367
time: 73.18 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1371220, upper bound: 49.1295481
time: 119.99 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -67.7056274, 43.9768219, -67.4936905, 44.0140953, -111.7197266, 111.4705124
1: -38.2925339, 35.2061424, -38.0908775, 35.2378998, -73.5304337, 73.2970200
2: -29.6065636, 37.8067856, -29.4570370, 37.8310394, -67.4376068, 67.2638245
3: -43.9030838, 37.4652367, -43.6678238, 37.5090485, -81.4121323, 81.1330566
4: -44.8008041, 39.5618019, -44.5921478, 39.6279373, -84.4287415, 84.1539459
5: -41.0328140, 41.8762245, -40.8343506, 41.9062119, -82.9390259, 82.7105713
6: -72.3024597, 13.2374477, -72.3229599, 13.2119064, -83.2538834, 83.3317871
7: -53.4890671, 32.0374985, -53.3139954, 32.0428543, -85.5319214, 85.3514938
8: -58.0590630, 39.3493080, -57.8224220, 39.3774796, -97.4365387, 97.1717300
9: -41.9980621, 42.6400681, -41.8494301, 42.7195854, -84.7176514, 84.4895020
10: -58.6401443, 48.8719482, -58.5904770, 48.8805199, -107.5206604, 107.4624252
11: -48.8223381, 27.8734245, -49.0553970, 27.8097916, -76.6321259, 76.9288177
12: -66.3440170, 41.7517204, -66.3612518, 41.6279907, -107.6407776, 107.8053284
13: -60.5530968, 50.0271873, -60.5078163, 50.1526566, -110.7057495, 110.5350037
14: -86.1501389, 36.2015533, -86.2044220, 36.1763687, -122.3265076, 122.4059753
15: -41.6541557, 44.9699326, -41.5206909, 45.0752678, -86.7294235, 86.4906235
16: -61.5467720, 39.4015694, -61.4723625, 39.4140778, -100.9608459, 100.8739319
17: -80.4106522, 33.1601067, -80.6000214, 33.0760002, -113.4866486, 113.7601318
18: -45.8597603, 45.8697205, -45.9039650, 45.7198715, -91.5796356, 91.7736816
19: -35.4555664, 30.2221756, -35.5846786, 30.1110191, -65.5665894, 65.8068542
20: -40.6760330, 26.9342957, -40.7328186, 26.7772903, -67.4533234, 67.6671143
21: -45.3590164, 34.0161209, -45.5170670, 33.9423294, -79.3013458, 79.5331879
22: -36.4818420, 39.5801773, -36.5449066, 39.4245491, -75.9063873, 76.1250839
23: -34.1364594, 34.9478073, -34.2877655, 34.8280563, -68.9645157, 69.2355728
24: -38.9868584, 35.3977013, -39.1255531, 35.2659492, -74.2528076, 74.5232544
25: -36.5794601, 42.8226089, -36.6732864, 42.6334763, -79.2129364, 79.4958954
26: -51.9052963, 55.0729904, -51.9739532, 54.8745003, -106.7798004, 107.0469437
27: -43.1031342, 31.4878788, -43.1786842, 31.4296360, -74.5327682, 74.6665649
28: -35.0310822, 38.2592888, -35.1419067, 38.0961533, -73.1272354, 73.4011993
29: -33.8461723, 32.3900490, -34.0032120, 32.2965050, -66.1426773, 66.3932648
30: -49.5490875, 30.4795780, -49.7478065, 30.3773041, -79.9263916, 80.2273865
31: -47.0994720, 37.3504906, -47.2341766, 37.2105331, -84.3100052, 84.5846710
32: -67.0111771, 15.9737320, -67.0702820, 16.0298672, -80.7832184, 80.7937012
33: -96.4416656, 32.2932892, -96.5290833, 32.3329391, -124.8795471, 124.9343338
34: -83.6278534, 15.8767662, -83.6718369, 15.8895569, -92.5496445, 92.6005249
35: -63.4618759, 33.4871750, -63.5033112, 33.4684753, -96.9303513, 96.9904861
36: -64.7107468, 35.0584412, -64.7694244, 35.0549698, -99.7657166, 99.8278656
37: -100.9399033, 22.0985870, -100.9732971, 22.0083618, -122.9482651, 123.0718842
38: -86.1424332, 33.5216904, -86.1395721, 33.5389824, -119.6814117, 119.6612625
39: -104.1096954, 26.7573299, -104.1810303, 26.8314590, -130.9411621, 130.9383545
40: -91.4764099, 3.2617064, -91.5220184, 3.3497057, -90.3857422, 90.3567276
41: -67.6165314, 22.3147087, -67.6550064, 22.3398285, -87.5126495, 87.5368195
42: -60.6201439, 15.0893135, -60.6314316, 15.0511341, -73.8204498, 73.9534912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1591

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.0748348, upper bound: 49.1321367
time: 74.20 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1964280, upper bound: 49.1295481
time: 65.09 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -67.4480896, 43.9553070, -67.3846436, 43.8146667, -111.2627563, 111.3399506
1: -38.0707054, 35.1903267, -38.0385208, 35.0693817, -73.1400909, 73.2288513
2: -29.4414387, 37.7877579, -29.4214039, 37.6848831, -67.1263199, 67.2091599
3: -43.5932846, 37.4487915, -43.6505737, 37.2709885, -80.8642731, 81.0993652
4: -44.5721512, 39.5373535, -44.5657806, 39.4011841, -83.9733353, 84.1031342
5: -40.7946014, 41.8557434, -40.8077126, 41.7272034, -82.5218048, 82.6634521
6: -72.2112579, 13.1908665, -72.2259216, 13.1315155, -83.1324310, 83.1624680
7: -53.2667007, 32.0011673, -53.2467804, 31.9078445, -85.1745453, 85.2479477
8: -57.8076210, 39.3041306, -57.7723923, 39.1841125, -96.9917297, 97.0765228
9: -41.7933426, 42.6799431, -41.8006134, 42.5148811, -84.3082275, 84.4805603
10: -58.5021172, 48.8501205, -58.4625092, 48.8072510, -107.3093719, 107.3126297
11: -48.9590492, 27.7949448, -48.6755295, 27.7836628, -76.7427139, 76.4704742
12: -66.2758636, 41.6004181, -66.1947784, 41.5486069, -107.5068359, 107.4620438
13: -60.3676071, 50.1158295, -60.4643250, 49.9398575, -110.3074646, 110.5801544
14: -86.1411896, 36.1077576, -85.9434662, 36.1575127, -122.2987061, 122.0512238
15: -41.4985886, 44.9792633, -41.4860611, 44.8368378, -86.3354263, 86.4653244
16: -61.3793602, 39.3820648, -61.3027878, 39.3371201, -100.7164764, 100.6848526
17: -80.5386581, 32.9992752, -80.2203522, 33.0324249, -113.5710831, 113.2196274
18: -45.8550911, 45.5761719, -45.7376862, 45.6804771, -91.5355682, 91.3138580
19: -35.5340309, 30.0685349, -35.3124084, 30.1009750, -65.6350098, 65.3809433
20: -40.6834259, 26.7196541, -40.5332336, 26.7559967, -67.4394226, 67.2528839
21: -45.4588737, 33.9057426, -45.2152061, 33.9342461, -79.3931198, 79.1209488
22: -36.4906654, 39.3184013, -36.3431702, 39.4046097, -75.8952789, 75.6615753
23: -34.2386131, 34.7933121, -33.9854202, 34.8068161, -69.0454254, 68.7787323
24: -39.0618744, 35.1819458, -38.8072433, 35.2520142, -74.3138885, 73.9891891
25: -36.6153259, 42.5500488, -36.3898697, 42.6140251, -79.2293549, 78.9399185
26: -51.9178085, 54.7322159, -51.7633591, 54.8307724, -106.7485809, 106.4955750
27: -43.1287575, 31.3263588, -42.9987679, 31.4087467, -74.5375061, 74.3251266
28: -35.0943222, 38.0343742, -34.8860016, 38.0728149, -73.1671371, 72.9203796
29: -33.9428024, 32.2431793, -33.6853561, 32.2816162, -66.2244186, 65.9285355
30: -49.6881943, 30.3589382, -49.3928680, 30.3614502, -80.0496445, 79.7518082
31: -47.1631660, 37.1430321, -46.9035988, 37.1931419, -84.3563080, 84.0466309
32: -66.9483032, 16.0109406, -66.9541931, 15.8719749, -80.5637054, 80.7079239
33: -96.4410019, 32.3082504, -96.4348297, 32.1560593, -124.6926270, 124.8363419
34: -83.5724640, 15.8668880, -83.6135712, 15.7345333, -92.3530731, 92.4913635
35: -63.4060669, 33.4483261, -63.4402847, 33.3539352, -96.7600021, 96.8886108
36: -64.6834030, 35.0400734, -64.6905060, 34.9172745, -99.6006775, 99.7305756
37: -100.8860626, 21.9847145, -100.7890320, 21.9418869, -122.8279495, 122.7737427
38: -86.0625000, 33.5139885, -86.0502777, 33.3634644, -119.4259644, 119.5642700
39: -104.0536957, 26.8115597, -104.0598907, 26.6494980, -130.7031860, 130.8714447
40: -91.4442596, 3.3301048, -91.4264526, 3.1826639, -90.1957550, 90.3172073
41: -67.5435486, 22.3179398, -67.5835953, 22.2434464, -87.3452911, 87.4554214
42: -60.5171967, 15.0286503, -60.5645447, 14.9878063, -73.7357178, 73.7329941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1639

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1321367, upper bound: 49.0586594
time: 83.14 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1321367, upper bound: 49.0586595
time: 80.96 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -67.5627518, 44.0801620, -67.3755188, 43.8123093, -111.3750610, 111.4556808
1: -38.1075821, 35.2704086, -38.0329247, 35.0676880, -73.1752701, 73.3033295
2: -29.5194969, 37.8785286, -29.4191360, 37.6833420, -67.2028351, 67.2976685
3: -43.7082977, 37.6845589, -43.6455917, 37.2672272, -80.9755249, 81.3301544
4: -44.7833023, 39.6415749, -44.5633469, 39.3930969, -84.1763992, 84.2049255
5: -40.8577805, 42.0288925, -40.8018417, 41.7251816, -82.5829620, 82.8307343
6: -72.3735962, 13.5152130, -72.2181778, 13.1300678, -83.2828598, 83.4848404
7: -53.3593063, 32.1532440, -53.2388916, 31.9061623, -85.2654724, 85.3921356
8: -57.9275970, 39.3975525, -57.7712097, 39.1777153, -97.1053162, 97.1687622
9: -41.8859482, 42.8674469, -41.7975578, 42.5123253, -84.3982697, 84.6650085
10: -58.6475220, 49.1468010, -58.4582214, 48.8047371, -107.4522552, 107.6050262
11: -49.1583939, 28.0216675, -48.6644402, 27.7821350, -76.9405289, 76.6861115
12: -66.3713379, 41.8251915, -66.1872025, 41.5466003, -107.6208801, 107.6706390
13: -60.5410728, 50.5787086, -60.4560318, 49.9362755, -110.4773483, 111.0347443
14: -86.4901962, 36.2165527, -85.9390945, 36.1483994, -122.6385956, 122.1556473
15: -41.7461014, 45.1233025, -41.4840317, 44.8309937, -86.5770950, 86.6073303
16: -61.5502167, 39.6796722, -61.2954865, 39.3348846, -100.8851013, 100.9751587
17: -80.7707367, 33.1467361, -80.2139130, 33.0236092, -113.7943420, 113.3606491
18: -46.3778572, 45.7702789, -45.7328339, 45.6757126, -92.0535736, 91.5031128
19: -35.7311821, 30.1282997, -35.3093872, 30.0981998, -65.8293839, 65.4376831
20: -40.9036942, 26.7902679, -40.5313339, 26.7518425, -67.6555328, 67.3216019
21: -45.6604843, 33.9698143, -45.2112541, 33.9315109, -79.5919952, 79.1810684
22: -36.8766632, 39.4733887, -36.3394241, 39.3987465, -76.2754059, 75.8128128
23: -34.4707413, 34.8558197, -33.9826126, 34.8041229, -69.2748642, 68.8384323
24: -39.3875885, 35.2764130, -38.8018570, 35.2473145, -74.6349030, 74.0782700
25: -36.8899231, 42.6741409, -36.3874588, 42.6085510, -79.4984741, 79.0615997
26: -52.4022484, 54.9070816, -51.7598648, 54.8214951, -107.2237396, 106.6669464
27: -43.5342979, 31.4444294, -42.9948311, 31.4023895, -74.9366913, 74.4392624
28: -35.3623047, 38.1193771, -34.8844223, 38.0682220, -73.4305267, 73.0037994
29: -34.1717491, 32.3234406, -33.6798058, 32.2781334, -66.4498825, 66.0032501
30: -49.9242935, 30.4538994, -49.3895416, 30.3594227, -80.2837143, 79.8434448
31: -47.4333916, 37.2319679, -46.8987503, 37.1893234, -84.6227112, 84.1307220
32: -67.1309280, 16.3186340, -66.9463959, 15.8704796, -80.7467804, 81.0129852
33: -96.5889435, 32.5682983, -96.4293671, 32.1535034, -124.8446503, 125.0946960
34: -83.7152328, 16.1015320, -83.6072540, 15.7318230, -92.5255127, 92.7268066
35: -63.5446053, 33.6742020, -63.4338989, 33.3518524, -96.8964539, 97.1081009
36: -64.8074875, 35.1895561, -64.6843719, 34.9156799, -99.7231674, 99.8739319
37: -101.0268555, 22.1720657, -100.7779922, 21.9397202, -122.9665756, 122.9500580
38: -86.1870880, 33.6571846, -86.0439911, 33.3605270, -119.5476151, 119.7011719
39: -104.2526855, 27.1249580, -104.0509491, 26.6465836, -130.8992615, 131.1759033
40: -91.6034698, 3.5803576, -91.4134903, 3.1794605, -90.3576965, 90.5612411
41: -67.7185822, 22.6223316, -67.5752563, 22.2420311, -87.5215530, 87.7562943
42: -60.6871262, 15.3980637, -60.5600090, 14.9853096, -73.9035187, 74.0989227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1639

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1295481, upper bound: 49.1204229
time: 113.25 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1295481, upper bound: 49.1204230
time: 63.77 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -67.4605865, 43.9915466, -67.7264709, 43.9924088, -111.4529953, 111.7180176
1: -38.0741806, 35.2205429, -38.3034897, 35.2216415, -73.2958221, 73.5240326
2: -29.4453564, 37.8139229, -29.6132469, 37.8207779, -67.2661362, 67.4271698
3: -43.5950623, 37.4878845, -43.9174995, 37.4814186, -81.0764771, 81.4053802
4: -44.5753136, 39.5730705, -44.8248711, 39.5811424, -84.1564560, 84.3979416
5: -40.7967529, 41.8867188, -41.0364151, 41.8933945, -82.6901474, 82.9231339
6: -72.2279129, 13.1960316, -72.3313980, 13.2455997, -83.2824249, 83.2745132
7: -53.2694931, 32.0291290, -53.4975586, 32.0507164, -85.3202057, 85.5266876
8: -57.8109589, 39.3401260, -58.0663147, 39.3741302, -97.1850891, 97.4064407
9: -41.7995720, 42.7056198, -42.0668869, 42.6494141, -84.4489899, 84.7725067
10: -58.5128059, 48.8590202, -58.6606941, 48.9088669, -107.4216766, 107.5197144
11: -48.9827919, 27.7985458, -48.8475533, 27.9593658, -76.9421539, 76.6461029
12: -66.3060608, 41.6039429, -66.3679276, 41.7666130, -107.7741699, 107.6377258
13: -60.3733940, 50.1277428, -60.6278229, 50.0473289, -110.4207230, 110.7555695
14: -86.1656570, 36.1102219, -86.1802139, 36.2530746, -122.4187317, 122.2904358
15: -41.5024567, 45.0060730, -41.7111206, 44.9865036, -86.4889603, 86.7171936
16: -61.3895302, 39.3936501, -61.5841827, 39.4181061, -100.8076324, 100.9778290
17: -80.5620728, 33.0062141, -80.4425659, 33.2519531, -113.8140259, 113.4487762
18: -45.8742828, 45.5808716, -45.8822441, 45.8960190, -91.7703018, 91.4631195
19: -35.5601196, 30.0705814, -35.4759827, 30.2980042, -65.8581238, 65.5465622
20: -40.7111244, 26.7228107, -40.6928253, 26.9559288, -67.6670532, 67.4156342
21: -45.4831619, 33.9077950, -45.3880119, 34.0904121, -79.5735779, 79.2958069
22: -36.5168953, 39.3209648, -36.5046730, 39.5909996, -76.1078949, 75.8256378
23: -34.2684250, 34.7956734, -34.1534805, 35.0124893, -69.2809143, 68.9491577
24: -39.0975876, 35.1841583, -39.0065193, 35.4426498, -74.5402374, 74.1906738
25: -36.6507111, 42.5528641, -36.6006012, 42.8804092, -79.5311203, 79.1534653
26: -51.9429169, 54.7370071, -51.9330521, 55.0899124, -107.0328293, 106.6700592
27: -43.1502266, 31.3292236, -43.1238899, 31.5018291, -74.6520538, 74.4531097
28: -35.1232872, 38.0373726, -35.0471535, 38.2923813, -73.4156647, 73.0845261
29: -33.9731216, 32.2453232, -33.8717957, 32.4285126, -66.4016342, 66.1171188
30: -49.7190437, 30.3619022, -49.5728607, 30.5417328, -80.2607727, 79.9347610
31: -47.2010765, 37.1461639, -47.1250839, 37.4398193, -84.6408997, 84.2712479
32: -66.9699554, 16.0146389, -67.0773773, 15.9915161, -80.7099609, 80.8342972
33: -96.4515839, 32.3141518, -96.5219421, 32.3023643, -124.8657990, 124.9339142
34: -83.5800934, 15.8712959, -83.6732559, 15.8937683, -92.5349884, 92.5653534
35: -63.4128647, 33.4528923, -63.5006256, 33.4946136, -96.9074783, 96.9535217
36: -64.6982880, 35.0431213, -64.7787323, 35.0710678, -99.7693558, 99.8218536
37: -100.9154816, 21.9887047, -100.9746857, 22.1056633, -123.0211487, 122.9633942
38: -86.0819702, 33.5198135, -86.1867065, 33.5432663, -119.6252365, 119.7065201
39: -104.0710602, 26.8147202, -104.2018890, 26.7647114, -130.8357697, 131.0166016
40: -91.4587021, 3.3341732, -91.5356903, 3.2708282, -90.3098145, 90.4299927
41: -67.5531921, 22.3222427, -67.6548920, 22.3268089, -87.4496765, 87.5326996
42: -60.5277481, 15.0319157, -60.6422195, 15.1004772, -73.9202881, 73.8154526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1639

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1321367, upper bound: 49.1181412
time: 351.11 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1321367, upper bound: 49.1181413
time: 68.07 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -67.5752563, 44.1164169, -67.7173767, 43.9901619, -111.5654144, 111.8337936
1: -38.1110229, 35.3006363, -38.2979584, 35.2199478, -73.3309708, 73.5985947
2: -29.5234184, 37.9046783, -29.6109295, 37.8192062, -67.3426208, 67.5156097
3: -43.7100449, 37.7236061, -43.9124985, 37.4776802, -81.1877289, 81.6361084
4: -44.7864113, 39.6773148, -44.8224068, 39.5730209, -84.3594360, 84.4997253
5: -40.8599319, 42.0599060, -41.0305634, 41.8913651, -82.7512970, 83.0904694
6: -72.3902969, 13.5203743, -72.3236694, 13.2441463, -83.4328766, 83.5969467
7: -53.3620682, 32.1811943, -53.4896851, 32.0490417, -85.4111099, 85.6708832
8: -57.9310074, 39.4335480, -58.0651016, 39.3677521, -97.2987595, 97.4986496
9: -41.8921814, 42.8931351, -42.0638809, 42.6468658, -84.5390472, 84.9570160
10: -58.6582146, 49.1557465, -58.6563950, 48.9063492, -107.5645599, 107.8121414
11: -49.1821442, 28.0252762, -48.8365059, 27.9578190, -77.1399612, 76.8617859
12: -66.4014130, 41.8287506, -66.3603287, 41.7645416, -107.8883057, 107.8464432
13: -60.5469055, 50.5905838, -60.6195450, 50.0436554, -110.5905609, 111.2101288
14: -86.5146179, 36.2190018, -86.1758118, 36.2440414, -122.7586594, 122.3948135
15: -41.7499199, 45.1500854, -41.7091446, 44.9806595, -86.7305756, 86.8592300
16: -61.5604210, 39.6912384, -61.5769234, 39.4158401, -100.9762573, 101.2681580
17: -80.7941360, 33.1536255, -80.4361572, 33.2430725, -114.0372086, 113.5897827
18: -46.3970757, 45.7750473, -45.8773766, 45.8912392, -92.2883148, 91.6524200
19: -35.7572060, 30.1303387, -35.4729500, 30.2952213, -66.0524292, 65.6032867
20: -40.9313850, 26.7934284, -40.6909294, 26.9518051, -67.8831940, 67.4843597
21: -45.6848450, 33.9718361, -45.3840675, 34.0876884, -79.7725372, 79.3559036
22: -36.9029160, 39.4759941, -36.5009232, 39.5851135, -76.4880295, 75.9769135
23: -34.5005417, 34.8582153, -34.1506577, 35.0098267, -69.5103683, 69.0088730
24: -39.4233360, 35.2786064, -39.0011101, 35.4379387, -74.8612747, 74.2797165
25: -36.9252892, 42.6769791, -36.5981522, 42.8749313, -79.8002167, 79.2751312
26: -52.4274406, 54.9119034, -51.9295654, 55.0806808, -107.5081177, 106.8414688
27: -43.5558052, 31.4472809, -43.1200066, 31.4954548, -75.0512619, 74.5672913
28: -35.3912773, 38.1223869, -35.0455589, 38.2877731, -73.6790466, 73.1679459
29: -34.2020950, 32.3255424, -33.8662491, 32.4250488, -66.6271439, 66.1917877
30: -49.9551964, 30.4568615, -49.5695534, 30.5397167, -80.4949112, 80.0264130
31: -47.4713135, 37.2350769, -47.1202583, 37.4360428, -84.9073563, 84.3553314
32: -67.1526642, 16.3223381, -67.0695801, 15.9899693, -80.8930435, 81.1393433
33: -96.5995026, 32.5741386, -96.5164795, 32.2997971, -125.0178223, 125.1923065
34: -83.7228241, 16.1058388, -83.6669159, 15.8911266, -92.7075500, 92.8007278
35: -63.5513458, 33.6787720, -63.4942398, 33.4925232, -97.0438690, 97.1730118
36: -64.8224030, 35.1926003, -64.7725830, 35.0695114, -99.8919144, 99.9651794
37: -101.0562134, 22.1760674, -100.9636688, 22.1035595, -123.1597748, 123.1397400
38: -86.2065353, 33.6631432, -86.1804352, 33.5403938, -119.7469330, 119.8435822
39: -104.2700500, 27.1280594, -104.1929626, 26.7618847, -131.0319366, 131.3210144
40: -91.6179428, 3.5844717, -91.5226974, 3.2676249, -90.4718018, 90.6740417
41: -67.7282715, 22.6266365, -67.6465759, 22.3254395, -87.6259460, 87.8335876
42: -60.6976891, 15.4013042, -60.6376953, 15.0979710, -74.0880890, 74.1813889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1639

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1295481, upper bound: 49.1798736
time: 75.07 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1295481, upper bound: 49.1798738
time: 91.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 169.39 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.0773161, upper bound: 49.2010725
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.1400608, upper bound: 49.1991301
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.0773161, upper bound: 49.2010725
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.1991298, upper bound: 49.1991301
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.0748348, upper bound: 49.1321367
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.1371220, upper bound: 49.1295481
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.0748348, upper bound: 49.1321367
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.1964280, upper bound: 49.1295481
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.1321367, upper bound: 49.0586594
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.1321367, upper bound: 49.0586595
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.1295481, upper bound: 49.1204229
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.1295481, upper bound: 49.1204230
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.1321367, upper bound: 49.1181412
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.1321367, upper bound: 49.1181413
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.1295481, upper bound: 49.1798736
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 169.39
Output dim: 29, lower bound: -49.1295481, upper bound: 49.1798738

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -67.3637543, 43.7988663, -67.3762054, 43.9048767, -111.2686310, 111.1750717
1: -38.0275154, 35.0538292, -38.0242653, 35.1434631, -73.1709747, 73.0780945
2: -29.4147491, 37.6708450, -29.4183369, 37.7457504, -67.1604996, 67.0891800
3: -43.6359749, 37.2547226, -43.5713081, 37.3715210, -81.0074921, 80.8260345
4: -44.5417175, 39.3816605, -44.5377884, 39.4524918, -83.9942093, 83.9194489
5: -40.8039627, 41.7099571, -40.7759323, 41.7997246, -82.6036835, 82.4858856
6: -72.1966858, 13.1233559, -72.1660614, 13.1262951, -83.0666046, 83.0396347
7: -53.2381325, 31.8946018, -53.2059174, 31.9797039, -85.2178345, 85.1005173
8: -57.7651138, 39.1591949, -57.7677689, 39.2496414, -97.0147552, 96.9269638
9: -41.7316093, 42.5054359, -41.7046394, 42.5823250, -84.3139343, 84.2100754
10: -58.4418297, 48.7703247, -58.4032593, 48.7837181, -107.2255478, 107.1735840
11: -48.6499825, 27.6977081, -48.6608086, 27.7000237, -76.3500061, 76.3585205
12: -66.1707001, 41.5336761, -66.2215195, 41.5243874, -107.3654785, 107.4322815
13: -60.3892097, 49.9197502, -60.2775230, 49.9525604, -110.3417664, 110.1972733
14: -85.9132538, 36.1058273, -85.9615021, 36.0506897, -121.9639435, 122.0673294
15: -41.4289780, 44.8200378, -41.4259186, 44.8459587, -86.2749329, 86.2459564
16: -61.2652130, 39.3205185, -61.2205658, 39.3444710, -100.6096802, 100.5410843
17: -80.1882629, 32.9404526, -80.2324219, 32.8977432, -113.0860062, 113.1728745
18: -45.7150764, 45.6538048, -45.7548180, 45.5327225, -91.2478027, 91.4086227
19: -35.2918701, 30.0250454, -35.3604012, 29.9927750, -65.2846451, 65.3854446
20: -40.5163422, 26.7342415, -40.5928192, 26.6914597, -67.2078018, 67.3270569
21: -45.1860504, 33.8598518, -45.2385292, 33.8331985, -79.0192490, 79.0983810
22: -36.3202820, 39.3935242, -36.3859406, 39.3004227, -75.6207047, 75.7794647
23: -33.9683762, 34.7420273, -34.0547752, 34.7193146, -68.6876907, 68.7967987
24: -38.7874908, 35.2068253, -38.8867073, 35.1335449, -73.9210358, 74.0935364
25: -36.3686142, 42.5559998, -36.4713478, 42.4868164, -78.8554306, 79.0273438
26: -51.7355118, 54.8133888, -51.7956429, 54.6943054, -106.4298172, 106.6090317
27: -42.9779358, 31.3945541, -43.0301437, 31.3048477, -74.2827835, 74.4246979
28: -34.8698120, 38.0395584, -34.9543304, 37.9929695, -72.8627777, 72.9938889
29: -33.6596336, 32.2429886, -33.7371864, 32.2006073, -65.8602448, 65.9801788
30: -49.3690033, 30.2992496, -49.4489365, 30.2956619, -79.6646652, 79.7481842
31: -46.8778839, 37.1036301, -46.9798470, 37.0507889, -83.9286728, 84.0834808
32: -66.8876724, 15.8541870, -66.8635712, 15.8527031, -80.4827118, 80.4594040
33: -96.3543320, 32.1469307, -96.3199463, 32.1500397, -124.6029358, 124.5707397
34: -83.5678558, 15.7174644, -83.5077515, 15.7152653, -92.3165665, 92.2801895
35: -63.4012604, 33.3464279, -63.3367081, 33.3489075, -96.7501678, 96.6831360
36: -64.6223373, 34.9045830, -64.6062241, 34.9046783, -99.5270157, 99.5108032
37: -100.7539978, 21.9347267, -100.7986755, 21.9304333, -122.6844330, 122.7333984
38: -86.0058212, 33.3417587, -86.0183182, 33.3445053, -119.3503265, 119.3600769
39: -103.9672546, 26.6419601, -103.9250946, 26.6378269, -130.6050873, 130.5670471
40: -91.3669662, 3.1735487, -91.3621063, 3.1733932, -90.1038971, 90.1000137
41: -67.5449066, 22.2312469, -67.4815369, 22.2293568, -87.3305054, 87.2696457
42: -60.5421524, 14.9765778, -60.4780006, 14.9698601, -73.6661987, 73.6288223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=365, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.0773161, upper bound: 49.1363506
time: 72.12 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.0773161, upper bound: 49.1991301
time: 76.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -67.3546524, 43.7965889, -67.4907990, 44.0297546, -111.3844070, 111.2873840
1: -38.0219727, 35.0521164, -38.0611343, 35.2234879, -73.2454605, 73.1132507
2: -29.4124336, 37.6693192, -29.4962978, 37.8364410, -67.2488708, 67.1656189
3: -43.6309891, 37.2509613, -43.6862793, 37.6071701, -81.2381592, 80.9372406
4: -44.5392532, 39.3734970, -44.7488708, 39.5564919, -84.0957489, 84.1223679
5: -40.7980804, 41.7079544, -40.8391571, 41.9727402, -82.7708206, 82.5471115
6: -72.1889343, 13.1218910, -72.3283463, 13.4505749, -83.3888016, 83.1898270
7: -53.2303276, 31.8928947, -53.2985229, 32.1318130, -85.3621368, 85.1914215
8: -57.7639008, 39.1528015, -57.8877296, 39.3430328, -97.1069336, 97.0405273
9: -41.7285843, 42.5029030, -41.7972488, 42.7698135, -84.4983978, 84.3001556
10: -58.4375725, 48.7677994, -58.5486755, 49.0803452, -107.5179138, 107.3164749
11: -48.6389198, 27.6961784, -48.8601875, 27.9268684, -76.5657883, 76.5563660
12: -66.1631088, 41.5316391, -66.3169327, 41.7490921, -107.5740051, 107.5464020
13: -60.3809662, 49.9161377, -60.4510689, 50.4155083, -110.7964783, 110.3672028
14: -85.9088440, 36.0967293, -86.3101959, 36.1594505, -122.0682983, 122.4069214
15: -41.4270287, 44.8142090, -41.6735878, 44.9900284, -86.4170532, 86.4877930
16: -61.2579041, 39.3182564, -61.3914604, 39.6418953, -100.8997955, 100.7097168
17: -80.1818848, 32.9316063, -80.4643860, 33.0451775, -113.2270660, 113.3959961
18: -45.7102509, 45.6489944, -46.2775612, 45.7268753, -91.4371262, 91.9265594
19: -35.2888451, 30.0222359, -35.5575523, 30.0525360, -65.3413849, 65.5797882
20: -40.5144653, 26.7300987, -40.8130798, 26.7620659, -67.2765350, 67.5431824
21: -45.1821098, 33.8570862, -45.4401054, 33.8972740, -79.0793839, 79.2971954
22: -36.3165283, 39.3876457, -36.7719040, 39.4553375, -75.7718658, 76.1595459
23: -33.9655838, 34.7393379, -34.2868195, 34.7819290, -68.7475128, 69.0261536
24: -38.7821159, 35.2020950, -39.2122459, 35.2280502, -74.0101624, 74.4143372
25: -36.3661880, 42.5505409, -36.7458687, 42.6109505, -78.9771423, 79.2964096
26: -51.7319260, 54.8041153, -52.2799606, 54.8691330, -106.6010590, 107.0840759
27: -42.9740715, 31.3881607, -43.4355812, 31.4229527, -74.3970261, 74.8237457
28: -34.8682022, 38.0349197, -35.2222366, 38.0779495, -72.9461517, 73.2571564
29: -33.6541061, 32.2395363, -33.9659119, 32.2808838, -65.9349899, 66.2054443
30: -49.3656807, 30.2971973, -49.6850090, 30.3905354, -79.7562180, 79.9822083
31: -46.8730392, 37.0998077, -47.2502899, 37.1397400, -84.0127792, 84.3500977
32: -66.8798370, 15.8526669, -67.0462494, 16.1603546, -80.7877350, 80.6425323
33: -96.3489075, 32.1443405, -96.4680481, 32.4100990, -124.8612671, 124.7229156
34: -83.5615463, 15.7147980, -83.6506500, 15.9500065, -92.5520477, 92.4527588
35: -63.3948860, 33.3443260, -63.4754143, 33.5748444, -96.9697266, 96.8197403
36: -64.6162109, 34.9030266, -64.7303467, 35.0542145, -99.6704254, 99.6333771
37: -100.7429276, 21.9326134, -100.9393616, 22.1177959, -122.8607254, 122.8719788
38: -85.9995728, 33.3389053, -86.1428223, 33.4878082, -119.4873810, 119.4817276
39: -103.9583511, 26.6391640, -104.1240768, 26.9512024, -130.9095459, 130.7632446
40: -91.3539886, 3.1703577, -91.5213699, 3.4236183, -90.3478699, 90.2619629
41: -67.5365601, 22.2298737, -67.6565704, 22.5336990, -87.6313934, 87.4457932
42: -60.5376053, 14.9740734, -60.6479263, 15.3392105, -74.0321426, 73.7963409

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=365, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1358819, upper bound: 49.1598762
time: 114.19 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1358819, upper bound: 49.1949540
time: 115.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -67.7055206, 43.9767685, -67.3887024, 43.9411545, -111.6466751, 111.3654709
1: -38.2924957, 35.2060852, -38.0277100, 35.1736794, -73.4661713, 73.2337952
2: -29.6065331, 37.8067398, -29.4222412, 37.7718735, -67.3784027, 67.2289810
3: -43.9028778, 37.4651871, -43.5730438, 37.4106064, -81.3134842, 81.0382309
4: -44.8007812, 39.5615997, -44.5408592, 39.4882050, -84.2889862, 84.1024628
5: -41.0326920, 41.8761673, -40.7781143, 41.8306732, -82.8633652, 82.6542816
6: -72.3021698, 13.2374039, -72.1828156, 13.1314545, -83.1786575, 83.1896667
7: -53.4889297, 32.0374680, -53.2086678, 32.0075989, -85.4965286, 85.2461395
8: -58.0590439, 39.3492050, -57.7711105, 39.2856560, -97.3446960, 97.1203156
9: -41.9978943, 42.6400299, -41.7108421, 42.6079788, -84.6058731, 84.3508759
10: -58.6398964, 48.8718948, -58.4139633, 48.7926674, -107.4325638, 107.2858582
11: -48.8220863, 27.8734035, -48.6845245, 27.7036018, -76.5256882, 76.5579300
12: -66.3438721, 41.7516403, -66.2516251, 41.5279121, -107.5411072, 107.6997604
13: -60.5526772, 50.0271263, -60.2833099, 49.9643974, -110.5170746, 110.3104401
14: -86.1499710, 36.2013245, -85.9859085, 36.0531158, -122.2030869, 122.1872330
15: -41.6540871, 44.9697227, -41.4297562, 44.8727608, -86.5268478, 86.3994751
16: -61.5465202, 39.4014969, -61.2307281, 39.3560486, -100.9025726, 100.6322250
17: -80.4105682, 33.1599159, -80.2558441, 32.9047012, -113.3152695, 113.4157562
18: -45.8596611, 45.8692780, -45.7739792, 45.5374489, -91.3971100, 91.6432571
19: -35.4555054, 30.2220554, -35.3864822, 29.9948158, -65.4503174, 65.6085358
20: -40.6759567, 26.9341393, -40.6205101, 26.6946220, -67.3705750, 67.5546494
21: -45.3589172, 34.0160103, -45.2628365, 33.8352585, -79.1941757, 79.2788467
22: -36.4817390, 39.5798759, -36.4121933, 39.3029633, -75.7846985, 75.9920654
23: -34.1363983, 34.9477005, -34.0845947, 34.7216949, -68.8580933, 69.0322952
24: -38.9867706, 35.3974609, -38.9223862, 35.1357651, -74.1225357, 74.3198471
25: -36.5793762, 42.8223953, -36.5067368, 42.4896240, -79.0690002, 79.3291321
26: -51.9052162, 55.0725555, -51.8207245, 54.6990547, -106.6042709, 106.8932800
27: -43.1030388, 31.4875832, -43.0516510, 31.3076801, -74.4107208, 74.5392303
28: -35.0310287, 38.2591209, -34.9833221, 37.9959412, -73.0269699, 73.2424469
29: -33.8460884, 32.3898849, -33.7674522, 32.2027130, -66.0487976, 66.1573334
30: -49.5490150, 30.4795303, -49.4798126, 30.2986069, -79.8476257, 79.9593430
31: -47.0993614, 37.3502960, -47.0177650, 37.0539322, -84.1532898, 84.3680573
32: -67.0108795, 15.9736919, -66.8852997, 15.8563957, -80.6090775, 80.6057358
33: -96.4414062, 32.2932663, -96.3304901, 32.1559219, -124.7005615, 124.7439804
34: -83.6275787, 15.8766785, -83.5154114, 15.7197132, -92.3904724, 92.4621506
35: -63.4615746, 33.4870834, -63.3434601, 33.3535156, -96.8150940, 96.8305435
36: -64.7105560, 35.0584259, -64.6211090, 34.9077606, -99.6183167, 99.6795349
37: -100.9397202, 22.0985298, -100.8280563, 21.9344597, -122.8741760, 122.9265900
38: -86.1422577, 33.5216446, -86.0378265, 33.3504143, -119.4926758, 119.5594711
39: -104.1093521, 26.7572937, -103.9424667, 26.6410236, -130.7503815, 130.6997681
40: -91.4761963, 3.2616682, -91.3765411, 3.1774616, -90.2166748, 90.2140961
41: -67.6162109, 22.3146496, -67.4911804, 22.2336578, -87.4078217, 87.3739853
42: -60.6198120, 15.0892372, -60.4885712, 14.9730949, -73.7486954, 73.8134079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=365, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1363505, upper bound: 49.1363506
time: 60.28 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1363505, upper bound: 49.1991301
time: 112.54 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -67.6964111, 43.9744110, -67.5032959, 44.0660553, -111.7624664, 111.4777069
1: -38.2869034, 35.2044296, -38.0646057, 35.2537155, -73.5406189, 73.2690353
2: -29.6042595, 37.8052025, -29.5002613, 37.8625565, -67.4668121, 67.3054657
3: -43.8978500, 37.4614677, -43.6880531, 37.6462975, -81.5441437, 81.1495209
4: -44.7983170, 39.5534973, -44.7519951, 39.5921898, -84.3905029, 84.3054962
5: -41.0267830, 41.8741684, -40.8412666, 42.0037689, -83.0305481, 82.7154388
6: -72.2944489, 13.2359447, -72.3450317, 13.4557037, -83.5009003, 83.3398209
7: -53.4810638, 32.0357628, -53.3012772, 32.1597214, -85.6407852, 85.3370361
8: -58.0578384, 39.3427811, -57.8911247, 39.3790588, -97.4368973, 97.2339020
9: -41.9948502, 42.6374664, -41.8035469, 42.7954826, -84.7903290, 84.4410095
10: -58.6356773, 48.8694153, -58.5593796, 49.0892754, -107.7249527, 107.4287949
11: -48.8110313, 27.8718510, -48.8839035, 27.9304733, -76.7415009, 76.7557526
12: -66.3362503, 41.7495956, -66.3470612, 41.7526512, -107.7497253, 107.8138046
13: -60.5444145, 50.0235443, -60.4568787, 50.4273376, -110.9717560, 110.4804230
14: -86.1456070, 36.1922913, -86.3346100, 36.1618919, -122.3074951, 122.5269012
15: -41.6520767, 44.9638710, -41.6773834, 45.0168419, -86.6689148, 86.6412506
16: -61.5391960, 39.3992310, -61.4016724, 39.6535416, -101.1927338, 100.8009033
17: -80.4041290, 33.1510925, -80.4877625, 33.0520973, -113.4562225, 113.6388550
18: -45.8548508, 45.8645134, -46.2967834, 45.7315826, -91.5864334, 92.1613007
19: -35.4524651, 30.2192688, -35.5836372, 30.0545845, -65.5070496, 65.8029022
20: -40.6740646, 26.9299889, -40.8407745, 26.7652588, -67.4393234, 67.7707672
21: -45.3549576, 34.0132484, -45.4644623, 33.8992920, -79.2542496, 79.4777069
22: -36.4780312, 39.5739746, -36.7981262, 39.4579239, -75.9359589, 76.3721008
23: -34.1335907, 34.9450302, -34.3166428, 34.7842789, -68.9178696, 69.2616730
24: -38.9813805, 35.3927689, -39.2479630, 35.2302628, -74.2116394, 74.6407318
25: -36.5769196, 42.8168869, -36.7812424, 42.6138039, -79.1907196, 79.5981293
26: -51.9016876, 55.0632439, -52.3051567, 54.8739471, -106.7756348, 107.3684006
27: -43.0991440, 31.4812317, -43.4570885, 31.4258003, -74.5249481, 74.9383240
28: -35.0294189, 38.2544975, -35.2512207, 38.0809746, -73.1103973, 73.5057220
29: -33.8405266, 32.3864098, -33.9962463, 32.2830048, -66.1235352, 66.3826599
30: -49.5456810, 30.4775238, -49.7158661, 30.3935070, -79.9391861, 80.1933899
31: -47.0945358, 37.3464775, -47.2881470, 37.1428375, -84.2373734, 84.6346283
32: -67.0030670, 15.9721909, -67.0679626, 16.1639748, -80.9141235, 80.7888107
33: -96.4360046, 32.2906914, -96.4786072, 32.4160118, -124.9588928, 124.8961639
34: -83.6212006, 15.8740520, -83.6582184, 15.9543381, -92.6260071, 92.6346512
35: -63.4551773, 33.4849968, -63.4821701, 33.5794525, -97.0346298, 96.9671631
36: -64.7044525, 35.0569000, -64.7453156, 35.0573120, -99.7617645, 99.8022156
37: -100.9286957, 22.0963840, -100.9687500, 22.1217575, -123.0504532, 123.0651321
38: -86.1360016, 33.5187225, -86.1622467, 33.4936447, -119.6296463, 119.6809692
39: -104.1004639, 26.7543907, -104.1414948, 26.9543037, -131.0547638, 130.8958893
40: -91.4632187, 3.2584887, -91.5358582, 3.4277315, -90.4606628, 90.3760071
41: -67.6078796, 22.3132515, -67.6662216, 22.5379963, -87.7087021, 87.5501938
42: -60.6152725, 15.0867462, -60.6584854, 15.3424530, -74.1146088, 73.9809036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=365, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=335, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1949538, upper bound: 49.1598762
time: 86.47 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1949538, upper bound: 49.1949540
time: 53.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -67.3637543, 43.7988663, -67.4480896, 43.9553070, -111.3190613, 111.2469559
1: -38.0275154, 35.0538292, -38.0707054, 35.1903267, -73.2178421, 73.1245346
2: -29.4147491, 37.6708450, -29.4414387, 37.7877579, -67.2025070, 67.1122818
3: -43.6359749, 37.2547226, -43.5932846, 37.4487915, -81.0847626, 80.8480072
4: -44.5417175, 39.3816605, -44.5721512, 39.5373535, -84.0790710, 83.9538116
5: -40.8039627, 41.7099571, -40.7946014, 41.8557434, -82.6597061, 82.5045624
6: -72.1966858, 13.1233559, -72.2112579, 13.1908665, -83.1245422, 83.0860977
7: -53.2381325, 31.8946018, -53.2667007, 32.0011673, -85.2393036, 85.1613007
8: -57.7651138, 39.1591949, -57.8076210, 39.3041306, -97.0692444, 96.9668121
9: -41.7316093, 42.5054359, -41.7933426, 42.6799431, -84.4115524, 84.2987823
10: -58.4418297, 48.7703247, -58.5021172, 48.8501205, -107.2919464, 107.2724457
11: -48.6499825, 27.6977081, -48.9590492, 27.7949448, -76.4449310, 76.6567535
12: -66.1707001, 41.5336761, -66.2758636, 41.6004181, -107.4394226, 107.4882812
13: -60.3892097, 49.9197502, -60.3676071, 50.1158295, -110.5050354, 110.2873535
14: -85.9132538, 36.1058273, -86.1411896, 36.1077576, -122.0210114, 122.2470169
15: -41.4289780, 44.8200378, -41.4985886, 44.9792633, -86.4082413, 86.3186264
16: -61.2652130, 39.3205185, -61.3793602, 39.3820648, -100.6472778, 100.6998749
17: -80.1882629, 32.9404526, -80.5386581, 32.9992752, -113.1875381, 113.4791107
18: -45.7150764, 45.6538048, -45.8550911, 45.5761719, -91.2912445, 91.5088959
19: -35.2918701, 30.0250454, -35.5340309, 30.0685349, -65.3604050, 65.5590744
20: -40.5163422, 26.7342415, -40.6834259, 26.7196541, -67.2359924, 67.4176636
21: -45.1860504, 33.8598518, -45.4588737, 33.9057426, -79.0917969, 79.3187256
22: -36.3202820, 39.3935242, -36.4906654, 39.3184013, -75.6386871, 75.8841858
23: -33.9683762, 34.7420273, -34.2386131, 34.7933121, -68.7616882, 68.9806366
24: -38.7874908, 35.2068253, -39.0618744, 35.1819458, -73.9694366, 74.2686996
25: -36.3686142, 42.5559998, -36.6153259, 42.5500488, -78.9186630, 79.1713257
26: -51.7355118, 54.8133888, -51.9178085, 54.7322159, -106.4677277, 106.7312012
27: -42.9779358, 31.3945541, -43.1287575, 31.3263588, -74.3042908, 74.5233154
28: -34.8698120, 38.0395584, -35.0943222, 38.0343742, -72.9041901, 73.1338806
29: -33.6596336, 32.2429886, -33.9428024, 32.2431793, -65.9028168, 66.1857910
30: -49.3690033, 30.2992496, -49.6881943, 30.3589382, -79.7279434, 79.9874420
31: -46.8778839, 37.1036301, -47.1631660, 37.1430321, -84.0209198, 84.2667999
32: -66.8876724, 15.8541870, -66.9483032, 16.0109406, -80.6408463, 80.5457077
33: -96.3543320, 32.1469307, -96.4410019, 32.3082504, -124.7599640, 124.6863861
34: -83.5678558, 15.7174644, -83.5724640, 15.8668880, -92.4537048, 92.3358994
35: -63.4012604, 33.3464279, -63.4060669, 33.4483261, -96.8495865, 96.7524948
36: -64.6223373, 34.9045830, -64.6834030, 35.0400734, -99.6624146, 99.5879822
37: -100.7539978, 21.9347267, -100.8860626, 21.9847145, -122.7387085, 122.8207855
38: -86.0058212, 33.3417587, -86.0625000, 33.5139885, -119.5198059, 119.4042587
39: -103.9672546, 26.6419601, -104.0536957, 26.8115597, -130.7788086, 130.6956482
40: -91.3669662, 3.1735487, -91.4442596, 3.3301048, -90.2560577, 90.1800919
41: -67.5449066, 22.2312469, -67.5435486, 22.3179398, -87.4168930, 87.3299561
42: -60.5421524, 14.9765778, -60.5171967, 15.0286503, -73.7153473, 73.6668701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=365, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1591

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.0748348, upper bound: 49.0673245
time: 88.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.0748348, upper bound: 49.1295481
time: 104.09 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -67.3546524, 43.7965889, -67.5627518, 44.0801620, -111.4348145, 111.3593445
1: -38.0219727, 35.0521164, -38.1075821, 35.2704086, -73.2923813, 73.1596985
2: -29.4124336, 37.6693192, -29.5194969, 37.8785286, -67.2909622, 67.1888123
3: -43.6309891, 37.2509613, -43.7082977, 37.6845589, -81.3155518, 80.9592590
4: -44.5392532, 39.3734970, -44.7833023, 39.6415749, -84.1808319, 84.1567993
5: -40.7980804, 41.7079544, -40.8577805, 42.0288925, -82.8269730, 82.5657349
6: -72.1889343, 13.1218910, -72.3735962, 13.5152130, -83.4469070, 83.2363358
7: -53.2303276, 31.8928947, -53.3593063, 32.1532440, -85.3835754, 85.2521973
8: -57.7639008, 39.1528015, -57.9275970, 39.3975525, -97.1614532, 97.0803986
9: -41.7285843, 42.5029030, -41.8859482, 42.8674469, -84.5960312, 84.3888550
10: -58.4375725, 48.7677994, -58.6475220, 49.1468010, -107.5843735, 107.4153214
11: -48.6389198, 27.6961784, -49.1583939, 28.0216675, -76.6605835, 76.8545685
12: -66.1631088, 41.5316391, -66.3713379, 41.8251915, -107.6480255, 107.6024780
13: -60.3809662, 49.9161377, -60.5410728, 50.5787086, -110.9596710, 110.4572144
14: -85.9088440, 36.0967293, -86.4901962, 36.2165527, -122.1253967, 122.5869293
15: -41.4270287, 44.8142090, -41.7461014, 45.1233025, -86.5503311, 86.5603104
16: -61.2579041, 39.3182564, -61.5502167, 39.6796722, -100.9375763, 100.8684692
17: -80.1818848, 32.9316063, -80.7707367, 33.1467361, -113.3286209, 113.7023468
18: -45.7102509, 45.6489944, -46.3778572, 45.7702789, -91.4805298, 92.0268555
19: -35.2888451, 30.0222359, -35.7311821, 30.1282997, -65.4171448, 65.7534180
20: -40.5144653, 26.7300987, -40.9036942, 26.7902679, -67.3047333, 67.6337891
21: -45.1821098, 33.8570862, -45.6604843, 33.9698143, -79.1519241, 79.5175705
22: -36.3165283, 39.3876457, -36.8766632, 39.4733887, -75.7899170, 76.2643127
23: -33.9655838, 34.7393379, -34.4707413, 34.8558197, -68.8214035, 69.2100830
24: -38.7821159, 35.2020950, -39.3875885, 35.2764130, -74.0585327, 74.5896835
25: -36.3661880, 42.5505409, -36.8899231, 42.6741409, -79.0403290, 79.4404602
26: -51.7319260, 54.8041153, -52.4022484, 54.9070816, -106.6390076, 107.2063599
27: -42.9740715, 31.3881607, -43.5342979, 31.4444294, -74.4185028, 74.9224548
28: -34.8682022, 38.0349197, -35.3623047, 38.1193771, -72.9875793, 73.3972244
29: -33.6541061, 32.2395363, -34.1717491, 32.3234406, -65.9775467, 66.4112854
30: -49.3656807, 30.2971973, -49.9242935, 30.4538994, -79.8195801, 80.2214890
31: -46.8730392, 37.0998077, -47.4333916, 37.2319679, -84.1050110, 84.5332031
32: -66.8798370, 15.8526669, -67.1309280, 16.3186340, -80.9459229, 80.7288132
33: -96.3489075, 32.1443405, -96.5889435, 32.5682983, -125.0182800, 124.8384552
34: -83.5615463, 15.7147980, -83.7152328, 16.1015320, -92.6891022, 92.5083542
35: -63.3948860, 33.3443260, -63.5446053, 33.6742020, -97.0690918, 96.8889313
36: -64.6162109, 34.9030266, -64.8074875, 35.1895561, -99.8057709, 99.7105103
37: -100.7429276, 21.9326134, -101.0268555, 22.1720657, -122.9149933, 122.9594727
38: -85.9995728, 33.3389053, -86.1870880, 33.6571846, -119.6567535, 119.5259933
39: -103.9583511, 26.6391640, -104.2526855, 27.1249580, -131.0833130, 130.8918457
40: -91.3539886, 3.1703577, -91.6034698, 3.5803576, -90.5001221, 90.3421021
41: -67.5365601, 22.2298737, -67.7185822, 22.6223316, -87.7178268, 87.5061493
42: -60.5376053, 14.9740734, -60.6871262, 15.3980637, -74.0812912, 73.8343353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=365, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.0685846, upper bound: 49.0902857
time: 73.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -49.1329505, upper bound: 49.1253783
time: 72.17 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 148.37 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 148.37
Output dim: 29, lower bound: -49.0773161, upper bound: 49.1363506
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 148.37
Output dim: 29, lower bound: -49.0773161, upper bound: 49.1991301
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 148.37
Output dim: 29, lower bound: -49.1358819, upper bound: 49.1598762
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 148.37
Output dim: 29, lower bound: -49.1358819, upper bound: 49.1949540
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 148.37
Output dim: 29, lower bound: -49.1363505, upper bound: 49.1363506
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 148.37
Output dim: 29, lower bound: -49.1363505, upper bound: 49.1991301
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 148.37
Output dim: 29, lower bound: -49.1949538, upper bound: 49.1598762
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 148.37
Output dim: 29, lower bound: -49.1949538, upper bound: 49.1949540
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 148.37
Output dim: 29, lower bound: -49.0748348, upper bound: 49.0673245
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 148.37
Output dim: 29, lower bound: -49.0748348, upper bound: 49.1295481
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 148.37
Output dim: 29, lower bound: -49.0685846, upper bound: 49.0902857
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 148.37
Output dim: 29, lower bound: -49.1329505, upper bound: 49.1253783
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 148.37
Output dim: 29, lower bound: -49.0748348, upper bound: 49.1321367
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 148.37
Output dim: 29, lower bound: -49.1964280, upper bound: 49.1295481
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 148.37
Output dim: 29, lower bound: -49.1321367, upper bound: 49.0586594
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 148.37
Output dim: 29, lower bound: -49.1321367, upper bound: 49.0586595
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 148.37
Output dim: 29, lower bound: -49.1295481, upper bound: 49.1204229
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 148.37
Output dim: 29, lower bound: -49.1295481, upper bound: 49.1204230
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 148.37
Output dim: 29, lower bound: -49.1321367, upper bound: 49.1181412
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 148.37
Output dim: 29, lower bound: -49.1321367, upper bound: 49.1181413
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 148.37
Output dim: 29, lower bound: -49.1295481, upper bound: 49.1798736
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 148.37
Output dim: 29, lower bound: -49.1295481, upper bound: 49.1798738
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=66.12645721435547
rel_dist={29: [-49.24526583204686, 49.24526583600755]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1658518, upper bound: 46.1251707
time: 71.11 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1558464, upper bound: 46.1558464
time: 62.97 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 134.20 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 134.20
Output dim: 29, lower bound: -46.1658518, upper bound: 46.1251707
IS_A2, status: Status.UNKNOWN, split count: 1, time: 134.20
Output dim: 29, lower bound: -46.1558464, upper bound: 46.1558464

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -67.4241028, 43.9691849, -67.4379272, 43.9797974, -110.8923950, 110.8958359
1: -38.0452576, 35.1950378, -38.0527496, 35.2052689, -72.8192673, 72.8170166
2: -29.4349117, 37.7924118, -29.4396362, 37.8016663, -66.9211807, 66.9174805
3: -43.6466942, 37.4368362, -43.6569519, 37.4476738, -81.0943680, 81.0937881
4: -44.5588341, 39.5477715, -44.5761147, 39.5604019, -84.1192322, 84.1238861
5: -40.8166656, 41.8543777, -40.8191566, 41.8661461, -82.6828156, 82.6735382
6: -72.2819138, 13.1482906, -72.3011017, 13.1536026, -80.8798676, 80.8957901
7: -53.2542725, 32.0250435, -53.2601166, 32.0348625, -85.2891388, 85.2851562
8: -57.7837524, 39.3276825, -57.7886925, 39.3446083, -97.1283569, 97.1163788
9: -41.7618484, 42.6260910, -41.8081284, 42.6325607, -84.2354965, 84.2766571
10: -58.4933395, 48.8165512, -58.5067253, 48.8420944, -107.3354340, 107.3232727
11: -48.7610054, 27.7154675, -48.7777939, 27.7689209, -76.5299225, 76.4932632
12: -66.3109055, 41.5532341, -66.3272552, 41.5628510, -106.4433746, 106.4486313
13: -60.4191322, 49.9962234, -60.4679565, 50.0097885, -110.4289246, 110.4641800
14: -86.0290222, 36.1202774, -86.0489578, 36.1527710, -122.1817932, 122.1692352
15: -41.4489746, 44.9460793, -41.4845200, 44.9569931, -86.4059677, 86.4306030
16: -61.3155556, 39.3793259, -61.3396645, 39.3909111, -100.7064667, 100.7189941
17: -80.2972641, 32.9762955, -80.3180695, 33.0346909, -113.3319550, 113.2943649
18: -45.8070526, 45.6773338, -45.8223915, 45.6948891, -91.5019379, 91.4997253
19: -35.4148064, 30.0358467, -35.4281502, 30.0834789, -65.4982834, 65.4639969
20: -40.6463013, 26.7496262, -40.6575317, 26.7650490, -67.4113464, 67.4071579
21: -45.3004723, 33.8702164, -45.3192635, 33.9171448, -79.2176208, 79.1894836
22: -36.4442444, 39.4071846, -36.4594383, 39.4145279, -75.3063354, 75.3119659
23: -34.1080399, 34.7547913, -34.1190186, 34.7946548, -68.9026947, 68.8738098
24: -38.9556770, 35.2179832, -38.9686508, 35.2478638, -74.2035370, 74.1866302
25: -36.5341148, 42.5710716, -36.5480003, 42.6087532, -79.1428680, 79.1190720
26: -51.8562584, 54.8376122, -51.8750725, 54.8485641, -106.7048187, 106.7126846
27: -43.0847702, 31.4085827, -43.0986862, 31.4183846, -74.5031586, 74.5072708
28: -35.0060806, 38.0555801, -35.0166283, 38.0771332, -73.0832138, 73.0722046
29: -33.8021812, 32.2545853, -33.8191452, 32.2793808, -65.9418335, 65.9334793
30: -49.5133018, 30.3148060, -49.5288925, 30.3558731, -79.8691711, 79.8436966
31: -47.0561867, 37.1188965, -47.0728836, 37.1745110, -84.2306976, 84.1917801
32: -66.9891434, 15.8724270, -67.0318298, 15.8840485, -79.2278595, 79.2590942
33: -96.4113007, 32.1759300, -96.4613190, 32.1825714, -122.5409851, 122.5817795
34: -83.6097717, 15.7388000, -83.6388626, 15.7497654, -89.1858521, 89.2007675
35: -63.4354172, 33.3702469, -63.4602661, 33.3752670, -95.8121796, 95.8352814
36: -64.6954803, 34.9204254, -64.7401886, 34.9285126, -99.3937531, 99.4300385
37: -100.8906860, 21.9551201, -100.9140167, 21.9600258, -122.0021973, 122.0200806
38: -86.0990906, 33.3707848, -86.1303711, 33.3844299, -119.4835205, 119.5011597
39: -104.0569763, 26.6585693, -104.1151733, 26.6636887, -130.5158386, 130.5696869
40: -91.4443054, 3.1938534, -91.4833527, 3.1997013, -88.3331146, 88.3674011
41: -67.5957184, 22.2521381, -67.6208801, 22.2599106, -86.3685150, 86.3860168
42: -60.5945625, 14.9930954, -60.6093216, 15.0004921, -71.5997314, 71.6000824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1637044, upper bound: 46.0864588
time: 57.84 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1637044, upper bound: 46.1230588
time: 80.23 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -67.4959259, 44.0195923, -67.4434891, 43.9835930, -110.9705505, 110.9538727
1: -38.0917435, 35.2418556, -38.0552940, 35.2096100, -72.8700638, 72.8708496
2: -29.4580460, 37.8343887, -29.4407196, 37.8054237, -66.9481201, 66.9638367
3: -43.6686363, 37.5141373, -43.6593170, 37.4517899, -81.1204224, 81.1734543
4: -44.5932007, 39.6324539, -44.5792427, 39.5662994, -84.1595001, 84.2117004
5: -40.8353195, 41.9103394, -40.8200760, 41.8697891, -82.7051086, 82.7304153
6: -72.3270569, 13.2128582, -72.3092041, 13.1559277, -80.9506989, 80.9614182
7: -53.3150978, 32.0465050, -53.2621231, 32.0359383, -85.3510361, 85.3086243
8: -57.8236465, 39.3821983, -57.7904739, 39.3503418, -97.1739883, 97.1726685
9: -41.8505096, 42.7237167, -41.8254166, 42.6346245, -84.3307648, 84.4022827
10: -58.5922127, 48.8829765, -58.5129929, 48.8493195, -107.4415283, 107.3959656
11: -49.0592079, 27.8105030, -48.7847595, 27.7980042, -76.8572083, 76.5952606
12: -66.3652802, 41.6292725, -66.3326874, 41.5675011, -106.5061340, 106.5275497
13: -60.5091972, 50.1594810, -60.4897232, 50.0146561, -110.5238495, 110.6492004
14: -86.2087250, 36.1774368, -86.0569687, 36.1703415, -122.3790665, 122.2344055
15: -41.5217171, 45.0794029, -41.5040321, 44.9618912, -86.4836121, 86.5834351
16: -61.4743958, 39.4169655, -61.3515205, 39.3939438, -100.8683395, 100.7684860
17: -80.6034775, 33.0777664, -80.3274384, 33.0646400, -113.6681213, 113.4052048
18: -45.9073524, 45.7207909, -45.8275299, 45.7019539, -91.6093063, 91.5483246
19: -35.5884514, 30.1116009, -35.4341164, 30.1101303, -65.6985779, 65.5457153
20: -40.7368851, 26.7777939, -40.6619186, 26.7683849, -67.5052719, 67.4397125
21: -45.5208397, 33.9427834, -45.3281898, 33.9418678, -79.4627075, 79.2709732
22: -36.5489273, 39.4252052, -36.4654922, 39.4175491, -75.4113312, 75.3407745
23: -34.2918701, 34.8287659, -34.1242523, 34.8181915, -69.1100616, 68.9530182
24: -39.1308250, 35.2664108, -38.9740028, 35.2604485, -74.3912735, 74.2404175
25: -36.6780777, 42.6343117, -36.5540314, 42.6263542, -79.3044281, 79.1883392
26: -51.9784889, 54.8755112, -51.8817711, 54.8542900, -106.8327789, 106.7572784
27: -43.1833572, 31.4301414, -43.1039696, 31.4212570, -74.6046143, 74.5341110
28: -35.1460381, 38.0970230, -35.0212708, 38.0874176, -73.2334595, 73.1182938
29: -34.0077400, 32.2971497, -33.8261032, 32.2913895, -66.1609955, 65.9823456
30: -49.7524719, 30.3781452, -49.5357170, 30.3724518, -80.1249237, 79.9138641
31: -47.2395172, 37.2110825, -47.0804329, 37.2064476, -84.4459686, 84.2915192
32: -67.0737762, 16.0306511, -67.0525208, 15.8891220, -79.3148880, 79.4378510
33: -96.5323563, 32.3340988, -96.4898224, 32.1836510, -122.6535645, 122.7649994
34: -83.6744080, 15.8904438, -83.6534119, 15.7551098, -89.2459869, 89.3462906
35: -63.5047493, 33.4696655, -63.4727249, 33.3770752, -95.8783264, 95.9532700
36: -64.7727356, 35.0557442, -64.7594299, 34.9326019, -99.4728546, 99.5846329
37: -100.9780731, 22.0094032, -100.9227982, 21.9615250, -122.0929718, 122.0847321
38: -86.1432571, 33.5402031, -86.1377411, 33.3916855, -119.5349426, 119.6779480
39: -104.1855621, 26.8324165, -104.1459656, 26.6653023, -130.6443481, 130.7795868
40: -91.5264664, 3.3505287, -91.4998779, 3.2025242, -88.4162292, 88.5346146
41: -67.6576767, 22.3406715, -67.6320038, 22.2637043, -86.4322739, 86.4824066
42: -60.6337433, 15.0519371, -60.6153870, 15.0035782, -71.6780701, 71.6428223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1537676, upper bound: 46.1171601
time: 182.95 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1537676, upper bound: 46.1537675
time: 94.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 280.29 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 280.29
Output dim: 29, lower bound: -46.1637044, upper bound: 46.0864588
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 280.29
Output dim: 29, lower bound: -46.1637044, upper bound: 46.1230588
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 280.29
Output dim: 29, lower bound: -46.1537676, upper bound: 46.1171601
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 280.29
Output dim: 29, lower bound: -46.1537676, upper bound: 46.1537675

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -67.3943939, 43.8846130, -67.3776703, 43.8095551, -110.6919098, 110.7506485
1: -38.0365753, 35.1257439, -38.0350151, 35.0640869, -72.6692581, 72.7299347
2: -29.4249649, 37.7327347, -29.4194698, 37.6801605, -66.7902222, 66.8382797
3: -43.6414719, 37.3474846, -43.6464767, 37.2655716, -80.9070435, 80.9939575
4: -44.5504074, 39.4662323, -44.5590782, 39.3944702, -83.9448776, 84.0253143
5: -40.8104210, 41.7832642, -40.8065948, 41.7217941, -82.5322113, 82.5898590
6: -72.2398224, 13.1359730, -72.2161484, 13.1287327, -80.8102951, 80.7917099
7: -53.2463531, 31.9610176, -53.2441101, 31.9044571, -85.1508102, 85.2051239
8: -57.7745628, 39.2451935, -57.7700653, 39.1762199, -96.9507828, 97.0152588
9: -41.7470131, 42.5664215, -41.7780914, 42.5119476, -84.1000595, 84.1867371
10: -58.4680977, 48.7936630, -58.4553871, 48.7959137, -107.2640076, 107.2490540
11: -48.7059708, 27.7067089, -48.6670456, 27.7512150, -76.4571838, 76.3737564
12: -66.2417908, 41.5435486, -66.1872253, 41.5433884, -106.3545532, 106.2991714
13: -60.4045715, 49.9584846, -60.4384499, 49.9333572, -110.3379288, 110.3969345
14: -85.9712296, 36.1132431, -85.9333115, 36.1384583, -122.1096878, 122.0465546
15: -41.4391251, 44.8834572, -41.4646225, 44.8311920, -86.2703171, 86.3480835
16: -61.2909126, 39.3501930, -61.2894707, 39.3321686, -100.6230774, 100.6396637
17: -80.2432556, 32.9587212, -80.2092285, 32.9990311, -113.2422867, 113.1679535
18: -45.7616501, 45.6659622, -45.7304459, 45.6717567, -91.4334106, 91.3964081
19: -35.3544235, 30.0305786, -35.3053246, 30.0728111, -65.4272308, 65.3359070
20: -40.5821457, 26.7421379, -40.5276184, 26.7498646, -67.3320084, 67.2697601
21: -45.2437897, 33.8651276, -45.2049179, 33.9068718, -79.1506653, 79.0700455
22: -36.3831978, 39.4006310, -36.3355331, 39.4011536, -75.2301178, 75.1784210
23: -34.0394058, 34.7485123, -33.9794044, 34.7819748, -68.8213806, 68.7279205
24: -38.8728218, 35.2126045, -38.8005867, 35.2369385, -74.1097565, 74.0131912
25: -36.4527664, 42.5637207, -36.3825912, 42.5939484, -79.0467148, 78.9463120
26: -51.7965660, 54.8258820, -51.7543449, 54.8248138, -106.6213837, 106.5802307
27: -43.0320473, 31.4018097, -42.9919281, 31.4046135, -74.4366608, 74.3937378
28: -34.9391174, 38.0477829, -34.8804092, 38.0612640, -73.0003815, 72.9281921
29: -33.7318459, 32.2489319, -33.6767120, 32.2679367, -65.8597946, 65.7854309
30: -49.4417191, 30.3071156, -49.3846397, 30.3403454, -79.7820663, 79.6917572
31: -46.9685860, 37.1114388, -46.8946915, 37.1594505, -84.1280365, 84.0061340
32: -66.9386444, 15.8634071, -66.9306488, 15.8658085, -79.1580658, 79.1473846
33: -96.3833160, 32.1616669, -96.4046555, 32.1535492, -122.4816132, 122.5087662
34: -83.5892639, 15.7283554, -83.5973206, 15.7284737, -89.1354523, 89.1347198
35: -63.4186592, 33.3585510, -63.4263992, 33.3515282, -95.7670288, 95.7837448
36: -64.6590805, 34.9126053, -64.6672363, 34.9127121, -99.3400879, 99.3473511
37: -100.8217087, 21.9450836, -100.7775192, 21.9396591, -121.9122925, 121.8730469
38: -86.0526428, 33.3565369, -86.0372925, 33.3554840, -119.4081268, 119.3938293
39: -104.0126343, 26.6503716, -104.0258789, 26.6471519, -130.4538574, 130.4714203
40: -91.4061890, 3.1838293, -91.4063110, 3.1794643, -88.2743530, 88.2807846
41: -67.5708771, 22.2418194, -67.5703735, 22.2391167, -86.3207855, 86.3227539
42: -60.5687332, 14.9849529, -60.5572433, 14.9840031, -71.5468750, 71.5191269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0284080
time: 645.79 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0703258
time: 52.87 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -67.4200897, 43.9592552, -67.7194901, 43.9873886, -110.8879089, 111.1677780
1: -38.0437622, 35.1879082, -38.2999802, 35.2163467, -72.8253098, 73.0584869
2: -29.4331284, 37.7863464, -29.6113033, 37.8160439, -66.9309998, 67.0892410
3: -43.6451836, 37.4277077, -43.9134026, 37.4760513, -81.1212311, 81.3411102
4: -44.5568848, 39.5395584, -44.8182068, 39.5744476, -84.1313324, 84.3577652
5: -40.8149490, 41.8468857, -41.0353088, 41.8880348, -82.7029877, 82.8821945
6: -72.2745361, 13.1465950, -72.3216019, 13.2427940, -80.9796219, 80.9078522
7: -53.2522163, 32.0184326, -53.4948654, 32.0473404, -85.2995605, 85.5132980
8: -57.7816238, 39.3192101, -58.0640373, 39.3662415, -97.1478653, 97.3832474
9: -41.7598839, 42.6186028, -42.0443878, 42.6465034, -84.2410889, 84.5069199
10: -58.4901428, 48.8121643, -58.6535225, 48.8975296, -107.3876724, 107.4656830
11: -48.7541161, 27.7141571, -48.8390961, 27.9268780, -76.6809921, 76.5532532
12: -66.3036118, 41.5508919, -66.3603058, 41.7613373, -106.6535645, 106.4786530
13: -60.4166527, 49.9838791, -60.6019249, 50.0407791, -110.4574280, 110.5858002
14: -86.0213699, 36.1183662, -86.1701279, 36.2341003, -122.2554703, 122.2884979
15: -41.4471397, 44.9386024, -41.6896896, 44.9808426, -86.4279785, 86.6282959
16: -61.3119316, 39.3741226, -61.5708771, 39.4131622, -100.7250977, 100.9449997
17: -80.2911072, 32.9730949, -80.4314880, 33.2185440, -113.5096512, 113.4045868
18: -45.8008881, 45.6756668, -45.8750496, 45.8872833, -91.6881714, 91.5507202
19: -35.4080009, 30.0347862, -35.4688797, 30.2698288, -65.6778259, 65.5036621
20: -40.6389618, 26.7486839, -40.6872025, 26.9497662, -67.5887299, 67.4358826
21: -45.2937317, 33.8694077, -45.3777657, 34.0630302, -79.3567657, 79.2471771
22: -36.4370232, 39.4059486, -36.4970436, 39.5875053, -75.4736938, 75.3394928
23: -34.1006012, 34.7534866, -34.1474266, 34.9876747, -69.0882721, 68.9009094
24: -38.9461098, 35.2171783, -38.9998512, 35.4275970, -74.3737030, 74.2170258
25: -36.5254059, 42.5696068, -36.5932770, 42.8602829, -79.3856888, 79.1628876
26: -51.8482018, 54.8357849, -51.9240608, 55.0839767, -106.9321747, 106.7598419
27: -43.0764389, 31.4076958, -43.1170578, 31.4977150, -74.5741577, 74.5247498
28: -34.9985809, 38.0539970, -35.0416260, 38.2808304, -73.2794113, 73.0956268
29: -33.7939606, 32.2533722, -33.8631401, 32.4148407, -66.0686951, 65.9722672
30: -49.5049591, 30.3132877, -49.5646324, 30.5206661, -80.0256271, 79.8779221
31: -47.0464973, 37.1178970, -47.1161728, 37.4061356, -84.4526367, 84.2340698
32: -66.9828110, 15.8710022, -67.0538788, 15.9853497, -79.3267975, 79.2778015
33: -96.4053650, 32.1737595, -96.4918060, 32.2999458, -122.6664734, 122.6101532
34: -83.6052551, 15.7372532, -83.6569061, 15.8877068, -89.3299026, 89.2090225
35: -63.4327545, 33.3681870, -63.4866867, 33.4922028, -95.9225006, 95.8635864
36: -64.6895599, 34.9190254, -64.7554550, 35.0665436, -99.5348816, 99.4406204
37: -100.8820877, 21.9533520, -100.9632263, 22.1035118, -122.1442719, 122.0648346
38: -86.0924149, 33.3685951, -86.1736984, 33.5353317, -119.6277466, 119.5422974
39: -104.0487518, 26.6569252, -104.1679764, 26.7624035, -130.6090088, 130.6191711
40: -91.4363098, 3.1923151, -91.5155029, 3.2676020, -88.4043732, 88.3964462
41: -67.5909348, 22.2506599, -67.6417160, 22.3225365, -86.4358521, 86.4043579
42: -60.5904350, 14.9916821, -60.6349258, 15.0966415, -71.7444000, 71.6011734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1591

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0649716
time: 74.23 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1474986, upper bound: 46.1068700
time: 61.16 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -67.4662247, 43.9350700, -67.3831940, 43.8133850, -110.7699280, 110.8087158
1: -38.0829811, 35.1725578, -38.0375633, 35.0684280, -72.7200699, 72.7836838
2: -29.4480915, 37.7747345, -29.4205799, 37.6839447, -66.8171234, 66.8846970
3: -43.6634254, 37.4247665, -43.6488190, 37.2696915, -80.9331207, 81.0735855
4: -44.5847626, 39.5509262, -44.5621452, 39.4003296, -83.9850922, 84.1130676
5: -40.8290863, 41.8392220, -40.8075180, 41.7254333, -82.5545197, 82.6467438
6: -72.2849960, 13.2005634, -72.2242889, 13.1310844, -80.8811111, 80.8573608
7: -53.3071442, 31.9824791, -53.2461128, 31.9055843, -85.2127304, 85.2285919
8: -57.8143997, 39.2996445, -57.7718239, 39.1819305, -96.9963303, 97.0714722
9: -41.8357544, 42.6640625, -41.7953644, 42.5140152, -84.1954498, 84.3123627
10: -58.5670090, 48.8601379, -58.4616470, 48.8031464, -107.3701553, 107.3217850
11: -49.0041656, 27.8017960, -48.6740150, 27.7802925, -76.7844543, 76.4758148
12: -66.2961731, 41.6195908, -66.1926117, 41.5480042, -106.4172668, 106.3779678
13: -60.4946671, 50.1217766, -60.4601555, 49.9382324, -110.4328995, 110.5819321
14: -86.1509018, 36.1703224, -85.9413834, 36.1560478, -122.3069458, 122.1117096
15: -41.5118561, 45.0167694, -41.4840775, 44.8360748, -86.3479309, 86.5008469
16: -61.4496727, 39.3878479, -61.3012886, 39.3352356, -100.7849121, 100.6891327
17: -80.5494537, 33.0602036, -80.2185669, 33.0289764, -113.5784302, 113.2787704
18: -45.8620110, 45.7093849, -45.7356224, 45.6787872, -91.5408020, 91.4450073
19: -35.5280457, 30.1063023, -35.3112907, 30.0994911, -65.6275330, 65.4175949
20: -40.6727066, 26.7703400, -40.5320091, 26.7531548, -67.4258575, 67.3023529
21: -45.4640846, 33.9377289, -45.2138519, 33.9315681, -79.3956528, 79.1515808
22: -36.4878654, 39.4186096, -36.3415871, 39.4041367, -75.3351212, 75.2072601
23: -34.2232132, 34.8224869, -33.9846344, 34.8054962, -69.0287094, 68.8071213
24: -39.0479660, 35.2610054, -38.8059807, 35.2495232, -74.2974854, 74.0669861
25: -36.5967064, 42.6269836, -36.3886261, 42.6115265, -79.2082367, 79.0156097
26: -51.9187393, 54.8638306, -51.7610550, 54.8304825, -106.7492218, 106.6248856
27: -43.1306458, 31.4233589, -42.9972229, 31.4074860, -74.5381317, 74.4205780
28: -35.0790405, 38.0892296, -34.8850746, 38.0715485, -73.1505890, 72.9743042
29: -33.9373589, 32.2915344, -33.6836662, 32.2799606, -66.0789108, 65.8343658
30: -49.6808815, 30.3704681, -49.3914833, 30.3569298, -80.0378113, 79.7619476
31: -47.1519089, 37.2036743, -46.9022560, 37.1913681, -84.3432770, 84.1059265
32: -67.0233231, 16.0216351, -66.9514160, 15.8709164, -79.2451019, 79.3261719
33: -96.5043869, 32.3198624, -96.4331665, 32.1546440, -122.5943298, 122.6919708
34: -83.6539307, 15.8799095, -83.6118546, 15.7338333, -89.1955719, 89.2802200
35: -63.4881287, 33.4579582, -63.4388695, 33.3533401, -95.8331909, 95.9017029
36: -64.7363129, 35.0479202, -64.6865082, 34.9167786, -99.4192810, 99.5019760
37: -100.9090500, 21.9993382, -100.7863159, 21.9411697, -122.0030823, 121.9375916
38: -86.0968323, 33.5259323, -86.0446854, 33.3627853, -119.4596176, 119.5706177
39: -104.1413040, 26.8241348, -104.0566635, 26.6487713, -130.5823975, 130.6813049
40: -91.4883957, 3.3405046, -91.4228210, 3.1822786, -88.3574982, 88.4480209
41: -67.6328659, 22.3303757, -67.5815048, 22.2429142, -86.3845673, 86.4192276
42: -60.6079216, 15.0437698, -60.5633392, 14.9871407, -71.6252365, 71.5618286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1591

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1375524, upper bound: 46.0590791
time: 83.94 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1375524, upper bound: 46.1009839
time: 63.93 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -67.4918976, 44.0096855, -67.7250214, 43.9911880, -110.9661407, 111.2259140
1: -38.0902023, 35.2347107, -38.3025513, 35.2207260, -72.8761520, 73.1122971
2: -29.4562550, 37.8283539, -29.6123905, 37.8198166, -66.9578781, 67.1356583
3: -43.6671524, 37.5049934, -43.9157219, 37.4801407, -81.1472931, 81.4207153
4: -44.5913162, 39.6242447, -44.8212662, 39.5802689, -84.1715851, 84.4455109
5: -40.8335762, 41.9028702, -41.0362549, 41.8916092, -82.7251892, 82.9391251
6: -72.3196335, 13.2111969, -72.3297729, 13.2451153, -81.0504303, 80.9735184
7: -53.3130569, 32.0398750, -53.4968719, 32.0483932, -85.3614502, 85.5367432
8: -57.8214493, 39.3736954, -58.0657501, 39.3719254, -97.1933746, 97.4394455
9: -41.8485603, 42.7162247, -42.0616379, 42.6485291, -84.3363419, 84.6325378
10: -58.5890121, 48.8785706, -58.6598396, 48.9048004, -107.4938126, 107.5384064
11: -49.0522881, 27.8091946, -48.8460693, 27.9560318, -77.0083160, 76.6552658
12: -66.3580017, 41.6269722, -66.3658066, 41.7659645, -106.7163391, 106.5575943
13: -60.5067482, 50.1471786, -60.6236649, 50.0456467, -110.5523987, 110.7708435
14: -86.2010727, 36.1755104, -86.1780319, 36.2516594, -122.4527283, 122.3535461
15: -41.5198631, 45.0718880, -41.7091255, 44.9857101, -86.5055695, 86.7810135
16: -61.4707642, 39.4117851, -61.5827179, 39.4161911, -100.8869553, 100.9945068
17: -80.5973816, 33.0746193, -80.4408722, 33.2485123, -113.8458939, 113.5154877
18: -45.9012070, 45.7190933, -45.8801804, 45.8943367, -91.7955475, 91.5992737
19: -35.5816574, 30.1105728, -35.4748497, 30.2964745, -65.8781281, 65.5854187
20: -40.7295380, 26.7768631, -40.6915970, 26.9531078, -67.6826477, 67.4684601
21: -45.5140724, 33.9419479, -45.3867073, 34.0877724, -79.6018448, 79.3286591
22: -36.5417023, 39.4239502, -36.5030708, 39.5904922, -75.5786896, 75.3683777
23: -34.2844162, 34.8274612, -34.1526794, 35.0111847, -69.2956009, 68.9801407
24: -39.1212997, 35.2655869, -39.0052528, 35.4401627, -74.5614624, 74.2708435
25: -36.6694107, 42.6328735, -36.5993042, 42.8778534, -79.5472641, 79.2321777
26: -51.9704247, 54.8737068, -51.9307785, 55.0895920, -107.0600128, 106.8044891
27: -43.1750755, 31.4292202, -43.1223412, 31.5005341, -74.6756134, 74.5515594
28: -35.1385651, 38.0954514, -35.0462227, 38.2911072, -73.4296722, 73.1416779
29: -33.9995575, 32.2959290, -33.8700943, 32.4268570, -66.2878494, 66.0211487
30: -49.7441101, 30.3766232, -49.5714569, 30.5372601, -80.2813721, 79.9480820
31: -47.2298431, 37.2100906, -47.1236725, 37.4380417, -84.6678848, 84.3337631
32: -67.0674591, 16.0292587, -67.0746155, 15.9904022, -79.4138336, 79.4565887
33: -96.5264435, 32.3320007, -96.5202942, 32.3010254, -122.7792053, 122.7933807
34: -83.6698914, 15.8889084, -83.6714935, 15.8930264, -89.3900223, 89.3545685
35: -63.5021210, 33.4675369, -63.4991913, 33.4940491, -95.9886780, 95.9815140
36: -64.7667542, 35.0543747, -64.7747345, 35.0706329, -99.6141205, 99.5952911
37: -100.9695358, 22.0076141, -100.9719925, 22.1050110, -122.2349548, 122.1293793
38: -86.1366043, 33.5380020, -86.1811371, 33.5426559, -119.6792603, 119.7191391
39: -104.1773453, 26.8306999, -104.1986771, 26.7640190, -130.7375488, 130.8291931
40: -91.5185013, 3.3490171, -91.5320282, 3.2704363, -88.4874878, 88.5636978
41: -67.6529160, 22.3391628, -67.6528244, 22.3262920, -86.4997025, 86.5008163
42: -60.6295929, 15.0505295, -60.6409988, 15.0997677, -71.8227234, 71.6439056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1591

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1375524, upper bound: 46.0956484
time: 98.35 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1375524, upper bound: 46.1375521
time: 86.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 187.44 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 187.44
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0284080
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 187.44
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0703258
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 187.44
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0649716
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 187.44
Output dim: 29, lower bound: -46.1474986, upper bound: 46.1068700
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 187.44
Output dim: 29, lower bound: -46.1375524, upper bound: 46.0590791
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 187.44
Output dim: 29, lower bound: -46.1375524, upper bound: 46.1009839
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 187.44
Output dim: 29, lower bound: -46.1375524, upper bound: 46.0956484
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 187.44
Output dim: 29, lower bound: -46.1375524, upper bound: 46.1375521

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -67.3612366, 43.8620605, -67.3659515, 43.8016396, -110.6503601, 110.7157288
1: -38.0198746, 35.1083832, -38.0290184, 35.0579605, -72.6426544, 72.7043610
2: -29.4132767, 37.7155762, -29.4153461, 37.6741638, -66.7656708, 66.8125916
3: -43.5687141, 37.3262596, -43.6207924, 37.2580986, -80.8268127, 80.9470520
4: -44.5335388, 39.4112434, -44.5531731, 39.3746338, -83.9081726, 83.9644165
5: -40.7728195, 41.7637177, -40.7932892, 41.7148819, -82.4877014, 82.5570068
6: -72.1448441, 13.1200676, -72.1829071, 13.1230946, -80.7086182, 80.7406387
7: -53.2019272, 31.9472809, -53.2283478, 31.8996239, -85.1015472, 85.1756287
8: -57.7630844, 39.2078094, -57.7660179, 39.1630402, -96.9261246, 96.9738312
9: -41.6971550, 42.5524940, -41.7604713, 42.5069885, -84.0451965, 84.1543427
10: -58.3904877, 48.7722626, -58.4277954, 48.7883949, -107.1788788, 107.2000580
11: -48.6333237, 27.6955910, -48.6402283, 27.7472878, -76.3806152, 76.3358154
12: -66.1865692, 41.5194778, -66.1677246, 41.5348358, -106.2959595, 106.2558746
13: -60.2701378, 49.9335327, -60.3914642, 49.9246140, -110.1947479, 110.3249969
14: -85.9323730, 36.0470924, -85.9195786, 36.1149330, -122.0473022, 121.9666748
15: -41.4209671, 44.8142548, -41.4582291, 44.8065720, -86.2275391, 86.2724838
16: -61.2080841, 39.3298187, -61.2601318, 39.3249969, -100.5330811, 100.5899506
17: -80.2052765, 32.8889084, -80.1954651, 32.9746475, -113.1799240, 113.0843735
18: -45.7320328, 45.5269318, -45.7200012, 45.6228218, -91.3548584, 91.2469330
19: -35.3298454, 29.9901314, -35.2966042, 30.0586624, -65.3885040, 65.2867355
20: -40.5604095, 26.6876640, -40.5199738, 26.7308044, -67.2912140, 67.2076416
21: -45.2098503, 33.8306198, -45.1928787, 33.8947792, -79.1046295, 79.0234985
22: -36.3551712, 39.2970543, -36.3256149, 39.3649826, -75.1643066, 75.0631638
23: -34.0200424, 34.7161636, -33.9725571, 34.7705803, -68.7906189, 68.6887207
24: -38.8448753, 35.1308289, -38.7907448, 35.2083588, -74.0532379, 73.9215698
25: -36.4301949, 42.4831123, -36.3746185, 42.5652695, -78.9954681, 78.8577271
26: -51.7655411, 54.6883698, -51.7433586, 54.7768707, -106.5424118, 106.4317322
27: -43.0036087, 31.3014183, -42.9818840, 31.3695698, -74.3731766, 74.2833023
28: -34.9204636, 37.9889946, -34.8738518, 38.0407219, -72.9611816, 72.8628464
29: -33.7017326, 32.1977768, -33.6659851, 32.2500076, -65.8117142, 65.7239456
30: -49.4128761, 30.2917652, -49.3743896, 30.3349895, -79.7478638, 79.6661530
31: -46.9355316, 37.0470619, -46.8830528, 37.1369781, -84.0725098, 83.9301147
32: -66.8383331, 15.8481617, -66.8948364, 15.8604126, -79.0508423, 79.0954895
33: -96.3057938, 32.1428032, -96.3775024, 32.1468506, -122.3993530, 122.4608307
34: -83.4974518, 15.7100334, -83.5650940, 15.7219219, -89.0450439, 89.0831833
35: -63.3282700, 33.3429871, -63.3947487, 33.3459702, -95.6701660, 95.7337036
36: -64.5880127, 34.9007530, -64.6423264, 34.9085007, -99.2646027, 99.3103256
37: -100.7638245, 21.9253368, -100.7572021, 21.9327698, -121.8508453, 121.8323517
38: -85.9950943, 33.3373413, -86.0171051, 33.3486290, -119.3437195, 119.3544464
39: -103.9027100, 26.6336823, -103.9874725, 26.6410713, -130.3358459, 130.4141846
40: -91.3428421, 3.1683712, -91.3841400, 3.1738205, -88.2058411, 88.2421036
41: -67.4689941, 22.2242126, -67.5347595, 22.2328873, -86.2119598, 86.2685623
42: -60.4650688, 14.9656849, -60.5202026, 14.9772606, -71.4369659, 71.4603882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1639

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0284080
time: 168.45 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0284080
time: 608.11 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -67.4758530, 43.9868927, -67.3626556, 43.8059311, -110.7738037, 110.8432388
1: -38.0567322, 35.1883926, -38.0259590, 35.0614014, -72.6807861, 72.7940369
2: -29.4912605, 37.8062668, -29.4157639, 37.6777687, -66.8320465, 66.9410248
3: -43.6837082, 37.5619926, -43.6398849, 37.2596245, -80.9433289, 81.2018738
4: -44.7446480, 39.5152473, -44.5552368, 39.3820114, -84.1266632, 84.0704803
5: -40.8360062, 41.9367523, -40.7972260, 41.7186737, -82.5546799, 82.7339783
6: -72.3070755, 13.4443340, -72.2041779, 13.1264362, -80.8606491, 81.0918198
7: -53.2944984, 32.0994072, -53.2312393, 31.9017811, -85.1962814, 85.3306427
8: -57.8830833, 39.3012161, -57.7680855, 39.1657753, -97.0488586, 97.0693054
9: -41.7897835, 42.7399750, -41.7732086, 42.5078659, -84.1347580, 84.3573685
10: -58.5359039, 49.0688477, -58.4482460, 48.7919083, -107.3278122, 107.5170898
11: -48.8326416, 27.9224644, -48.6490440, 27.7486954, -76.5813370, 76.5715103
12: -66.2820282, 41.7442513, -66.1751709, 41.5400658, -106.4071045, 106.4757080
13: -60.4437256, 50.3964729, -60.4242210, 49.9275818, -110.3713074, 110.8206940
14: -86.2810593, 36.1559067, -85.9263000, 36.1243210, -122.4053802, 122.0822067
15: -41.6685867, 44.9583473, -41.4613724, 44.8214874, -86.4900742, 86.4197235
16: -61.3789902, 39.6273041, -61.2772484, 39.3285217, -100.7075119, 100.9045563
17: -80.4371490, 33.0363007, -80.1991577, 32.9844856, -113.4216309, 113.2354584
18: -46.2548027, 45.7211266, -45.7228622, 45.6647911, -91.9195938, 91.4439850
19: -35.5270157, 30.0498772, -35.3004799, 30.0680637, -65.5950775, 65.3503571
20: -40.7806396, 26.7583160, -40.5246201, 26.7427750, -67.5234146, 67.2829361
21: -45.4114761, 33.8947067, -45.1986313, 33.9021759, -79.3136520, 79.0933380
22: -36.7410736, 39.4519768, -36.3295364, 39.3910637, -75.5876007, 75.2235413
23: -34.2520676, 34.7787476, -33.9748154, 34.7774658, -69.0295334, 68.7535629
24: -39.1704407, 35.2253456, -38.7918167, 35.2287903, -74.3992310, 74.0171661
25: -36.7046967, 42.6073265, -36.3786888, 42.5847321, -79.2894287, 78.9860153
26: -52.2498741, 54.8632011, -51.7487259, 54.8093033, -107.0591736, 106.6119232
27: -43.4090195, 31.4195328, -42.9857101, 31.3949966, -74.8040161, 74.4052429
28: -35.1883392, 38.0739975, -34.8778687, 38.0534248, -73.2417603, 72.9518661
29: -33.9304504, 32.2780266, -33.6678238, 32.2622147, -66.0574799, 65.8068771
30: -49.6489105, 30.3866425, -49.3794289, 30.3370132, -79.9859238, 79.7660675
31: -47.2060013, 37.1359558, -46.8867950, 37.1529236, -84.3589249, 84.0227509
32: -67.0210114, 16.1557846, -66.9185410, 15.8634834, -79.2322845, 79.4321594
33: -96.4539185, 32.4028702, -96.3964767, 32.1495438, -122.5437012, 122.7425079
34: -83.6403046, 15.9446754, -83.5867767, 15.7241821, -89.2018433, 89.3413391
35: -63.4669342, 33.5689087, -63.4155807, 33.3482666, -95.8079834, 95.9840546
36: -64.7121277, 35.0502930, -64.6569824, 34.9101639, -99.3907471, 99.4807434
37: -100.9044800, 22.1127491, -100.7594452, 21.9362831, -122.0207825, 122.0162659
38: -86.1195755, 33.4806290, -86.0270386, 33.3508873, -119.4704590, 119.5076675
39: -104.1017609, 26.9470501, -104.0109177, 26.6425915, -130.5362244, 130.7546082
40: -91.5021362, 3.4185734, -91.3856354, 3.1742764, -88.3694458, 88.4996796
41: -67.6440506, 22.5284653, -67.5574951, 22.2368793, -86.3879700, 86.6009369
42: -60.6349449, 15.3350906, -60.5495605, 14.9800930, -71.6013794, 71.8606110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1639

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0703258
time: 85.76 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0703258
time: 55.70 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -67.3869171, 43.9367065, -67.7077866, 43.9794540, -110.8464661, 111.1329422
1: -38.0270500, 35.1705017, -38.2939758, 35.2102203, -72.7987366, 73.0329590
2: -29.4214516, 37.7691879, -29.6071606, 37.8100319, -66.9064636, 67.0635605
3: -43.5724258, 37.4064789, -43.8876801, 37.4686203, -81.0410461, 81.2941589
4: -44.5400391, 39.4845352, -44.8122559, 39.5546150, -84.0946503, 84.2967911
5: -40.7772980, 41.8273621, -41.0219536, 41.8810883, -82.6583862, 82.8493195
6: -72.1794662, 13.1306725, -72.2884064, 13.2371578, -80.8779602, 80.8567963
7: -53.2077942, 32.0046463, -53.4791412, 32.0425034, -85.2502975, 85.4837875
8: -57.7701569, 39.2818527, -58.0599518, 39.3530693, -97.1232300, 97.3418045
9: -41.7099991, 42.6046143, -42.0267639, 42.6415291, -84.1862183, 84.4746094
10: -58.4125290, 48.7906876, -58.6259880, 48.8899651, -107.3024902, 107.4166718
11: -48.6814575, 27.7030468, -48.8122597, 27.9229584, -76.6044159, 76.5153046
12: -66.2483597, 41.5269051, -66.3408127, 41.7527924, -106.5950623, 106.4353867
13: -60.2822113, 49.9589462, -60.5549469, 50.0320435, -110.3142548, 110.5138931
14: -85.9824753, 36.0522919, -86.1563721, 36.2104797, -122.1929550, 122.2086639
15: -41.4289131, 44.8694229, -41.6832657, 44.9562302, -86.3851471, 86.5526886
16: -61.2291222, 39.3537140, -61.5415306, 39.4059792, -100.6351013, 100.8952484
17: -80.2531433, 32.9033127, -80.4177322, 33.1940994, -113.4472427, 113.3210449
18: -45.7712440, 45.5366783, -45.8645515, 45.8383713, -91.6096191, 91.4012299
19: -35.3834419, 29.9943504, -35.4601746, 30.2556801, -65.6391220, 65.4545288
20: -40.6172409, 26.6942425, -40.6795807, 26.9307575, -67.5479965, 67.3738251
21: -45.2598381, 33.8348694, -45.3657303, 34.0509415, -79.3107758, 79.2005997
22: -36.4089928, 39.3024216, -36.4870796, 39.5513458, -75.4078522, 75.2243042
23: -34.0812531, 34.7211151, -34.1405869, 34.9762573, -69.0575104, 68.8617020
24: -38.9181328, 35.1353874, -38.9899940, 35.3990288, -74.3171616, 74.1253815
25: -36.5028610, 42.4889755, -36.5852890, 42.8316345, -79.3344955, 79.0742645
26: -51.8171921, 54.6983032, -51.9131203, 55.0359993, -106.8531952, 106.6114197
27: -43.0480194, 31.3072929, -43.1070175, 31.4626446, -74.5106659, 74.4143066
28: -34.9799652, 37.9952583, -35.0350227, 38.2602844, -73.2402496, 73.0302811
29: -33.7638435, 32.2021599, -33.8523941, 32.3969116, -66.0206375, 65.9106674
30: -49.4760971, 30.2979164, -49.5543633, 30.5152988, -79.9913940, 79.8522797
31: -47.0134201, 37.0534935, -47.1045380, 37.3835793, -84.3970032, 84.1580353
32: -66.8824844, 15.8557510, -67.0180206, 15.9799271, -79.2195740, 79.2259064
33: -96.3278732, 32.1549530, -96.4646301, 32.2932091, -122.5842133, 122.5621567
34: -83.5134735, 15.7190208, -83.6247177, 15.8811340, -89.2394867, 89.1574554
35: -63.3423347, 33.3525810, -63.4550476, 33.4865875, -95.8255768, 95.8134995
36: -64.6185226, 34.9071350, -64.7305145, 35.0623131, -99.4594727, 99.4036331
37: -100.8242111, 21.9336739, -100.9428711, 22.0965862, -122.0826569, 122.0240784
38: -86.0349045, 33.3494339, -86.1535339, 33.5284500, -119.5633545, 119.5029678
39: -103.9388275, 26.6402569, -104.1294708, 26.7563057, -130.4910126, 130.5620117
40: -91.3730316, 3.1767845, -91.4933472, 3.2620115, -88.3358612, 88.3578568
41: -67.4890594, 22.2330074, -67.6060638, 22.3162804, -86.3270569, 86.3501816
42: -60.4867401, 14.9724464, -60.5979118, 15.0898590, -71.6344986, 71.5424500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1639

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0649716
time: 78.63 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0649715
time: 67.36 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -67.5015182, 44.0615768, -67.7044754, 43.9837723, -110.9698334, 111.2604446
1: -38.0639343, 35.2505417, -38.2909546, 35.2136993, -72.8369064, 73.1226654
2: -29.4994240, 37.8598557, -29.6075611, 37.8136253, -66.9728088, 67.1919403
3: -43.6873856, 37.6422081, -43.9067764, 37.4700737, -81.1574554, 81.5489807
4: -44.7511292, 39.5885010, -44.8143539, 39.5619507, -84.3130798, 84.4028549
5: -40.8405075, 42.0004272, -41.0258560, 41.8848419, -82.7253494, 83.0262833
6: -72.3416901, 13.4549351, -72.3097153, 13.2404900, -81.0299377, 81.2079391
7: -53.3003845, 32.1567612, -53.4820175, 32.0446205, -85.3450012, 85.6387787
8: -57.8902054, 39.3752594, -58.0620422, 39.3557701, -97.2459717, 97.4373016
9: -41.8026466, 42.7921257, -42.0395432, 42.6424561, -84.2757874, 84.6775360
10: -58.5579643, 49.0873375, -58.6464386, 48.8934784, -107.4514465, 107.7337799
11: -48.8808479, 27.9298935, -48.8210907, 27.9243584, -76.8052063, 76.7509842
12: -66.3438034, 41.7516098, -66.3482895, 41.7580528, -106.7061920, 106.6551971
13: -60.4558105, 50.4218330, -60.5876961, 50.0350494, -110.4908600, 111.0095291
14: -86.3312149, 36.1610985, -86.1630325, 36.2198830, -122.5511017, 122.3241272
15: -41.6765556, 45.0134773, -41.6864777, 44.9711609, -86.6477203, 86.6999512
16: -61.4000473, 39.6512527, -61.5586929, 39.4094925, -100.8095398, 101.2099457
17: -80.4850769, 33.0507050, -80.4213943, 33.2039528, -113.6890259, 113.4720993
18: -46.2940140, 45.7308311, -45.8674393, 45.8802872, -92.1743011, 91.5982666
19: -35.5805893, 30.0541115, -35.4640388, 30.2650776, -65.8456650, 65.5181503
20: -40.8374863, 26.7648411, -40.6842041, 26.9426956, -67.7801819, 67.4490433
21: -45.4614182, 33.8989258, -45.3714409, 34.0583534, -79.5197754, 79.2703705
22: -36.7949333, 39.4573898, -36.4910393, 39.5774384, -75.8311005, 75.3846512
23: -34.3133011, 34.7836990, -34.1428604, 34.9831390, -69.2964401, 68.9265594
24: -39.2436829, 35.2299118, -38.9911003, 35.4194756, -74.6631622, 74.2210083
25: -36.7773361, 42.6131363, -36.5893936, 42.8511658, -79.6285019, 79.2025299
26: -52.3015747, 54.8731346, -51.9184570, 55.0684280, -107.3700027, 106.7915955
27: -43.4534416, 31.4253960, -43.1108208, 31.4880638, -74.9415054, 74.5362167
28: -35.2478638, 38.0802917, -35.0390472, 38.2730026, -73.5208664, 73.1193390
29: -33.9926224, 32.2824326, -33.8542442, 32.4091187, -66.2665329, 65.9936600
30: -49.7121277, 30.3928280, -49.5593529, 30.5172997, -80.2294312, 79.9521790
31: -47.2838135, 37.1423798, -47.1083107, 37.3996048, -84.6834183, 84.2506866
32: -67.0652008, 16.1633530, -67.0417404, 15.9829369, -79.4009933, 79.5625916
33: -96.4759521, 32.4150124, -96.4835663, 32.2959518, -122.7286072, 122.8438721
34: -83.6563110, 15.9536781, -83.6463776, 15.8833809, -89.3962479, 89.4156418
35: -63.4809875, 33.5784988, -63.4758987, 33.4888840, -95.9635162, 96.0638885
36: -64.7426605, 35.0567017, -64.7451782, 35.0640030, -99.5855408, 99.5741043
37: -100.9649048, 22.1210098, -100.9451370, 22.1000805, -122.2526855, 122.2081299
38: -86.1593552, 33.4927025, -86.1634674, 33.5307541, -119.6901093, 119.6561737
39: -104.1378250, 26.9535675, -104.1529083, 26.7579689, -130.6913605, 130.9025726
40: -91.5322876, 3.4270401, -91.4949036, 3.2624865, -88.4994965, 88.6154556
41: -67.6641388, 22.5373344, -67.6287842, 22.3202534, -86.5030670, 86.6824875
42: -60.6566315, 15.3418198, -60.6272049, 15.0927849, -71.7988892, 71.9426651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1639

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1474986, upper bound: 46.1068700
time: 59.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1474986, upper bound: 46.1068700
time: 75.20 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -67.4330902, 43.9124603, -67.3714752, 43.8054657, -110.7285156, 110.7739029
1: -38.0662842, 35.1551971, -38.0315475, 35.0622787, -72.6934128, 72.7582397
2: -29.4364052, 37.7575874, -29.4164467, 37.6779175, -66.7925568, 66.8590164
3: -43.5906487, 37.4035339, -43.6231232, 37.2622681, -80.8529205, 81.0266571
4: -44.5679092, 39.4960861, -44.5562325, 39.3805389, -83.9484482, 84.0523224
5: -40.7914581, 41.8197556, -40.7941628, 41.7185059, -82.5099640, 82.6139221
6: -72.1900024, 13.1846657, -72.1910553, 13.1254425, -80.7796326, 80.8063049
7: -53.2626915, 31.9687881, -53.2303543, 31.9007626, -85.1634521, 85.1991425
8: -57.8029709, 39.2623215, -57.7677765, 39.1687775, -96.9717484, 97.0300980
9: -41.7859077, 42.6500931, -41.7777214, 42.5090714, -84.1406021, 84.2800140
10: -58.4893417, 48.8386230, -58.4340897, 48.7956085, -107.2849503, 107.2727127
11: -48.9314804, 27.7905674, -48.6471519, 27.7763767, -76.7078552, 76.4377213
12: -66.2409515, 41.5955429, -66.1731567, 41.5395317, -106.3586426, 106.3347321
13: -60.3602333, 50.0968742, -60.4132004, 49.9295044, -110.2897339, 110.5100708
14: -86.1120453, 36.1042290, -85.9276276, 36.1324997, -122.2445450, 122.0318604
15: -41.4935913, 44.9476166, -41.4776421, 44.8114777, -86.3050690, 86.4252625
16: -61.3668480, 39.3674240, -61.2719536, 39.3280830, -100.6949310, 100.6393738
17: -80.5114594, 32.9904289, -80.2048340, 33.0045853, -113.5160446, 113.1952667
18: -45.8323364, 45.5703888, -45.7250938, 45.6298370, -91.4621735, 91.2954865
19: -35.5034752, 30.0658531, -35.3025818, 30.0853386, -65.5888138, 65.3684387
20: -40.6509857, 26.7158775, -40.5244064, 26.7341442, -67.3851318, 67.2402802
21: -45.4301796, 33.9031792, -45.2018433, 33.9194717, -79.3496552, 79.1050262
22: -36.4598312, 39.3150482, -36.3316650, 39.3679543, -75.2693024, 75.0920334
23: -34.2038498, 34.7901115, -33.9777756, 34.7940979, -68.9979477, 68.7678833
24: -39.0200386, 35.1792297, -38.7961044, 35.2209663, -74.2410049, 73.9753342
25: -36.5741806, 42.5463486, -36.3806572, 42.5828629, -79.1570435, 78.9270020
26: -51.8877754, 54.7263184, -51.7501183, 54.7824478, -106.6702271, 106.4764404
27: -43.1021805, 31.3229351, -42.9872360, 31.3723755, -74.4745560, 74.3101730
28: -35.0604248, 38.0304184, -34.8784904, 38.0510330, -73.1114578, 72.9089050
29: -33.9072990, 32.2403488, -33.6729126, 32.2620354, -66.0308380, 65.7728271
30: -49.6520653, 30.3550987, -49.3812141, 30.3515797, -80.0036469, 79.7363129
31: -47.1188126, 37.1392746, -46.8905792, 37.1688576, -84.2876740, 84.0298538
32: -66.9229889, 16.0064030, -66.9156342, 15.8654900, -79.1378479, 79.2743225
33: -96.4268799, 32.3010712, -96.4060516, 32.1479988, -122.5119476, 122.6439819
34: -83.5621185, 15.8616467, -83.5796738, 15.7272701, -89.1052399, 89.2287216
35: -63.3976898, 33.4423828, -63.4072037, 33.3478165, -95.7363586, 95.8516312
36: -64.6651993, 35.0360794, -64.6616211, 34.9125710, -99.3438263, 99.4649353
37: -100.8512115, 21.9796715, -100.7659454, 21.9342308, -121.9416199, 121.8969574
38: -86.0392685, 33.5067139, -86.0244598, 33.3558998, -119.3951721, 119.5311737
39: -104.0313110, 26.8074150, -104.0181885, 26.6426716, -130.4644470, 130.6240692
40: -91.4250259, 3.3250351, -91.4006195, 3.1766558, -88.2889862, 88.4094162
41: -67.5310059, 22.3127155, -67.5458832, 22.2366543, -86.2757721, 86.3650665
42: -60.5042419, 15.0245438, -60.5263290, 14.9802876, -71.5154800, 71.5031128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1674

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1137446, upper bound: 46.0555783
time: 90.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1341164, upper bound: 46.0555783
time: 64.14 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -67.5477600, 44.0373383, -67.3681488, 43.8097839, -110.8521118, 110.9013824
1: -38.1031609, 35.2353516, -38.0285034, 35.0657578, -72.7315979, 72.8479919
2: -29.5144215, 37.8483810, -29.4168491, 37.6815414, -66.8589706, 66.9874039
3: -43.7057114, 37.6393242, -43.6421928, 37.2637177, -80.9694290, 81.2815170
4: -44.7790565, 39.6003189, -44.5583496, 39.3878784, -84.1669312, 84.1586685
5: -40.8546600, 41.9928436, -40.7980652, 41.7222824, -82.5769424, 82.7909088
6: -72.3523407, 13.5090218, -72.2123260, 13.1287956, -80.9318542, 81.1576080
7: -53.3553276, 32.1208344, -53.2332687, 31.9029140, -85.2582397, 85.3541031
8: -57.9229469, 39.3557777, -57.7698746, 39.1715088, -97.0944519, 97.1256561
9: -41.8785439, 42.8375969, -41.7904816, 42.5099792, -84.2300720, 84.4829254
10: -58.6347351, 49.1353226, -58.4545517, 48.7991028, -107.4338379, 107.5898743
11: -49.1308556, 28.0172081, -48.6560059, 27.7777863, -76.9086456, 76.6732178
12: -66.3363953, 41.8202972, -66.1806030, 41.5447311, -106.4698029, 106.5545654
13: -60.5337410, 50.5597115, -60.4459229, 49.9324913, -110.4662323, 111.0056305
14: -86.4609985, 36.2130241, -85.9342957, 36.1418762, -122.6028748, 122.1473236
15: -41.7410889, 45.0916061, -41.4808159, 44.8263779, -86.5674667, 86.5724182
16: -61.5377007, 39.6650085, -61.2890892, 39.3315353, -100.8692322, 100.9541016
17: -80.7434998, 33.1378555, -80.2085266, 33.0144081, -113.7579041, 113.3463821
18: -46.3551254, 45.7645493, -45.7280045, 45.6718140, -92.0269394, 91.4925537
19: -35.7006035, 30.1256638, -35.3064308, 30.0947266, -65.7953339, 65.4320984
20: -40.8712692, 26.7864990, -40.5290146, 26.7460938, -67.6173630, 67.3155136
21: -45.6318207, 33.9672585, -45.2075272, 33.9269333, -79.5587540, 79.1747894
22: -36.8458443, 39.4700851, -36.3356171, 39.3940353, -75.6927948, 75.2523804
23: -34.4359589, 34.8526917, -33.9800415, 34.8009720, -69.2369308, 68.8327332
24: -39.3457222, 35.2736969, -38.7972031, 35.2413864, -74.5871124, 74.0709000
25: -36.8487549, 42.6705017, -36.3847046, 42.6023331, -79.4510880, 79.0552063
26: -52.3721390, 54.9011993, -51.7554092, 54.8148880, -107.1870270, 106.6566086
27: -43.5077133, 31.4410000, -42.9910011, 31.3978519, -74.9055634, 74.4319992
28: -35.3284264, 38.1154137, -34.8824768, 38.0636864, -73.3921127, 72.9978943
29: -34.1362305, 32.3206253, -33.6747322, 32.2742348, -66.2767792, 65.8557816
30: -49.8881989, 30.4500141, -49.3862228, 30.3536053, -80.2418060, 79.8362350
31: -47.3890800, 37.2282219, -46.8943405, 37.1848488, -84.5739288, 84.1225586
32: -67.1056671, 16.3141003, -66.9393463, 15.8684788, -79.3193130, 79.6109543
33: -96.5748901, 32.5611038, -96.4249725, 32.1506233, -122.6561584, 122.9256744
34: -83.7048874, 16.0962448, -83.6013260, 15.7294884, -89.2619629, 89.4867172
35: -63.5362549, 33.6682854, -63.4280319, 33.3501282, -95.8741913, 96.1019287
36: -64.7892227, 35.1855965, -64.6762543, 34.9142647, -99.4698486, 99.6353149
37: -100.9919586, 22.1669369, -100.7682495, 21.9377193, -122.1116486, 122.0808716
38: -86.1638489, 33.6500015, -86.0344391, 33.3581886, -119.5220337, 119.6844406
39: -104.2303391, 27.1208038, -104.0416718, 26.6442642, -130.6647644, 130.9646606
40: -91.5843201, 3.5752583, -91.4021606, 3.1771202, -88.4526367, 88.6669540
41: -67.7060699, 22.6171265, -67.5686417, 22.2406712, -86.4518051, 86.6974487
42: -60.6742134, 15.3939438, -60.5556259, 14.9832382, -71.6802063, 71.9033203

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1674

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1137446, upper bound: 46.0975397
time: 67.28 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1137446, upper bound: 46.0975397
time: 111.33 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -67.4587555, 43.9871254, -67.7133331, 43.9832611, -110.9245453, 111.1910934
1: -38.0735016, 35.2173691, -38.2965317, 35.2145309, -72.8494873, 73.0867920
2: -29.4445515, 37.8111649, -29.6082497, 37.8137817, -66.9333267, 67.1099625
3: -43.5944252, 37.4837952, -43.8900528, 37.4727097, -81.0671387, 81.3738480
4: -44.5744400, 39.5693893, -44.8153152, 39.5604706, -84.1349106, 84.3847046
5: -40.7959938, 41.8833771, -41.0228424, 41.8846931, -82.6806870, 82.9062195
6: -72.2246170, 13.1952591, -72.2965469, 13.2395153, -80.9489212, 80.9225006
7: -53.2686234, 32.0261536, -53.4811783, 32.0436249, -85.3122482, 85.5073318
8: -57.8099899, 39.3363266, -58.0617065, 39.3587875, -97.1687775, 97.3980331
9: -41.7987099, 42.7022858, -42.0440369, 42.6436310, -84.2815704, 84.6002197
10: -58.5114136, 48.8570862, -58.6322556, 48.8972397, -107.4086533, 107.4893417
11: -48.9797363, 27.7979774, -48.8192139, 27.9520779, -76.9318161, 76.6171875
12: -66.3028259, 41.6028976, -66.3462753, 41.7574768, -106.6578522, 106.5142899
13: -60.3723068, 50.1222496, -60.5766678, 50.0369148, -110.4092255, 110.6989136
14: -86.1622314, 36.1093330, -86.1644058, 36.2280502, -122.3902817, 122.2737427
15: -41.5016136, 45.0027504, -41.7027435, 44.9610863, -86.4626999, 86.7054901
16: -61.3879776, 39.3913345, -61.5533943, 39.4090500, -100.7970276, 100.9447327
17: -80.5594025, 33.0048065, -80.4271622, 33.2241173, -113.7835236, 113.4319687
18: -45.8715363, 45.5801239, -45.8697090, 45.8453674, -91.7169037, 91.4498291
19: -35.5571022, 30.0701275, -35.4661560, 30.2823486, -65.8394470, 65.5362854
20: -40.7078323, 26.7223892, -40.6839600, 26.9340630, -67.6418915, 67.4063492
21: -45.4801903, 33.9074326, -45.3746758, 34.0756912, -79.5558777, 79.2821045
22: -36.5136909, 39.3204269, -36.4931564, 39.5543213, -75.5129013, 75.2531128
23: -34.2650871, 34.7950935, -34.1458206, 34.9997673, -69.2648544, 68.9409180
24: -39.0933113, 35.1837845, -38.9953651, 35.4116211, -74.5049286, 74.1791534
25: -36.6468277, 42.5522308, -36.5913429, 42.8492279, -79.4960556, 79.1435699
26: -51.9394150, 54.7362213, -51.9198532, 55.0416412, -106.9810562, 106.6560745
27: -43.1466026, 31.3288231, -43.1123123, 31.4654617, -74.6120605, 74.4411316
28: -35.1199226, 38.0366669, -35.0396347, 38.2705536, -73.3904724, 73.0763016
29: -33.9694710, 32.2447472, -33.8593788, 32.4089241, -66.2397919, 65.9596024
30: -49.7153130, 30.3612652, -49.5611916, 30.5319023, -80.2472153, 79.9224548
31: -47.1967545, 37.1456985, -47.1120872, 37.4155312, -84.6122894, 84.2577820
32: -66.9671478, 16.0139847, -67.0388336, 15.9850044, -79.3066101, 79.4047089
33: -96.4489212, 32.3131943, -96.4931183, 32.2943459, -122.6968536, 122.7454529
34: -83.5781097, 15.8706017, -83.6392975, 15.8865051, -89.2996826, 89.3030319
35: -63.4116516, 33.4519653, -63.4674988, 33.4883842, -95.8917999, 95.9314423
36: -64.6956711, 35.0424728, -64.7498474, 35.0664444, -99.5386353, 99.5582733
37: -100.9116516, 21.9879646, -100.9515839, 22.0981293, -122.1734924, 122.0886841
38: -86.0790405, 33.5188141, -86.1609344, 33.5357513, -119.6147919, 119.6797485
39: -104.0673676, 26.8139420, -104.1602173, 26.7580013, -130.6195374, 130.7720032
40: -91.4551544, 3.3334475, -91.5098877, 3.2648449, -88.4189911, 88.5251160
41: -67.5510864, 22.3215752, -67.6171951, 22.3200741, -86.3908234, 86.4466171
42: -60.5258789, 15.0312920, -60.6039658, 15.0930042, -71.7129974, 71.5851517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1674

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1137446, upper bound: 46.0921709
time: 107.07 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1341164, upper bound: 46.0921709
time: 73.03 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -67.5734634, 44.1119614, -67.7099762, 43.9875793, -111.0481415, 111.3185806
1: -38.1103287, 35.2974701, -38.2935028, 35.2180214, -72.8876648, 73.1765900
2: -29.5225830, 37.9019470, -29.6086502, 37.8173981, -66.9996490, 67.2383804
3: -43.7094040, 37.7195587, -43.9090996, 37.4741974, -81.1836014, 81.6286621
4: -44.7855759, 39.6736374, -44.8174286, 39.5677795, -84.3533554, 84.4910660
5: -40.8591805, 42.0565262, -41.0267639, 41.8884964, -82.7476807, 83.0832901
6: -72.3869553, 13.5196304, -72.3178101, 13.2428417, -81.1011200, 81.2738037
7: -53.3612061, 32.1782150, -53.4840546, 32.0457840, -85.4069901, 85.6622696
8: -57.9300385, 39.4297409, -58.0637970, 39.3614922, -97.2915344, 97.4935379
9: -41.8913345, 42.8897781, -42.0567703, 42.6445427, -84.3710480, 84.8031540
10: -58.6567955, 49.1537552, -58.6527100, 48.9007034, -107.5574951, 107.8064651
11: -49.1790390, 28.0246944, -48.8280602, 27.9534798, -77.1325226, 76.8527527
12: -66.3981857, 41.8276825, -66.3537827, 41.7626190, -106.7689209, 106.7341156
13: -60.5458260, 50.5851440, -60.6094284, 50.0398903, -110.5857162, 111.1945724
14: -86.5112305, 36.2181816, -86.1710510, 36.2375069, -122.7487335, 122.3892365
15: -41.7490845, 45.1467323, -41.7058983, 44.9759941, -86.7250824, 86.8526306
16: -61.5588493, 39.6889496, -61.5705299, 39.4125519, -100.9714050, 101.2594757
17: -80.7914734, 33.1522255, -80.4307938, 33.2339478, -114.0254211, 113.5830231
18: -46.3942909, 45.7742844, -45.8725586, 45.8872986, -92.2815857, 91.6468430
19: -35.7541885, 30.1298618, -35.4700012, 30.2917233, -66.0459137, 65.5998611
20: -40.9281006, 26.7930031, -40.6885948, 26.9460354, -67.8741379, 67.4815979
21: -45.6818047, 33.9714775, -45.3803482, 34.0831146, -79.7649231, 79.3518219
22: -36.8996964, 39.4754143, -36.4970894, 39.5804138, -75.9362793, 75.4134979
23: -34.4972305, 34.8576431, -34.1480942, 35.0066490, -69.5038757, 69.0057373
24: -39.4190331, 35.2782707, -38.9964600, 35.4320488, -74.8510818, 74.2747345
25: -36.9214020, 42.6763306, -36.5953789, 42.8687096, -79.7901154, 79.2717133
26: -52.4238281, 54.9110641, -51.9251823, 55.0741005, -107.4979248, 106.8362427
27: -43.5521622, 31.4468708, -43.1161423, 31.4908905, -75.0430527, 74.5630112
28: -35.3879433, 38.1216965, -35.0436325, 38.2832527, -73.6711960, 73.1653290
29: -34.1984444, 32.3250160, -33.8611870, 32.4211273, -66.4858322, 66.0425186
30: -49.9514351, 30.4562130, -49.5662117, 30.5339336, -80.4853668, 80.0224228
31: -47.4669533, 37.2346268, -47.1157837, 37.4315071, -84.8984604, 84.3504105
32: -67.1498413, 16.3216896, -67.0625305, 15.9880219, -79.4879913, 79.7413788
33: -96.5968781, 32.5732117, -96.5121002, 32.2970390, -122.8410339, 123.0271301
34: -83.7208710, 16.1052132, -83.6609192, 15.8887291, -89.4564133, 89.5610809
35: -63.5502014, 33.6778564, -63.4883537, 33.4907837, -96.0296631, 96.1818237
36: -64.8197784, 35.1919708, -64.7644882, 35.0680618, -99.6646118, 99.7286682
37: -101.0523682, 22.1753044, -100.9539032, 22.1015739, -122.3435974, 122.2726593
38: -86.2035599, 33.6621475, -86.1709061, 33.5380745, -119.7416382, 119.8330536
39: -104.2664032, 27.1272888, -104.1836853, 26.7594872, -130.8199158, 131.1125793
40: -91.6143723, 3.5837612, -91.5113831, 3.2652664, -88.5827332, 88.7827454
41: -67.7261658, 22.6259956, -67.6399536, 22.3240395, -86.5668869, 86.7789993
42: -60.6958275, 15.4007034, -60.6333084, 15.0958691, -71.8776855, 71.9853668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1674

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1137446, upper bound: 46.1341162
time: 53.42 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1137446, upper bound: 46.1341162
time: 57.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 113.58 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0284080
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0284080
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0703258
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0703258
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0649716
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0649715
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1474986, upper bound: 46.1068700
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1474986, upper bound: 46.1068700
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1137446, upper bound: 46.0555783
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1341164, upper bound: 46.0555783
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1137446, upper bound: 46.0975397
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1137446, upper bound: 46.0975397
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1137446, upper bound: 46.0921709
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1341164, upper bound: 46.0921709
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1137446, upper bound: 46.1341162
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 113.58
Output dim: 29, lower bound: -46.1137446, upper bound: 46.1341162

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -67.3612366, 43.8620605, -67.3521423, 43.7910194, -110.6392822, 110.7012711
1: -38.0198746, 35.1083832, -38.0215912, 35.0477448, -72.6328354, 72.6968231
2: -29.4132767, 37.7155762, -29.4106655, 37.6648750, -66.7565918, 66.8072891
3: -43.5687141, 37.3262596, -43.6104965, 37.2473297, -80.8160400, 80.9367523
4: -44.5335388, 39.4112434, -44.5358200, 39.3620262, -83.8955688, 83.9470673
5: -40.7728195, 41.7637177, -40.7907333, 41.7030754, -82.4758911, 82.5544510
6: -72.1448441, 13.1200676, -72.1637268, 13.1178036, -80.6984863, 80.7145844
7: -53.2019272, 31.9472809, -53.2225380, 31.8898010, -85.0917282, 85.1698151
8: -57.7630844, 39.2078094, -57.7611046, 39.1461258, -96.9092102, 96.9689178
9: -41.6971550, 42.5524940, -41.7141151, 42.5005341, -84.0388031, 84.1068192
10: -58.3904877, 48.7722626, -58.4145012, 48.7628517, -107.1533356, 107.1867676
11: -48.6333237, 27.6955910, -48.6234703, 27.6938820, -76.3272095, 76.3190613
12: -66.1865692, 41.5194778, -66.1513824, 41.5252686, -106.2857666, 106.2403793
13: -60.2701378, 49.9335327, -60.3426247, 49.9110222, -110.1811600, 110.2761536
14: -85.9323730, 36.0470924, -85.8996277, 36.0823822, -122.0147552, 121.9467163
15: -41.4209671, 44.8142548, -41.4226379, 44.7956390, -86.2166061, 86.2368927
16: -61.2080841, 39.3298187, -61.2360954, 39.3134232, -100.5215073, 100.5659180
17: -80.2052765, 32.8889084, -80.1746521, 32.9162140, -113.1214905, 113.0635605
18: -45.7320328, 45.5269318, -45.7046661, 45.6052933, -91.3373260, 91.2315979
19: -35.3298454, 29.9901314, -35.2832756, 30.0109901, -65.3408356, 65.2734070
20: -40.5604095, 26.6876640, -40.5087662, 26.7153893, -67.2758026, 67.1964264
21: -45.2098503, 33.8306198, -45.1741371, 33.8478699, -79.0577240, 79.0047607
22: -36.3551712, 39.2970543, -36.3104172, 39.3576736, -75.1544800, 75.0477371
23: -34.0200424, 34.7161636, -33.9615707, 34.7307205, -68.7507629, 68.6777344
24: -38.8448753, 35.1308289, -38.7777214, 35.1784935, -74.0233688, 73.9085541
25: -36.4301949, 42.4831123, -36.3607559, 42.5276031, -78.9577942, 78.8438721
26: -51.7655411, 54.6883698, -51.7246361, 54.7658157, -106.5313568, 106.4130096
27: -43.0036087, 31.3014183, -42.9679832, 31.3597603, -74.3633728, 74.2694016
28: -34.9204636, 37.9889946, -34.8632660, 38.0191879, -72.9396515, 72.8522644
29: -33.7017326, 32.1977768, -33.6489792, 32.2252235, -65.7864990, 65.7070847
30: -49.4128761, 30.2917652, -49.3588028, 30.2939167, -79.7067947, 79.6505661
31: -46.9355316, 37.0470619, -46.8663445, 37.0813332, -84.0168610, 83.9134064
32: -66.8383331, 15.8481617, -66.8521423, 15.8488007, -79.0388947, 79.0523376
33: -96.3057938, 32.1428032, -96.3274078, 32.1402969, -122.3926239, 122.4132843
34: -83.4974518, 15.7100334, -83.5359650, 15.7109833, -89.0344009, 89.0575943
35: -63.3282700, 33.3429871, -63.3699074, 33.3409195, -95.6642609, 95.7046814
36: -64.5880127, 34.9007530, -64.5976868, 34.9003868, -99.2563324, 99.2657471
37: -100.7638245, 21.9253368, -100.7338257, 21.9278107, -121.8457031, 121.8092957
38: -85.9950943, 33.3373413, -85.9857941, 33.3349686, -119.3300629, 119.3231354
39: -103.9027100, 26.6336823, -103.9291611, 26.6360226, -130.3304291, 130.3548737
40: -91.3428421, 3.1683712, -91.3450165, 3.1680288, -88.1992493, 88.2012024
41: -67.4689941, 22.2242126, -67.5095673, 22.2250957, -86.2039871, 86.2430725
42: -60.4650688, 14.9656849, -60.5054169, 14.9698334, -71.4205704, 71.4435883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=365, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1440819, upper bound: 46.0046154
time: 75.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1440819, upper bound: 46.0249228
time: 53.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -67.3612366, 43.8620605, -67.4239502, 43.8414154, -110.6922913, 110.7742615
1: -38.0198746, 35.1083832, -38.0680199, 35.0945473, -72.6838989, 72.7435837
2: -29.4132767, 37.7155762, -29.4337463, 37.7068748, -66.8005829, 66.8311234
3: -43.5687141, 37.3262596, -43.6324577, 37.3245316, -80.8932495, 80.9587173
4: -44.5335388, 39.4112434, -44.5701981, 39.4467239, -83.9802628, 83.9814453
5: -40.7728195, 41.7637177, -40.8093948, 41.7590752, -82.5318909, 82.5731125
6: -72.1448441, 13.1200676, -72.2088852, 13.1823845, -80.7564240, 80.7610931
7: -53.2019272, 31.9472809, -53.2833138, 31.9113083, -85.1132355, 85.2305908
8: -57.7630844, 39.2078094, -57.8009224, 39.2006226, -96.9637070, 97.0087280
9: -41.6971550, 42.5524940, -41.8029633, 42.5981750, -84.1419983, 84.2008057
10: -58.3904877, 48.7722626, -58.5133400, 48.8293076, -107.2197952, 107.2855988
11: -48.6333237, 27.6955910, -48.9216576, 27.7888660, -76.4221878, 76.6172485
12: -66.1865692, 41.5194778, -66.2057495, 41.6012611, -106.3596497, 106.2964249
13: -60.2701378, 49.9335327, -60.4327393, 50.0742989, -110.3444366, 110.3662720
14: -85.9323730, 36.0470924, -86.0792236, 36.1395111, -122.0718842, 122.1263123
15: -41.4209671, 44.8142548, -41.4953194, 44.9289703, -86.3499374, 86.3095703
16: -61.2080841, 39.3298187, -61.3948402, 39.3510323, -100.5591125, 100.7246552
17: -80.2052765, 32.8889084, -80.4808121, 33.0177689, -113.2230453, 113.3697205
18: -45.7320328, 45.5269318, -45.8049850, 45.6487198, -91.3807526, 91.3319168
19: -35.3298454, 29.9901314, -35.4568443, 30.0867805, -65.4166260, 65.4469757
20: -40.5604095, 26.6876640, -40.5993423, 26.7435665, -67.3039780, 67.2870026
21: -45.2098503, 33.8306198, -45.3943825, 33.9203873, -79.1302338, 79.2250061
22: -36.3551712, 39.2970543, -36.4150810, 39.3756371, -75.1743088, 75.1501923
23: -34.0200424, 34.7161636, -34.1453934, 34.8046722, -68.8247147, 68.8615570
24: -38.8448753, 35.1308289, -38.9528236, 35.2269592, -74.0718384, 74.0836487
25: -36.4301949, 42.4831123, -36.5046692, 42.5908432, -79.0210419, 78.9877777
26: -51.7655411, 54.6883698, -51.8468361, 54.8037491, -106.5692902, 106.5352020
27: -43.0036087, 31.3014183, -43.0665627, 31.3812923, -74.3849030, 74.3679810
28: -34.9204636, 37.9889946, -35.0032043, 38.0606384, -72.9811020, 72.9922028
29: -33.7017326, 32.1977768, -33.8545380, 32.2678223, -65.8299408, 65.9146423
30: -49.4128761, 30.2917652, -49.5979652, 30.3572350, -79.7701111, 79.8897324
31: -46.9355316, 37.0470619, -47.0495644, 37.1735497, -84.1090851, 84.0966263
32: -66.8383331, 15.8481617, -66.9368820, 16.0070152, -79.1970062, 79.1386795
33: -96.3057938, 32.1428032, -96.4485474, 32.2984924, -122.5495911, 122.5290375
34: -83.4974518, 15.7100334, -83.6006470, 15.8625622, -89.1715546, 89.1132812
35: -63.3282700, 33.3429871, -63.4393272, 33.4403152, -95.7580872, 95.7732315
36: -64.5880127, 34.9007530, -64.6749268, 35.0357323, -99.3930206, 99.3417358
37: -100.7638245, 21.9253368, -100.8212357, 21.9820824, -121.9017181, 121.8982239
38: -85.9950943, 33.3373413, -86.0299835, 33.5043068, -119.4994049, 119.3673248
39: -103.9027100, 26.6336823, -104.0577545, 26.8097610, -130.5048370, 130.4832916
40: -91.3428421, 3.1683712, -91.4271622, 3.3246756, -88.3514557, 88.2812729
41: -67.4689941, 22.2242126, -67.5715942, 22.3136597, -86.2903061, 86.3033524
42: -60.4650688, 14.9656849, -60.5446320, 15.0286560, -71.4697342, 71.4816055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=365, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1440819, upper bound: 46.0046154
time: 62.47 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1440819, upper bound: 46.0249228
time: 102.75 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -67.4758530, 43.9868927, -67.3488007, 43.7953529, -110.7627563, 110.8287964
1: -38.0567322, 35.1883926, -38.0185394, 35.0511818, -72.6710205, 72.7865601
2: -29.4912605, 37.8062668, -29.4110184, 37.6685104, -66.8229828, 66.9357224
3: -43.6837082, 37.5619926, -43.6295853, 37.2487755, -80.9324799, 81.1915741
4: -44.7446480, 39.5152473, -44.5378838, 39.3693352, -84.1139832, 84.0531311
5: -40.8360062, 41.9367523, -40.7946472, 41.7068825, -82.5428925, 82.7313995
6: -72.3070755, 13.4443340, -72.1849899, 13.1211643, -80.8505096, 81.0657501
7: -53.2944984, 32.0994072, -53.2254524, 31.8919353, -85.1864319, 85.3248596
8: -57.8830833, 39.3012161, -57.7631607, 39.1488724, -97.0319519, 97.0643768
9: -41.7897835, 42.7399750, -41.7268982, 42.5014610, -84.1283722, 84.3097839
10: -58.5359039, 49.0688477, -58.4349327, 48.7663651, -107.3022690, 107.5037842
11: -48.8326416, 27.9224644, -48.6322632, 27.6952820, -76.5279236, 76.5547256
12: -66.2820282, 41.7442513, -66.1587982, 41.5304413, -106.3969421, 106.4602661
13: -60.4437256, 50.3964729, -60.3753929, 49.9140511, -110.3577728, 110.7718658
14: -86.2810593, 36.1559067, -85.9063110, 36.0918541, -122.3729095, 122.0622177
15: -41.6685867, 44.9583473, -41.4258537, 44.8105431, -86.4791260, 86.3842010
16: -61.3789902, 39.6273041, -61.2532959, 39.3168869, -100.6958771, 100.8806000
17: -80.4371490, 33.0363007, -80.1783295, 32.9260941, -113.3632431, 113.2146301
18: -46.2548027, 45.7211266, -45.7075424, 45.6472321, -91.9020386, 91.4286652
19: -35.5270157, 30.0498772, -35.2871056, 30.0204048, -65.5474243, 65.3369827
20: -40.7806396, 26.7583160, -40.5134048, 26.7273483, -67.5079880, 67.2717209
21: -45.4114761, 33.8947067, -45.1798439, 33.8552742, -79.2667542, 79.0745544
22: -36.7410736, 39.4519768, -36.3143845, 39.3837433, -75.5777740, 75.2080917
23: -34.2520676, 34.7787476, -33.9638596, 34.7375908, -68.9896545, 68.7426071
24: -39.1704407, 35.2253456, -38.7788200, 35.1989021, -74.3693390, 74.0041656
25: -36.7046967, 42.6073265, -36.3647842, 42.5470734, -79.2517700, 78.9721069
26: -52.2498741, 54.8632011, -51.7299385, 54.7982941, -107.0481720, 106.5931396
27: -43.4090195, 31.4195328, -42.9718285, 31.3852043, -74.7942200, 74.3913574
28: -35.1883392, 38.0739975, -34.8672867, 38.0318527, -73.2201920, 72.9412842
29: -33.9304504, 32.2780266, -33.6508331, 32.2374229, -66.0322952, 65.7900391
30: -49.6489105, 30.3866425, -49.3638344, 30.2959518, -79.9448624, 79.7504730
31: -47.2060013, 37.1359558, -46.8700943, 37.0972977, -84.3032990, 84.0060501
32: -67.0210114, 16.1557846, -66.8758545, 15.8518333, -79.2203674, 79.3889618
33: -96.4539185, 32.4028702, -96.3463287, 32.1429329, -122.5369568, 122.6949921
34: -83.6403046, 15.9446754, -83.5576324, 15.7132511, -89.1912079, 89.3157272
35: -63.4669342, 33.5689087, -63.3907166, 33.3432693, -95.8021698, 95.9550323
36: -64.7121277, 35.0502930, -64.6123352, 34.9020996, -99.3824768, 99.4360962
37: -100.9044800, 22.1127491, -100.7360229, 21.9313526, -122.0156555, 121.9932404
38: -86.1195755, 33.4806290, -85.9957504, 33.3372650, -119.4568405, 119.4763794
39: -104.1017609, 26.9470501, -103.9525528, 26.6375122, -130.5308533, 130.6953430
40: -91.5021362, 3.4185734, -91.3465042, 3.1684446, -88.3628845, 88.4587631
41: -67.6440506, 22.5284653, -67.5323334, 22.2290993, -86.3799438, 86.5754089
42: -60.6349449, 15.3350906, -60.5347672, 14.9727325, -71.5849609, 71.8437881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=365, inp2_unstable=365, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1674

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1440819, upper bound: 46.0465376
time: 70.19 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -46.1440819, upper bound: 46.0668820
time: 64.55 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 137.17 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 137.17
Output dim: 29, lower bound: -46.1440819, upper bound: 46.0046154
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 137.17
Output dim: 29, lower bound: -46.1440819, upper bound: 46.0249228
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 137.17
Output dim: 29, lower bound: -46.1440819, upper bound: 46.0046154
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 137.17
Output dim: 29, lower bound: -46.1440819, upper bound: 46.0249228
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 137.17
Output dim: 29, lower bound: -46.1440819, upper bound: 46.0465376
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 137.17
Output dim: 29, lower bound: -46.1440819, upper bound: 46.0668820
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 137.17
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0703258
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 137.17
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0649716
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 137.17
Output dim: 29, lower bound: -46.1474986, upper bound: 46.0649715
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 137.17
Output dim: 29, lower bound: -46.1474986, upper bound: 46.1068700
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 137.17
Output dim: 29, lower bound: -46.1474986, upper bound: 46.1068700
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 137.17
Output dim: 29, lower bound: -46.1137446, upper bound: 46.0555783
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 137.17
Output dim: 29, lower bound: -46.1341164, upper bound: 46.0555783
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 137.17
Output dim: 29, lower bound: -46.1137446, upper bound: 46.0975397
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 137.17
Output dim: 29, lower bound: -46.1137446, upper bound: 46.0975397
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 137.17
Output dim: 29, lower bound: -46.1137446, upper bound: 46.0921709
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 137.17
Output dim: 29, lower bound: -46.1341164, upper bound: 46.0921709
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 137.17
Output dim: 29, lower bound: -46.1137446, upper bound: 46.1341162
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 137.17
Output dim: 29, lower bound: -46.1137446, upper bound: 46.1341162
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=65.98654174804688
rel_dist={29: [-46.200630847122696, 46.20063084433468]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -45.0417567, upper bound: 45.0116393
time: 59.16 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -45.0362190, upper bound: 45.0362190
time: 61.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 121.05 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 121.05
Output dim: 29, lower bound: -45.0417567, upper bound: 45.0116393
IS_A2, status: Status.UNKNOWN, split count: 1, time: 121.05
Output dim: 29, lower bound: -45.0362190, upper bound: 45.0362190

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -67.4241028, 43.9691849, -67.4350586, 43.9775848, -110.6429443, 110.6456451
1: -38.0452576, 35.1950378, -38.0511665, 35.2031250, -72.6679230, 72.6661835
2: -29.4349117, 37.7924118, -29.4386616, 37.7997589, -66.6806335, 66.6776352
3: -43.6466942, 37.4368362, -43.6547623, 37.4454193, -81.0921173, 81.0915985
4: -44.5588341, 39.5477715, -44.5725937, 39.5577812, -84.1166153, 84.1203613
5: -40.8166656, 41.8543777, -40.8186340, 41.8637161, -82.6803818, 82.6730118
6: -72.2819138, 13.1482906, -72.2971191, 13.1525078, -80.1069946, 80.1194839
7: -53.2542725, 32.0250435, -53.2589111, 32.0328255, -85.2870941, 85.2839508
8: -57.7837524, 39.3276825, -57.7876740, 39.3411484, -97.1249008, 97.1153564
9: -41.7618484, 42.6260910, -41.7985458, 42.6311989, -84.0588379, 84.0915451
10: -58.4933395, 48.8165512, -58.5039406, 48.8367996, -107.3301392, 107.3204956
11: -48.7610054, 27.7154675, -48.7742691, 27.7577877, -76.5187912, 76.4897385
12: -66.3109055, 41.5532341, -66.3238449, 41.5608749, -106.0737000, 106.0779266
13: -60.4191322, 49.9962234, -60.4578857, 50.0070267, -110.4261627, 110.4541092
14: -86.0290222, 36.1202774, -86.0447998, 36.1460266, -122.1750488, 122.1650772
15: -41.4489746, 44.9460793, -41.4771996, 44.9547043, -86.4036789, 86.4232788
16: -61.3155556, 39.3793259, -61.3346024, 39.3884697, -100.6390457, 100.6481400
17: -80.2972641, 32.9762955, -80.3137512, 33.0226479, -113.3199158, 113.2900467
18: -45.8070526, 45.6773338, -45.8192444, 45.6912079, -91.4982605, 91.4965820
19: -35.4148064, 30.0358467, -35.4253540, 30.0736809, -65.4884872, 65.4611969
20: -40.6463013, 26.7496262, -40.6551971, 26.7618847, -67.4081879, 67.4048233
21: -45.3004723, 33.8702164, -45.3153687, 33.9074821, -79.2079544, 79.1855850
22: -36.4442444, 39.4071846, -36.4562683, 39.4129829, -75.0870056, 75.0914001
23: -34.1080399, 34.7547913, -34.1167297, 34.7863426, -68.8943787, 68.8715210
24: -38.9556770, 35.2179832, -38.9659233, 35.2417221, -74.1974030, 74.1839066
25: -36.5341148, 42.5710716, -36.5451202, 42.6009560, -79.1350708, 79.1161957
26: -51.8562584, 54.8376122, -51.8711052, 54.8462753, -106.7025299, 106.7087173
27: -43.0847702, 31.4085827, -43.0958023, 31.4163857, -74.5011597, 74.5043869
28: -35.0060806, 38.0555801, -35.0144348, 38.0726967, -73.0787811, 73.0700150
29: -33.8021812, 32.2545853, -33.8156052, 32.2742577, -65.8006439, 65.7939835
30: -49.5133018, 30.3148060, -49.5256004, 30.3474407, -79.8607407, 79.8404083
31: -47.0561867, 37.1188965, -47.0694008, 37.1630478, -84.2192383, 84.1882935
32: -66.9891434, 15.8724270, -67.0229797, 15.8816071, -78.7621765, 78.7869797
33: -96.4113007, 32.1759300, -96.4509811, 32.1811829, -121.8208771, 121.8529968
34: -83.6097717, 15.7388000, -83.6328430, 15.7474785, -88.1089783, 88.1208267
35: -63.4354172, 33.3702469, -63.4550858, 33.3742332, -95.2259521, 95.2443237
36: -64.6954803, 34.9204254, -64.7309723, 34.9268341, -99.2955170, 99.3243637
37: -100.8906860, 21.9551201, -100.9091644, 21.9589844, -121.7042389, 121.7183990
38: -86.0990906, 33.3707848, -86.1238861, 33.3815422, -119.4806366, 119.4946747
39: -104.0569763, 26.6585693, -104.1030579, 26.6626358, -130.2802582, 130.3229523
40: -91.4443054, 3.1938534, -91.4752731, 3.1984835, -87.7064972, 87.7337112
41: -67.5957184, 22.2521381, -67.6156464, 22.2582932, -86.0180817, 86.0319748
42: -60.5945625, 14.9930954, -60.6062546, 14.9989548, -70.8687897, 70.8691711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 29, lower bound: -45.0400176, upper bound: 44.9825714
time: 124.18 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 29, lower bound: -45.0400176, upper bound: 45.0098861
time: 57.69 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -67.4959259, 44.0195923, -67.4427490, 43.9829788, -110.7227783, 110.7058334
1: -38.0917435, 35.2418556, -38.0548401, 35.2091522, -72.7203751, 72.7211304
2: -29.4580460, 37.8343887, -29.4403496, 37.8049889, -66.7088394, 66.7244339
3: -43.6686363, 37.5141373, -43.6584015, 37.4512177, -81.1198578, 81.1725388
4: -44.5932007, 39.6324539, -44.5776176, 39.5658150, -84.1590118, 84.2100677
5: -40.8353195, 41.9103394, -40.8199081, 41.8689651, -82.7042847, 82.7302475
6: -72.3270569, 13.2128582, -72.3083496, 13.1557217, -80.1757660, 80.1889343
7: -53.3150978, 32.0465050, -53.2617798, 32.0350037, -85.3500977, 85.3082886
8: -57.8236465, 39.3821983, -57.7901993, 39.3493614, -97.1730042, 97.1723938
9: -41.8505096, 42.7237167, -41.8232574, 42.6341858, -84.1549225, 84.2250443
10: -58.5922127, 48.8829765, -58.5124931, 48.8474197, -107.4396362, 107.3954697
11: -49.0592079, 27.8105030, -48.7840042, 27.7965813, -76.8557892, 76.5945053
12: -66.3652802, 41.6292725, -66.3316650, 41.5671997, -106.1379547, 106.1589966
13: -60.5091972, 50.1594810, -60.4876938, 50.0138931, -110.5230865, 110.6471710
14: -86.2087250, 36.1774368, -86.0559769, 36.1696281, -122.3783569, 122.2334137
15: -41.5217171, 45.0794029, -41.5032272, 44.9614868, -86.4832001, 86.5826263
16: -61.4743958, 39.4169655, -61.3506737, 39.3930969, -100.8084717, 100.7019806
17: -80.6034775, 33.0777664, -80.3266373, 33.0629730, -113.6664505, 113.4044037
18: -45.9073524, 45.7207909, -45.8265877, 45.7010040, -91.6083527, 91.5473785
19: -35.5884514, 30.1116009, -35.4335823, 30.1093826, -65.6978302, 65.5451813
20: -40.7368851, 26.7777939, -40.6613579, 26.7670021, -67.5038910, 67.4391479
21: -45.5208397, 33.9427834, -45.3275337, 33.9406052, -79.4614410, 79.2703171
22: -36.5489273, 39.4252052, -36.4647522, 39.4171982, -75.1932449, 75.1221008
23: -34.2918701, 34.8287659, -34.1238861, 34.8175583, -69.1094284, 68.9526520
24: -39.1308250, 35.2664108, -38.9734573, 35.2592239, -74.3900452, 74.2398682
25: -36.6780777, 42.6343117, -36.5534515, 42.6251984, -79.3032761, 79.1877594
26: -51.9784889, 54.8755112, -51.8806953, 54.8539276, -106.8324127, 106.7562103
27: -43.1833572, 31.4301414, -43.1032143, 31.4205399, -74.6038971, 74.5333557
28: -35.1460381, 38.0970230, -35.0208282, 38.0867691, -73.2328033, 73.1178513
29: -34.0077400, 32.2971497, -33.8253212, 32.2906952, -66.0241470, 65.8453979
30: -49.7524719, 30.3781452, -49.5350723, 30.3704147, -80.1228867, 79.9132156
31: -47.2395172, 37.2110825, -47.0797653, 37.2055740, -84.4450912, 84.2908478
32: -67.0737762, 16.0306511, -67.0511551, 15.8886452, -78.8497925, 78.9731598
33: -96.5323563, 32.3340988, -96.4889832, 32.1830559, -121.9338684, 122.0451050
34: -83.6744080, 15.8904438, -83.6525345, 15.7547035, -88.1705933, 88.2699585
35: -63.5047493, 33.4696655, -63.4719315, 33.3768349, -95.2920837, 95.3668289
36: -64.7727356, 35.0557442, -64.7575378, 34.9323883, -99.3758850, 99.4862823
37: -100.9780731, 22.0094032, -100.9215698, 21.9611835, -121.7957458, 121.7865448
38: -86.1432571, 33.5402031, -86.1351471, 33.3914032, -119.5346603, 119.6753540
39: -104.1855621, 26.8324165, -104.1443939, 26.6649132, -130.4092407, 130.5434418
40: -91.5264664, 3.3505287, -91.4982681, 3.2023621, -87.7889099, 87.9070511
41: -67.6576767, 22.3406715, -67.6310196, 22.2634563, -86.0821381, 86.1323395
42: -60.6337433, 15.0519371, -60.6147003, 15.0032616, -70.9463425, 70.9122086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=367, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 1335
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1373
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1486
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1341
type: B, layer: 1, pos: 987
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 978
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1490
type: B, layer: 1, pos: 1511
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 906
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1340
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 1637
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1512
type: B, layer: 1, pos: 894
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1404
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1332
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 1622
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1309
type: B, layer: 1, pos: 1474
type: B, layer: 1, pos: 1478
type: B, layer: 1, pos: 991
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1320
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1513
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1325
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 1348
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 975
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1420
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 1375
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1357
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 792
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 836
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 1477
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1336
type: B, layer: 1, pos: 1316
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1458
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 1364
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 1412
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1668
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1331
type: B, layer: 1, pos: 1388
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 1380
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1307
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 1356
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1445
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 1652
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 810
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 1381
type: B, layer: 1, pos: 1324
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1355
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1010
type: B, layer: 1, pos: 1281
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 1300
type: B, layer: 1, pos: 796
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1347
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 996
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 1354
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1321
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 802
type: B, layer: 1, pos: 789
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1338
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1737
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1315
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 775
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 1359
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1007
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1441
type: B, layer: 1, pos: 1363
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1510
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 797
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1306
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1379
type: B, layer: 1, pos: 1683
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1284
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 777
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1509
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1009
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1459
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 795
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1396
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1519
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1662

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 29, lower bound: -45.0344555, upper bound: 45.0071613
time: 55.48 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 29, lower bound: -45.0344555, upper bound: 45.0344554
time: 82.54 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 140.44 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 140.44
Output dim: 29, lower bound: -45.0400176, upper bound: 44.9825714
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 140.44
Output dim: 29, lower bound: -45.0400176, upper bound: 45.0098861
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 140.44
Output dim: 29, lower bound: -45.0344555, upper bound: 45.0071613
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 140.44
Output dim: 29, lower bound: -45.0344555, upper bound: 45.0344554

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -67.3881683, 43.8667908, -67.3748627, 43.8073578, -110.4361420, 110.4826279
1: -38.0347137, 35.1111069, -38.0334778, 35.0619583, -72.5160446, 72.5644226
2: -29.4228783, 37.7201538, -29.4184895, 37.6782684, -66.5476151, 66.5859528
3: -43.6403923, 37.3286591, -43.6443176, 37.2633362, -80.9037323, 80.9729767
4: -44.5485992, 39.4490814, -44.5555115, 39.3918648, -83.9404602, 84.0045929
5: -40.8091278, 41.7682571, -40.8061218, 41.7193832, -82.5285110, 82.5743790
6: -72.2310028, 13.1333961, -72.2121735, 13.1276016, -80.0279465, 80.0129471
7: -53.2446671, 31.9475403, -53.2429123, 31.9024353, -85.1471024, 85.1904526
8: -57.7725906, 39.2277527, -57.7690201, 39.1727066, -96.9452972, 96.9967728
9: -41.7439041, 42.5539856, -41.7685013, 42.5106049, -83.9203491, 83.9890976
10: -58.4627724, 48.7888756, -58.4525948, 48.7906418, -107.2534180, 107.2414703
11: -48.6945305, 27.7048435, -48.6635704, 27.7400608, -76.4345932, 76.3684158
12: -66.2272034, 41.5415421, -66.1838303, 41.5413589, -105.9703369, 105.9263611
13: -60.4015350, 49.9506111, -60.4283867, 49.9305687, -110.3321075, 110.3789978
14: -85.9592438, 36.1117363, -85.9291840, 36.1317825, -122.0910263, 122.0409241
15: -41.4370651, 44.8705063, -41.4572449, 44.8289032, -86.2659683, 86.3277512
16: -61.2857170, 39.3441315, -61.2844810, 39.3297653, -100.5491638, 100.5616150
17: -80.2319794, 32.9550514, -80.2049103, 32.9869843, -113.2189636, 113.1599579
18: -45.7522011, 45.6635857, -45.7273102, 45.6681061, -91.4203033, 91.3908997
19: -35.3416977, 30.0294437, -35.3025398, 30.0629921, -65.4046936, 65.3319855
20: -40.5686264, 26.7405396, -40.5252800, 26.7466545, -67.3152771, 67.2658234
21: -45.2318878, 33.8641090, -45.2010155, 33.8972054, -79.1290894, 79.0651245
22: -36.3703728, 39.3992310, -36.3323822, 39.3996239, -74.9976501, 74.9564590
23: -34.0249329, 34.7472115, -33.9771118, 34.7736320, -68.7985687, 68.7243195
24: -38.8553963, 35.2114677, -38.7978859, 35.2307549, -74.0861511, 74.0093536
25: -36.4356155, 42.5621834, -36.3796844, 42.5861588, -79.0217743, 78.9418640
26: -51.7840309, 54.8234253, -51.7504196, 54.8225479, -106.6065826, 106.5738449
27: -43.0210190, 31.4003601, -42.9890594, 31.4026012, -74.4236221, 74.3894196
28: -34.9250031, 38.0461159, -34.8782196, 38.0568275, -72.9818268, 72.9243317
29: -33.7170067, 32.2477570, -33.6732178, 32.2628250, -65.7038422, 65.6447754
30: -49.4267044, 30.3055286, -49.3814011, 30.3318977, -79.7586060, 79.6869278
31: -46.9501190, 37.1098785, -46.8912048, 37.1479988, -84.0981140, 84.0010834
32: -66.9282227, 15.8615246, -66.9218445, 15.8634052, -78.6818237, 78.6732635
33: -96.3773804, 32.1585999, -96.3943558, 32.1522141, -121.7555084, 121.7768097
34: -83.5849304, 15.7261333, -83.5912476, 15.7261887, -88.0528107, 88.0524521
35: -63.4152069, 33.3560791, -63.4212341, 33.3504639, -95.1768646, 95.1900558
36: -64.6515503, 34.9109840, -64.6580200, 34.9110374, -99.2342529, 99.2399139
37: -100.8075867, 21.9429474, -100.7726440, 21.9386616, -121.6003113, 121.5691071
38: -86.0429840, 33.3535690, -86.0308533, 33.3526497, -119.3956299, 119.3844223
39: -104.0034485, 26.6486702, -104.0137177, 26.6461277, -130.2089844, 130.2226868
40: -91.3981781, 3.1817045, -91.3982010, 3.1782331, -87.6398926, 87.6448822
41: -67.5656357, 22.2396545, -67.5651550, 22.2375278, -85.9649353, 85.9664688
42: -60.5633621, 14.9832325, -60.5541763, 14.9824486, -70.8084106, 70.7864990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=334, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1591

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 29, lower bound: -45.0266204, upper bound: 44.9368815
time: 103.62 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 29, lower bound: -45.0266204, upper bound: 44.9690641
time: 64.29 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -67.4192810, 43.9573517, -67.7166443, 43.9851952, -110.6367188, 110.9157791
1: -38.0434837, 35.1865387, -38.2984238, 35.2142448, -72.6733246, 72.9062195
2: -29.4327965, 37.7851944, -29.6103268, 37.8141098, -66.6897583, 66.8482742
3: -43.6449356, 37.4259567, -43.9112244, 37.4737701, -81.1187057, 81.3371811
4: -44.5565414, 39.5379791, -44.8145828, 39.5717812, -84.1283264, 84.3525620
5: -40.8146133, 41.8454628, -41.0348129, 41.8855515, -82.7001648, 82.8802795
6: -72.2731018, 13.1463013, -72.3176193, 13.2416573, -80.2051468, 80.1307526
7: -53.2518997, 32.0171318, -53.4936905, 32.0452728, -85.2971725, 85.5108185
8: -57.7812653, 39.3175964, -58.0629654, 39.3627167, -97.1439819, 97.3805618
9: -41.7594986, 42.6171646, -42.0348129, 42.6451607, -84.0635376, 84.3203278
10: -58.4895401, 48.8113403, -58.6507797, 48.8922424, -107.3817825, 107.4621201
11: -48.7528648, 27.7139111, -48.8355942, 27.9157333, -76.6685944, 76.5495071
12: -66.3022690, 41.5504875, -66.3569489, 41.7593040, -106.2825165, 106.1074829
13: -60.4162102, 49.9815941, -60.5918655, 50.0379601, -110.4541702, 110.5734558
14: -86.0199356, 36.1179733, -86.1659088, 36.2273369, -122.2472687, 122.2838821
15: -41.4467430, 44.9371567, -41.6823425, 44.9785728, -86.4253159, 86.6194992
16: -61.3112984, 39.3732491, -61.5658684, 39.4107170, -100.6528625, 100.8725586
17: -80.2902145, 32.9724922, -80.4271545, 33.2064743, -113.4966888, 113.3996429
18: -45.7997360, 45.6753578, -45.8719025, 45.8835754, -91.6833115, 91.5472565
19: -35.4067154, 30.0346088, -35.4661140, 30.2600060, -65.6667175, 65.5007248
20: -40.6375694, 26.7484856, -40.6848984, 26.9465866, -67.5841522, 67.4333801
21: -45.2924690, 33.8692474, -45.3738747, 34.0533600, -79.3458252, 79.2431183
22: -36.4356918, 39.4057312, -36.4938660, 39.5859833, -75.2529449, 75.1177521
23: -34.0991859, 34.7532578, -34.1451569, 34.9793320, -69.0785217, 68.8984146
24: -38.9443016, 35.2170296, -38.9971390, 35.4213943, -74.3656921, 74.2141724
25: -36.5237312, 42.5693359, -36.5903854, 42.8525162, -79.3762512, 79.1597214
26: -51.8467522, 54.8354301, -51.9201546, 55.0816536, -106.9284058, 106.7555847
27: -43.0749207, 31.4075317, -43.1141777, 31.4956512, -74.5705719, 74.5217133
28: -34.9971466, 38.0537491, -35.0394249, 38.2764053, -73.2735519, 73.0931702
29: -33.7924232, 32.2531433, -33.8596611, 32.4097290, -65.9260025, 65.8319244
30: -49.5033531, 30.3130093, -49.5614090, 30.5121880, -80.0155411, 79.8744202
31: -47.0446472, 37.1176949, -47.1127014, 37.3946266, -84.4392700, 84.2303925
32: -66.9816437, 15.8707085, -67.0450363, 15.9829483, -78.8598785, 78.8053436
33: -96.4042664, 32.1734161, -96.4814453, 32.2985840, -121.9452515, 121.8800201
34: -83.6044159, 15.7370300, -83.6509247, 15.8854523, -88.2523804, 88.1273041
35: -63.4322243, 33.3677979, -63.4815521, 33.4911613, -95.3351593, 95.2714310
36: -64.6885223, 34.9187775, -64.7461853, 35.0648575, -99.4353790, 99.3344193
37: -100.8805008, 21.9530449, -100.9583206, 22.1025162, -121.8446350, 121.7626190
38: -86.0912170, 33.3682137, -86.1672821, 33.5325165, -119.6237335, 119.5354919
39: -104.0472565, 26.6565857, -104.1557465, 26.7613907, -130.3718109, 130.3716888
40: -91.4348068, 3.1920252, -91.5074005, 3.2664433, -87.7765198, 87.7619781
41: -67.5900879, 22.2503815, -67.6364899, 22.3208675, -86.0844803, 86.0498352
42: -60.5896606, 14.9914055, -60.6318512, 15.0951042, -71.0115051, 70.8683014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=366, inp2_unstable=366, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=334, inp2_unstable=335, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=24, inp2_unstable=24, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 1335
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1373
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1486
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1341
type: A, layer: 1, pos: 987
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 978
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 1490
type: A, layer: 1, pos: 1511
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 906
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1340
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 1637
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1512
type: A, layer: 1, pos: 894
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 1404
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1332
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 1622
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1309
type: A, layer: 1, pos: 1474
type: A, layer: 1, pos: 1478
type: A, layer: 1, pos: 991
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1320
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1513
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1325
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 1348
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 975
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1420
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 1375
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1357
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 792
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 836
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1477
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1336
type: A, layer: 1, pos: 1316
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1458
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1364
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 1412
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1668
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1331
type: A, layer: 1, pos: 1388
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1380
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1307
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 1356
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1445
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1652
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 810
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1381
type: A, layer: 1, pos: 1324
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1355
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1010
type: A, layer: 1, pos: 1281
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 1300
type: A, layer: 1, pos: 796
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1347
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 996
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 1354
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1321
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 802
type: A, layer: 1, pos: 789
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1338
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1737
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1315
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 775
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 1359
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1007
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1441
type: A, layer: 1, pos: 1363
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1510
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 797
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1306
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1683
type: A, layer: 1, pos: 1379
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1284
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 777
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1509
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1009
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1459
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 795
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1396
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1519
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 799

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1591

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 29, lower bound: -45.0266204, upper bound: 44.9642282
time: 62.58 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 29, lower bound: -45.0266204, upper bound: 44.9964294
time: 69.35 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 134.37 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 134.37
Output dim: 29, lower bound: -45.0266204, upper bound: 44.9368815
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 134.37
Output dim: 29, lower bound: -45.0266204, upper bound: 44.9690641
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 134.37
Output dim: 29, lower bound: -45.0266204, upper bound: 44.9642282
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 134.37
Output dim: 29, lower bound: -45.0266204, upper bound: 44.9964294
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=65.85055541992188
rel_dist={29: [-45.07674778233984, 45.07674777663058]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 9034.28 seconds
