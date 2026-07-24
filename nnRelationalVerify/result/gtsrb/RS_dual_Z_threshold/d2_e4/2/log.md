## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 3600 seconds
Split limit: 100
Threshold: 46.7201561769


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=328, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-41.8581047, 32.1681671, -41.8581047, 32.1681671, -74.0262756, 74.0262756)
1: (-27.6238575, 29.8764305, -27.6238575, 29.8764305, -57.5002899, 57.5002899)
2: (-22.0896931, 25.5827484, -22.0896931, 25.5827484, -47.6724396, 47.6724396)
3: (-25.4082108, 31.8256626, -25.4082108, 31.8256626, -57.2338715, 57.2338715)
4: (-29.0485497, 28.7367477, -29.0485497, 28.7367477, -57.7852974, 57.7852974)
5: (-28.2786865, 29.0868778, -28.2786865, 29.0868778, -57.3655624, 57.3655624)
6: (-47.3176498, 14.1689835, -47.3176498, 14.1689835, -61.0804901, 61.0804787)
7: (-38.5263596, 25.7286282, -38.5263596, 25.7286282, -64.2549896, 64.2549896)
8: (-35.1668663, 29.4639416, -35.1668663, 29.4639416, -64.6308060, 64.6308060)
9: (-22.0164375, 30.4925995, -22.0164375, 30.4925995, -52.5090370, 52.5090370)
10: (-41.9452057, 32.8714218, -41.9452057, 32.8714218, -74.8166275, 74.8166275)
11: (-48.6259537, 19.9987221, -48.6259537, 19.9987221, -68.6246796, 68.6246796)
12: (-43.4501343, 25.0467262, -43.4501343, 25.0467262, -68.4968567, 68.4968567)
13: (-30.5095196, 38.7206345, -30.5095196, 38.7206345, -69.2301559, 69.2301559)
14: (-77.4074554, 5.2998600, -77.4074554, 5.2998600, -82.7073135, 82.7073135)
15: (-29.5153503, 36.1136436, -29.5153503, 36.1136436, -65.6289978, 65.6289978)
16: (-46.8135033, 28.1260738, -46.8135033, 28.1260738, -74.9395752, 74.9395752)
17: (-79.2747726, 16.1380520, -79.2747726, 16.1380520, -95.4128265, 95.4128265)
18: (-45.1141205, 18.2228985, -45.1141205, 18.2228985, -63.3370209, 63.3370209)
19: (-36.2994308, 11.3947449, -36.2994308, 11.3947449, -47.6941757, 47.6941757)
20: (-31.0101185, 15.6616783, -31.0101185, 15.6616783, -46.6717987, 46.6717987)
21: (-43.6235428, 14.5215607, -43.6235428, 14.5215607, -58.1451035, 58.1451035)
22: (-37.6735344, 21.3282833, -37.6735344, 21.3282833, -59.0018158, 59.0018158)
23: (-32.3645401, 18.2318859, -32.3645401, 18.2318859, -50.5964279, 50.5964279)
24: (-30.8573360, 16.2691994, -30.8573360, 16.2691994, -47.1265335, 47.1265335)
25: (-26.9438877, 24.1879005, -26.9438877, 24.1879005, -51.1317902, 51.1317902)
26: (-50.0373726, 24.3668594, -50.0373726, 24.3668594, -74.4042358, 74.4042358)
27: (-41.6567307, 12.1853085, -41.6567307, 12.1853085, -53.8420410, 53.8420410)
28: (-32.7484970, 22.9294147, -32.7484970, 22.9294147, -55.6779099, 55.6779099)
29: (-37.8354340, 17.8038578, -37.8354340, 17.8038578, -55.6392899, 55.6392899)
30: (-41.0930443, 21.7399483, -41.0930443, 21.7399483, -62.8329926, 62.8329926)
31: (-41.8162842, 14.3614702, -41.8162842, 14.3614702, -56.1777534, 56.1777534)
32: (-36.9501724, 18.5844498, -36.9501724, 18.5844498, -55.5346222, 55.5346222)
33: (-46.8246155, 30.6463928, -46.8246155, 30.6463928, -77.4710083, 77.4710083)
34: (-44.5230598, 25.8768120, -44.5230598, 25.8768120, -70.3998718, 70.3998718)
35: (-35.8817444, 29.5253220, -35.8817444, 29.5253220, -65.4070663, 65.4070663)
36: (-39.7122803, 26.8040981, -39.7122803, 26.8040981, -66.5163803, 66.5163803)
37: (-60.5529633, 23.0308647, -60.5529633, 23.0308647, -83.5838318, 83.5838318)
38: (-56.7473831, 26.3439636, -56.7473831, 26.3439636, -83.0913467, 83.0913467)
39: (-56.9161797, 21.4197426, -56.9161797, 21.4197426, -78.3359222, 78.3359222)
40: (-58.8069649, 16.5587921, -58.8069649, 16.5587921, -74.8190613, 74.8190613)
41: (-38.5574493, 19.0406857, -38.5574493, 19.0406857, -57.5981369, 57.5981369)
42: (-31.9986248, 16.9168606, -31.9986248, 16.9168606, -48.9154854, 48.9154854)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.84 + 67.39 = 70.23 seconds
status: Status.UNKNOWN
relational distance
Output dim: 13, lower bound: -46.7669231, upper bound: 46.7669231

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1360
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1488
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 962

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1725

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7196339, upper bound: 46.7646324
time: 53.89 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7646324, upper bound: 46.7196339
time: 54.69 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 108.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 108.70
Output dim: 13, lower bound: -46.7196339, upper bound: 46.7646324
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 108.70
Output dim: 13, lower bound: -46.7646324, upper bound: 46.7196339

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -41.8581047, 32.1681671, -41.8581047, 32.1681671, -74.0262756, 74.0262756
1: -27.6238575, 29.8764305, -27.6238575, 29.8764305, -57.5002899, 57.5002899
2: -22.0896931, 25.5827484, -22.0896931, 25.5827484, -47.6724396, 47.6724396
3: -25.4082108, 31.8256626, -25.4082108, 31.8256626, -57.2338715, 57.2338715
4: -29.0485497, 28.7367477, -29.0485497, 28.7367477, -57.7852974, 57.7852974
5: -28.2786865, 29.0868778, -28.2786865, 29.0868778, -57.3655624, 57.3655624
6: -47.3176498, 14.1689835, -47.3176498, 14.1689835, -61.0537567, 61.0573997
7: -38.5263596, 25.7286282, -38.5263596, 25.7286282, -64.2549896, 64.2549896
8: -35.1668663, 29.4639416, -35.1668663, 29.4639416, -64.6308060, 64.6308060
9: -22.0164375, 30.4925995, -22.0164375, 30.4925995, -52.5090370, 52.5090370
10: -41.9452057, 32.8714218, -41.9452057, 32.8714218, -74.8166275, 74.8166275
11: -48.6259537, 19.9987221, -48.6259537, 19.9987221, -68.6246796, 68.6246796
12: -43.4501343, 25.0467262, -43.4501343, 25.0467262, -68.4968567, 68.4968567
13: -30.5095196, 38.7206345, -30.5095196, 38.7206345, -69.2301559, 69.2301559
14: -77.4074554, 5.2998600, -77.4074554, 5.2998600, -82.7073135, 82.7073135
15: -29.5153503, 36.1136436, -29.5153503, 36.1136436, -65.6289978, 65.6289978
16: -46.8135033, 28.1260738, -46.8135033, 28.1260738, -74.9395752, 74.9395752
17: -79.2747726, 16.1380520, -79.2747726, 16.1380520, -95.4128265, 95.4128265
18: -45.1141205, 18.2228985, -45.1141205, 18.2228985, -63.3370209, 63.3370209
19: -36.2994308, 11.3947449, -36.2994308, 11.3947449, -47.6941757, 47.6941757
20: -31.0101185, 15.6616783, -31.0101185, 15.6616783, -46.6717987, 46.6717987
21: -43.6235428, 14.5215607, -43.6235428, 14.5215607, -58.1451035, 58.1451035
22: -37.6735344, 21.3282833, -37.6735344, 21.3282833, -59.0018158, 59.0018158
23: -32.3645401, 18.2318859, -32.3645401, 18.2318859, -50.5964279, 50.5964279
24: -30.8573360, 16.2691994, -30.8573360, 16.2691994, -47.1265335, 47.1265335
25: -26.9438877, 24.1879005, -26.9438877, 24.1879005, -51.1317902, 51.1317902
26: -50.0373726, 24.3668594, -50.0373726, 24.3668594, -74.4042358, 74.4042358
27: -41.6567307, 12.1853085, -41.6567307, 12.1853085, -53.8420410, 53.8420410
28: -32.7484970, 22.9294147, -32.7484970, 22.9294147, -55.6779099, 55.6779099
29: -37.8354340, 17.8038578, -37.8354340, 17.8038578, -55.6392899, 55.6392899
30: -41.0930443, 21.7399483, -41.0930443, 21.7399483, -62.8329926, 62.8329926
31: -41.8162842, 14.3614702, -41.8162842, 14.3614702, -56.1777534, 56.1777534
32: -36.9501724, 18.5844498, -36.9501724, 18.5844498, -55.5346222, 55.5346222
33: -46.8246155, 30.6463928, -46.8246155, 30.6463928, -77.4710083, 77.4710083
34: -44.5230598, 25.8768120, -44.5230598, 25.8768120, -70.3998718, 70.3998718
35: -35.8817444, 29.5253220, -35.8817444, 29.5253220, -65.4070663, 65.4070663
36: -39.7122803, 26.8040981, -39.7122803, 26.8040981, -66.5163803, 66.5163803
37: -60.5529633, 23.0308647, -60.5529633, 23.0308647, -83.5838318, 83.5838318
38: -56.7473831, 26.3439636, -56.7473831, 26.3439636, -83.0913467, 83.0913467
39: -56.9161797, 21.4197426, -56.9161797, 21.4197426, -78.3359222, 78.3359222
40: -58.8069649, 16.5587921, -58.8069649, 16.5587921, -74.7947998, 74.7978058
41: -38.5574493, 19.0406857, -38.5574493, 19.0406857, -57.5975037, 57.5981369
42: -31.9986248, 16.9168606, -31.9986248, 16.9168606, -48.9154854, 48.9154854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=328, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1360
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1488
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 962

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1751

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7178643, upper bound: 46.7391019
time: 57.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.6940921, upper bound: 46.7628576
time: 125.01 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -41.8581047, 32.1681671, -41.8581047, 32.1681671, -74.0262756, 74.0262756
1: -27.6238575, 29.8764305, -27.6238575, 29.8764305, -57.5002899, 57.5002899
2: -22.0896931, 25.5827484, -22.0896931, 25.5827484, -47.6724396, 47.6724396
3: -25.4082108, 31.8256626, -25.4082108, 31.8256626, -57.2338715, 57.2338715
4: -29.0485497, 28.7367477, -29.0485497, 28.7367477, -57.7852974, 57.7852974
5: -28.2786865, 29.0868778, -28.2786865, 29.0868778, -57.3655624, 57.3655624
6: -47.3176498, 14.1689835, -47.3176498, 14.1689835, -61.0574036, 61.0537605
7: -38.5263596, 25.7286282, -38.5263596, 25.7286282, -64.2549896, 64.2549896
8: -35.1668663, 29.4639416, -35.1668663, 29.4639416, -64.6308060, 64.6308060
9: -22.0164375, 30.4925995, -22.0164375, 30.4925995, -52.5090370, 52.5090370
10: -41.9452057, 32.8714218, -41.9452057, 32.8714218, -74.8166275, 74.8166275
11: -48.6259537, 19.9987221, -48.6259537, 19.9987221, -68.6246796, 68.6246796
12: -43.4501343, 25.0467262, -43.4501343, 25.0467262, -68.4968567, 68.4968567
13: -30.5095196, 38.7206345, -30.5095196, 38.7206345, -69.2301559, 69.2301559
14: -77.4074554, 5.2998600, -77.4074554, 5.2998600, -82.7073135, 82.7073135
15: -29.5153503, 36.1136436, -29.5153503, 36.1136436, -65.6289978, 65.6289978
16: -46.8135033, 28.1260738, -46.8135033, 28.1260738, -74.9395752, 74.9395752
17: -79.2747726, 16.1380520, -79.2747726, 16.1380520, -95.4128265, 95.4128265
18: -45.1141205, 18.2228985, -45.1141205, 18.2228985, -63.3370209, 63.3370209
19: -36.2994308, 11.3947449, -36.2994308, 11.3947449, -47.6941757, 47.6941757
20: -31.0101185, 15.6616783, -31.0101185, 15.6616783, -46.6717987, 46.6717987
21: -43.6235428, 14.5215607, -43.6235428, 14.5215607, -58.1451035, 58.1451035
22: -37.6735344, 21.3282833, -37.6735344, 21.3282833, -59.0018158, 59.0018158
23: -32.3645401, 18.2318859, -32.3645401, 18.2318859, -50.5964279, 50.5964279
24: -30.8573360, 16.2691994, -30.8573360, 16.2691994, -47.1265335, 47.1265335
25: -26.9438877, 24.1879005, -26.9438877, 24.1879005, -51.1317902, 51.1317902
26: -50.0373726, 24.3668594, -50.0373726, 24.3668594, -74.4042358, 74.4042358
27: -41.6567307, 12.1853085, -41.6567307, 12.1853085, -53.8420410, 53.8420410
28: -32.7484970, 22.9294147, -32.7484970, 22.9294147, -55.6779099, 55.6779099
29: -37.8354340, 17.8038578, -37.8354340, 17.8038578, -55.6392899, 55.6392899
30: -41.0930443, 21.7399483, -41.0930443, 21.7399483, -62.8329926, 62.8329926
31: -41.8162842, 14.3614702, -41.8162842, 14.3614702, -56.1777534, 56.1777534
32: -36.9501724, 18.5844498, -36.9501724, 18.5844498, -55.5346222, 55.5346222
33: -46.8246155, 30.6463928, -46.8246155, 30.6463928, -77.4710083, 77.4710083
34: -44.5230598, 25.8768120, -44.5230598, 25.8768120, -70.3998718, 70.3998718
35: -35.8817444, 29.5253220, -35.8817444, 29.5253220, -65.4070663, 65.4070663
36: -39.7122803, 26.8040981, -39.7122803, 26.8040981, -66.5163803, 66.5163803
37: -60.5529633, 23.0308647, -60.5529633, 23.0308647, -83.5838318, 83.5838318
38: -56.7473831, 26.3439636, -56.7473831, 26.3439636, -83.0913467, 83.0913467
39: -56.9161797, 21.4197426, -56.9161797, 21.4197426, -78.3359222, 78.3359222
40: -58.8069649, 16.5587921, -58.8069649, 16.5587921, -74.7977905, 74.7948151
41: -38.5574493, 19.0406857, -38.5574493, 19.0406857, -57.5981369, 57.5974998
42: -31.9986248, 16.9168606, -31.9986248, 16.9168606, -48.9154854, 48.9154854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=328, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1751
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1360
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1488
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 962

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1751

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7628576, upper bound: 46.6940921
time: 84.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -46.7391019, upper bound: 46.7178643
time: 57.93 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 144.35 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 144.35
Output dim: 13, lower bound: -46.7178643, upper bound: 46.7391019
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 144.35
Output dim: 13, lower bound: -46.6940921, upper bound: 46.7628576
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 144.35
Output dim: 13, lower bound: -46.7628576, upper bound: 46.6940921
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 144.35
Output dim: 13, lower bound: -46.7391019, upper bound: 46.7178643

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -41.8581047, 32.1681671, -41.8581047, 32.1681671, -74.0262756, 74.0262756
1: -27.6238575, 29.8764305, -27.6238575, 29.8764305, -57.5002899, 57.5002899
2: -22.0896931, 25.5827484, -22.0896931, 25.5827484, -47.6724396, 47.6724396
3: -25.4082108, 31.8256626, -25.4082108, 31.8256626, -57.2338715, 57.2338715
4: -29.0485497, 28.7367477, -29.0485497, 28.7367477, -57.7852974, 57.7852974
5: -28.2786865, 29.0868778, -28.2786865, 29.0868778, -57.3655624, 57.3655624
6: -47.3176498, 14.1689835, -47.3176498, 14.1689835, -61.0559387, 61.0595512
7: -38.5263596, 25.7286282, -38.5263596, 25.7286282, -64.2549896, 64.2549896
8: -35.1668663, 29.4639416, -35.1668663, 29.4639416, -64.6308060, 64.6308060
9: -22.0164375, 30.4925995, -22.0164375, 30.4925995, -52.5090370, 52.5090370
10: -41.9452057, 32.8714218, -41.9452057, 32.8714218, -74.8166275, 74.8166275
11: -48.6259537, 19.9987221, -48.6259537, 19.9987221, -68.6246796, 68.6246796
12: -43.4501343, 25.0467262, -43.4501343, 25.0467262, -68.4968567, 68.4968567
13: -30.5095196, 38.7206345, -30.5095196, 38.7206345, -69.2301559, 69.2301559
14: -77.4074554, 5.2998600, -77.4074554, 5.2998600, -82.7073135, 82.7073135
15: -29.5153503, 36.1136436, -29.5153503, 36.1136436, -65.6289978, 65.6289978
16: -46.8135033, 28.1260738, -46.8135033, 28.1260738, -74.9395752, 74.9395752
17: -79.2747726, 16.1380520, -79.2747726, 16.1380520, -95.4128265, 95.4128265
18: -45.1141205, 18.2228985, -45.1141205, 18.2228985, -63.3370209, 63.3370209
19: -36.2994308, 11.3947449, -36.2994308, 11.3947449, -47.6941757, 47.6941757
20: -31.0101185, 15.6616783, -31.0101185, 15.6616783, -46.6717987, 46.6717987
21: -43.6235428, 14.5215607, -43.6235428, 14.5215607, -58.1451035, 58.1451035
22: -37.6735344, 21.3282833, -37.6735344, 21.3282833, -59.0018158, 59.0018158
23: -32.3645401, 18.2318859, -32.3645401, 18.2318859, -50.5964279, 50.5964279
24: -30.8573360, 16.2691994, -30.8573360, 16.2691994, -47.1265335, 47.1265335
25: -26.9438877, 24.1879005, -26.9438877, 24.1879005, -51.1317902, 51.1317902
26: -50.0373726, 24.3668594, -50.0373726, 24.3668594, -74.4042358, 74.4042358
27: -41.6567307, 12.1853085, -41.6567307, 12.1853085, -53.8420410, 53.8420410
28: -32.7484970, 22.9294147, -32.7484970, 22.9294147, -55.6779099, 55.6779099
29: -37.8354340, 17.8038578, -37.8354340, 17.8038578, -55.6392899, 55.6392899
30: -41.0930443, 21.7399483, -41.0930443, 21.7399483, -62.8329926, 62.8329926
31: -41.8162842, 14.3614702, -41.8162842, 14.3614702, -56.1777534, 56.1777534
32: -36.9501724, 18.5844498, -36.9501724, 18.5844498, -55.5346222, 55.5346222
33: -46.8246155, 30.6463928, -46.8246155, 30.6463928, -77.4710083, 77.4710083
34: -44.5230598, 25.8768120, -44.5230598, 25.8768120, -70.3998718, 70.3998718
35: -35.8817444, 29.5253220, -35.8817444, 29.5253220, -65.4070663, 65.4070663
36: -39.7122803, 26.8040981, -39.7122803, 26.8040981, -66.5163803, 66.5163803
37: -60.5529633, 23.0308647, -60.5529633, 23.0308647, -83.5838318, 83.5838318
38: -56.7473831, 26.3439636, -56.7473831, 26.3439636, -83.0913467, 83.0913467
39: -56.9161797, 21.4197426, -56.9161797, 21.4197426, -78.3359222, 78.3359222
40: -58.8069649, 16.5587921, -58.8069649, 16.5587921, -74.7965698, 74.7995453
41: -38.5574493, 19.0406857, -38.5574493, 19.0406857, -57.5981293, 57.5981369
42: -31.9986248, 16.9168606, -31.9986248, 16.9168606, -48.9154854, 48.9154854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=328, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1360
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1488
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 962

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 665

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6706576, upper bound: 46.6917937
time: 69.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6706576, upper bound: 46.6917937
time: 86.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -41.8581047, 32.1681671, -41.8581047, 32.1681671, -74.0262756, 74.0262756
1: -27.6238575, 29.8764305, -27.6238575, 29.8764305, -57.5002899, 57.5002899
2: -22.0896931, 25.5827484, -22.0896931, 25.5827484, -47.6724396, 47.6724396
3: -25.4082108, 31.8256626, -25.4082108, 31.8256626, -57.2338715, 57.2338715
4: -29.0485497, 28.7367477, -29.0485497, 28.7367477, -57.7852974, 57.7852974
5: -28.2786865, 29.0868778, -28.2786865, 29.0868778, -57.3655624, 57.3655624
6: -47.3176498, 14.1689835, -47.3176498, 14.1689835, -61.0559082, 61.0595779
7: -38.5263596, 25.7286282, -38.5263596, 25.7286282, -64.2549896, 64.2549896
8: -35.1668663, 29.4639416, -35.1668663, 29.4639416, -64.6308060, 64.6308060
9: -22.0164375, 30.4925995, -22.0164375, 30.4925995, -52.5090370, 52.5090370
10: -41.9452057, 32.8714218, -41.9452057, 32.8714218, -74.8166275, 74.8166275
11: -48.6259537, 19.9987221, -48.6259537, 19.9987221, -68.6246796, 68.6246796
12: -43.4501343, 25.0467262, -43.4501343, 25.0467262, -68.4968567, 68.4968567
13: -30.5095196, 38.7206345, -30.5095196, 38.7206345, -69.2301559, 69.2301559
14: -77.4074554, 5.2998600, -77.4074554, 5.2998600, -82.7073135, 82.7073135
15: -29.5153503, 36.1136436, -29.5153503, 36.1136436, -65.6289978, 65.6289978
16: -46.8135033, 28.1260738, -46.8135033, 28.1260738, -74.9395752, 74.9395752
17: -79.2747726, 16.1380520, -79.2747726, 16.1380520, -95.4128265, 95.4128265
18: -45.1141205, 18.2228985, -45.1141205, 18.2228985, -63.3370209, 63.3370209
19: -36.2994308, 11.3947449, -36.2994308, 11.3947449, -47.6941757, 47.6941757
20: -31.0101185, 15.6616783, -31.0101185, 15.6616783, -46.6717987, 46.6717987
21: -43.6235428, 14.5215607, -43.6235428, 14.5215607, -58.1451035, 58.1451035
22: -37.6735344, 21.3282833, -37.6735344, 21.3282833, -59.0018158, 59.0018158
23: -32.3645401, 18.2318859, -32.3645401, 18.2318859, -50.5964279, 50.5964279
24: -30.8573360, 16.2691994, -30.8573360, 16.2691994, -47.1265335, 47.1265335
25: -26.9438877, 24.1879005, -26.9438877, 24.1879005, -51.1317902, 51.1317902
26: -50.0373726, 24.3668594, -50.0373726, 24.3668594, -74.4042358, 74.4042358
27: -41.6567307, 12.1853085, -41.6567307, 12.1853085, -53.8420410, 53.8420410
28: -32.7484970, 22.9294147, -32.7484970, 22.9294147, -55.6779099, 55.6779099
29: -37.8354340, 17.8038578, -37.8354340, 17.8038578, -55.6392899, 55.6392899
30: -41.0930443, 21.7399483, -41.0930443, 21.7399483, -62.8329926, 62.8329926
31: -41.8162842, 14.3614702, -41.8162842, 14.3614702, -56.1777534, 56.1777534
32: -36.9501724, 18.5844498, -36.9501724, 18.5844498, -55.5346222, 55.5346222
33: -46.8246155, 30.6463928, -46.8246155, 30.6463928, -77.4710083, 77.4710083
34: -44.5230598, 25.8768120, -44.5230598, 25.8768120, -70.3998718, 70.3998718
35: -35.8817444, 29.5253220, -35.8817444, 29.5253220, -65.4070663, 65.4070663
36: -39.7122803, 26.8040981, -39.7122803, 26.8040981, -66.5163803, 66.5163803
37: -60.5529633, 23.0308647, -60.5529633, 23.0308647, -83.5838318, 83.5838318
38: -56.7473831, 26.3439636, -56.7473831, 26.3439636, -83.0913467, 83.0913467
39: -56.9161797, 21.4197426, -56.9161797, 21.4197426, -78.3359222, 78.3359222
40: -58.8069649, 16.5587921, -58.8069649, 16.5587921, -74.7965698, 74.7995605
41: -38.5574493, 19.0406857, -38.5574493, 19.0406857, -57.5981293, 57.5981369
42: -31.9986248, 16.9168606, -31.9986248, 16.9168606, -48.9154854, 48.9154854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=328, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1360
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1488
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 962

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 665

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6467374, upper bound: 46.7156811
time: 55.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6467374, upper bound: 46.7156811
time: 56.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -41.8581047, 32.1681671, -41.8581047, 32.1681671, -74.0262756, 74.0262756
1: -27.6238575, 29.8764305, -27.6238575, 29.8764305, -57.5002899, 57.5002899
2: -22.0896931, 25.5827484, -22.0896931, 25.5827484, -47.6724396, 47.6724396
3: -25.4082108, 31.8256626, -25.4082108, 31.8256626, -57.2338715, 57.2338715
4: -29.0485497, 28.7367477, -29.0485497, 28.7367477, -57.7852974, 57.7852974
5: -28.2786865, 29.0868778, -28.2786865, 29.0868778, -57.3655624, 57.3655624
6: -47.3176498, 14.1689835, -47.3176498, 14.1689835, -61.0595703, 61.0559120
7: -38.5263596, 25.7286282, -38.5263596, 25.7286282, -64.2549896, 64.2549896
8: -35.1668663, 29.4639416, -35.1668663, 29.4639416, -64.6308060, 64.6308060
9: -22.0164375, 30.4925995, -22.0164375, 30.4925995, -52.5090370, 52.5090370
10: -41.9452057, 32.8714218, -41.9452057, 32.8714218, -74.8166275, 74.8166275
11: -48.6259537, 19.9987221, -48.6259537, 19.9987221, -68.6246796, 68.6246796
12: -43.4501343, 25.0467262, -43.4501343, 25.0467262, -68.4968567, 68.4968567
13: -30.5095196, 38.7206345, -30.5095196, 38.7206345, -69.2301559, 69.2301559
14: -77.4074554, 5.2998600, -77.4074554, 5.2998600, -82.7073135, 82.7073135
15: -29.5153503, 36.1136436, -29.5153503, 36.1136436, -65.6289978, 65.6289978
16: -46.8135033, 28.1260738, -46.8135033, 28.1260738, -74.9395752, 74.9395752
17: -79.2747726, 16.1380520, -79.2747726, 16.1380520, -95.4128265, 95.4128265
18: -45.1141205, 18.2228985, -45.1141205, 18.2228985, -63.3370209, 63.3370209
19: -36.2994308, 11.3947449, -36.2994308, 11.3947449, -47.6941757, 47.6941757
20: -31.0101185, 15.6616783, -31.0101185, 15.6616783, -46.6717987, 46.6717987
21: -43.6235428, 14.5215607, -43.6235428, 14.5215607, -58.1451035, 58.1451035
22: -37.6735344, 21.3282833, -37.6735344, 21.3282833, -59.0018158, 59.0018158
23: -32.3645401, 18.2318859, -32.3645401, 18.2318859, -50.5964279, 50.5964279
24: -30.8573360, 16.2691994, -30.8573360, 16.2691994, -47.1265335, 47.1265335
25: -26.9438877, 24.1879005, -26.9438877, 24.1879005, -51.1317902, 51.1317902
26: -50.0373726, 24.3668594, -50.0373726, 24.3668594, -74.4042358, 74.4042358
27: -41.6567307, 12.1853085, -41.6567307, 12.1853085, -53.8420410, 53.8420410
28: -32.7484970, 22.9294147, -32.7484970, 22.9294147, -55.6779099, 55.6779099
29: -37.8354340, 17.8038578, -37.8354340, 17.8038578, -55.6392899, 55.6392899
30: -41.0930443, 21.7399483, -41.0930443, 21.7399483, -62.8329926, 62.8329926
31: -41.8162842, 14.3614702, -41.8162842, 14.3614702, -56.1777534, 56.1777534
32: -36.9501724, 18.5844498, -36.9501724, 18.5844498, -55.5346222, 55.5346222
33: -46.8246155, 30.6463928, -46.8246155, 30.6463928, -77.4710083, 77.4710083
34: -44.5230598, 25.8768120, -44.5230598, 25.8768120, -70.3998718, 70.3998718
35: -35.8817444, 29.5253220, -35.8817444, 29.5253220, -65.4070663, 65.4070663
36: -39.7122803, 26.8040981, -39.7122803, 26.8040981, -66.5163803, 66.5163803
37: -60.5529633, 23.0308647, -60.5529633, 23.0308647, -83.5838318, 83.5838318
38: -56.7473831, 26.3439636, -56.7473831, 26.3439636, -83.0913467, 83.0913467
39: -56.9161797, 21.4197426, -56.9161797, 21.4197426, -78.3359222, 78.3359222
40: -58.8069649, 16.5587921, -58.8069649, 16.5587921, -74.7995605, 74.7965546
41: -38.5574493, 19.0406857, -38.5574493, 19.0406857, -57.5981369, 57.5981216
42: -31.9986248, 16.9168606, -31.9986248, 16.9168606, -48.9154854, 48.9154854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=328, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1360
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1488
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 962

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 665

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.7156812, upper bound: 46.6467373
time: 67.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.7156812, upper bound: 46.6467373
time: 66.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -41.8581047, 32.1681671, -41.8581047, 32.1681671, -74.0262756, 74.0262756
1: -27.6238575, 29.8764305, -27.6238575, 29.8764305, -57.5002899, 57.5002899
2: -22.0896931, 25.5827484, -22.0896931, 25.5827484, -47.6724396, 47.6724396
3: -25.4082108, 31.8256626, -25.4082108, 31.8256626, -57.2338715, 57.2338715
4: -29.0485497, 28.7367477, -29.0485497, 28.7367477, -57.7852974, 57.7852974
5: -28.2786865, 29.0868778, -28.2786865, 29.0868778, -57.3655624, 57.3655624
6: -47.3176498, 14.1689835, -47.3176498, 14.1689835, -61.0595551, 61.0559387
7: -38.5263596, 25.7286282, -38.5263596, 25.7286282, -64.2549896, 64.2549896
8: -35.1668663, 29.4639416, -35.1668663, 29.4639416, -64.6308060, 64.6308060
9: -22.0164375, 30.4925995, -22.0164375, 30.4925995, -52.5090370, 52.5090370
10: -41.9452057, 32.8714218, -41.9452057, 32.8714218, -74.8166275, 74.8166275
11: -48.6259537, 19.9987221, -48.6259537, 19.9987221, -68.6246796, 68.6246796
12: -43.4501343, 25.0467262, -43.4501343, 25.0467262, -68.4968567, 68.4968567
13: -30.5095196, 38.7206345, -30.5095196, 38.7206345, -69.2301559, 69.2301559
14: -77.4074554, 5.2998600, -77.4074554, 5.2998600, -82.7073135, 82.7073135
15: -29.5153503, 36.1136436, -29.5153503, 36.1136436, -65.6289978, 65.6289978
16: -46.8135033, 28.1260738, -46.8135033, 28.1260738, -74.9395752, 74.9395752
17: -79.2747726, 16.1380520, -79.2747726, 16.1380520, -95.4128265, 95.4128265
18: -45.1141205, 18.2228985, -45.1141205, 18.2228985, -63.3370209, 63.3370209
19: -36.2994308, 11.3947449, -36.2994308, 11.3947449, -47.6941757, 47.6941757
20: -31.0101185, 15.6616783, -31.0101185, 15.6616783, -46.6717987, 46.6717987
21: -43.6235428, 14.5215607, -43.6235428, 14.5215607, -58.1451035, 58.1451035
22: -37.6735344, 21.3282833, -37.6735344, 21.3282833, -59.0018158, 59.0018158
23: -32.3645401, 18.2318859, -32.3645401, 18.2318859, -50.5964279, 50.5964279
24: -30.8573360, 16.2691994, -30.8573360, 16.2691994, -47.1265335, 47.1265335
25: -26.9438877, 24.1879005, -26.9438877, 24.1879005, -51.1317902, 51.1317902
26: -50.0373726, 24.3668594, -50.0373726, 24.3668594, -74.4042358, 74.4042358
27: -41.6567307, 12.1853085, -41.6567307, 12.1853085, -53.8420410, 53.8420410
28: -32.7484970, 22.9294147, -32.7484970, 22.9294147, -55.6779099, 55.6779099
29: -37.8354340, 17.8038578, -37.8354340, 17.8038578, -55.6392899, 55.6392899
30: -41.0930443, 21.7399483, -41.0930443, 21.7399483, -62.8329926, 62.8329926
31: -41.8162842, 14.3614702, -41.8162842, 14.3614702, -56.1777534, 56.1777534
32: -36.9501724, 18.5844498, -36.9501724, 18.5844498, -55.5346222, 55.5346222
33: -46.8246155, 30.6463928, -46.8246155, 30.6463928, -77.4710083, 77.4710083
34: -44.5230598, 25.8768120, -44.5230598, 25.8768120, -70.3998718, 70.3998718
35: -35.8817444, 29.5253220, -35.8817444, 29.5253220, -65.4070663, 65.4070663
36: -39.7122803, 26.8040981, -39.7122803, 26.8040981, -66.5163803, 66.5163803
37: -60.5529633, 23.0308647, -60.5529633, 23.0308647, -83.5838318, 83.5838318
38: -56.7473831, 26.3439636, -56.7473831, 26.3439636, -83.0913467, 83.0913467
39: -56.9161797, 21.4197426, -56.9161797, 21.4197426, -78.3359222, 78.3359222
40: -58.8069649, 16.5587921, -58.8069649, 16.5587921, -74.7995300, 74.7965698
41: -38.5574493, 19.0406857, -38.5574493, 19.0406857, -57.5981369, 57.5981293
42: -31.9986248, 16.9168606, -31.9986248, 16.9168606, -48.9154854, 48.9154854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=328, inp2_unstable=328, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=359, inp2_unstable=359, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=26, inp2_unstable=26, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1752
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1661
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 633
type: RSZ, layer: 1, pos: 1756
type: RSZ, layer: 1, pos: 1768
type: RSZ, layer: 1, pos: 1767
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1724
type: RSZ, layer: 1, pos: 632
type: RSZ, layer: 1, pos: 1769
type: RSZ, layer: 1, pos: 1754
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1721
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1645
type: RSZ, layer: 1, pos: 725
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 631
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 845
type: RSZ, layer: 1, pos: 693
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 861
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1740
type: RSZ, layer: 1, pos: 601
type: RSZ, layer: 1, pos: 1735
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 829
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 585
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 726
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 599
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 584
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1770
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 828
type: RSZ, layer: 1, pos: 744
type: RSZ, layer: 1, pos: 1738
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1369
type: RSZ, layer: 1, pos: 597
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1781
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1399
type: RSZ, layer: 1, pos: 583
type: RSZ, layer: 1, pos: 878
type: RSZ, layer: 1, pos: 617
type: RSZ, layer: 1, pos: 1789
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 634
type: RSZ, layer: 1, pos: 603
type: RSZ, layer: 1, pos: 600
type: RSZ, layer: 1, pos: 1367
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1469
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1406
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 709
type: RSZ, layer: 1, pos: 613
type: RSZ, layer: 1, pos: 1636
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 615
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 602
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1723
type: RSZ, layer: 1, pos: 1739
type: RSZ, layer: 1, pos: 696
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 586
type: RSZ, layer: 1, pos: 616
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1676
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 827
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1390
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1610
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1674
type: RSZ, layer: 1, pos: 1667
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1499
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 695
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 598
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1501
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 662
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 650
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 826
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1411
type: RSZ, layer: 1, pos: 742
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 818
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1766
type: RSZ, layer: 1, pos: 895
type: RSZ, layer: 1, pos: 1383
type: RSZ, layer: 1, pos: 727
type: RSZ, layer: 1, pos: 1453
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 614
type: RSZ, layer: 1, pos: 743
type: RSZ, layer: 1, pos: 1782
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1483
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1391
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1500
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 908
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 923
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1395
type: RSZ, layer: 1, pos: 1525
type: RSZ, layer: 1, pos: 1439
type: RSZ, layer: 1, pos: 1305
type: RSZ, layer: 1, pos: 1502
type: RSZ, layer: 1, pos: 1449
type: RSZ, layer: 1, pos: 1750
type: RSZ, layer: 1, pos: 781
type: RSZ, layer: 1, pos: 865
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1378
type: RSZ, layer: 1, pos: 1011
type: RSZ, layer: 1, pos: 1516
type: RSZ, layer: 1, pos: 1508
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 881
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1524
type: RSZ, layer: 1, pos: 995
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 798
type: RSZ, layer: 1, pos: 1733
type: RSZ, layer: 1, pos: 1480
type: RSZ, layer: 1, pos: 1523
type: RSZ, layer: 1, pos: 1476
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 963
type: RSZ, layer: 1, pos: 1297
type: RSZ, layer: 1, pos: 1643
type: RSZ, layer: 1, pos: 1507
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1371
type: RSZ, layer: 1, pos: 1329
type: RSZ, layer: 1, pos: 924
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 946
type: RSZ, layer: 1, pos: 1377
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 1532
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1534
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1518
type: RSZ, layer: 1, pos: 653
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1432
type: RSZ, layer: 1, pos: 1361
type: RSZ, layer: 1, pos: 1460
type: RSZ, layer: 1, pos: 1319
type: RSZ, layer: 1, pos: 1484
type: RSZ, layer: 1, pos: 1304
type: RSZ, layer: 1, pos: 1749
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 954
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 799
type: RSZ, layer: 1, pos: 1533
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1468
type: RSZ, layer: 1, pos: 1345
type: RSZ, layer: 1, pos: 1517
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 994
type: RSZ, layer: 1, pos: 782
type: RSZ, layer: 1, pos: 1444
type: RSZ, layer: 1, pos: 618
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1506
type: RSZ, layer: 1, pos: 768
type: RSZ, layer: 1, pos: 1423
type: RSZ, layer: 1, pos: 1298
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 801
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 1485
type: RSZ, layer: 1, pos: 1299
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 774
type: RSZ, layer: 1, pos: 993
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 832
type: RSZ, layer: 1, pos: 1360
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1428
type: RSZ, layer: 1, pos: 1008
type: RSZ, layer: 1, pos: 773
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1313
type: RSZ, layer: 1, pos: 1505
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1282
type: RSZ, layer: 1, pos: 776
type: RSZ, layer: 1, pos: 848
type: RSZ, layer: 1, pos: 1288
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1452
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 800
type: RSZ, layer: 1, pos: 1290
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1488
type: RSZ, layer: 1, pos: 896
type: RSZ, layer: 1, pos: 939
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 864
type: RSZ, layer: 1, pos: 1492
type: RSZ, layer: 1, pos: 1287
type: RSZ, layer: 1, pos: 961
type: RSZ, layer: 1, pos: 962

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 665

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6917937, upper bound: 46.6706576
time: 62.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 13, lower bound: -46.6917937, upper bound: 46.6706576
time: 61.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 126.48 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 126.48
Output dim: 13, lower bound: -46.6706576, upper bound: 46.6917937
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 126.48
Output dim: 13, lower bound: -46.6706576, upper bound: 46.6917937
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 126.48
Output dim: 13, lower bound: -46.6467374, upper bound: 46.7156811
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 126.48
Output dim: 13, lower bound: -46.6467374, upper bound: 46.7156811
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 126.48
Output dim: 13, lower bound: -46.7156812, upper bound: 46.6467373
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 126.48
Output dim: 13, lower bound: -46.7156812, upper bound: 46.6467373
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 126.48
Output dim: 13, lower bound: -46.6917937, upper bound: 46.6706576
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 126.48
Output dim: 13, lower bound: -46.6917937, upper bound: 46.6706576

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 70.23 + 973.95 = 1044.18 seconds
