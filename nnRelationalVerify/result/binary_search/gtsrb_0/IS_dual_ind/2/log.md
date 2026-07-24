## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 18000 seconds
Threshold: 94.1958136395
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-79.9393311, 61.1772537, -79.9393311, 61.1772537, -141.1165771, 141.1165771)
1: (-50.0317688, 51.1016350, -50.0317688, 51.1016350, -101.1334000, 101.1334000)
2: (-42.4567261, 46.6027298, -42.4567261, 46.6027298, -89.0594559, 89.0594559)
3: (-47.3553696, 57.8598480, -47.3553696, 57.8598480, -105.2152176, 105.2152176)
4: (-51.8037834, 52.2581177, -51.8037834, 52.2581177, -104.0619049, 104.0619049)
5: (-50.8114319, 58.5132256, -50.8114319, 58.5132256, -109.3246613, 109.3246460)
6: (-74.8497467, 38.0356293, -74.8497467, 38.0356293, -112.8853760, 112.8853760)
7: (-65.0037994, 54.7058144, -65.0037994, 54.7058144, -119.7096100, 119.7096100)
8: (-60.1765327, 58.8050842, -60.1765327, 58.8050842, -118.9816132, 118.9816132)
9: (-46.0628548, 49.9744415, -46.0628548, 49.9744415, -96.0372925, 96.0372925)
10: (-75.2597351, 63.5731163, -75.2597351, 63.5731163, -138.8328552, 138.8328552)
11: (-78.1371918, 46.0559273, -78.1371918, 46.0559273, -124.1931152, 124.1931152)
12: (-79.5543671, 54.1946144, -79.5543671, 54.1946144, -133.7489777, 133.7489777)
13: (-62.1063805, 77.8704605, -62.1063805, 77.8704605, -139.9768372, 139.9768372)
14: (-119.2853241, 37.9927673, -119.2853241, 37.9927673, -157.2780914, 157.2780914)
15: (-58.1623039, 61.8424454, -58.1623039, 61.8424454, -120.0047455, 120.0047455)
16: (-85.5287781, 58.0552330, -85.5287781, 58.0552330, -143.5839996, 143.5840149)
17: (-119.7697144, 53.9870377, -119.7697144, 53.9870377, -173.7567444, 173.7567444)
18: (-75.0466919, 43.4233780, -75.0466919, 43.4233780, -118.4700699, 118.4700699)
19: (-59.4997101, 27.6425838, -59.4997101, 27.6425838, -87.1422958, 87.1422882)
20: (-51.0560646, 35.2854233, -51.0560646, 35.2854233, -86.3414917, 86.3414917)
21: (-71.6610260, 35.8609695, -71.6610260, 35.8609695, -107.5219803, 107.5219955)
22: (-68.7558136, 41.7444153, -68.7558136, 41.7444153, -110.5002289, 110.5002289)
23: (-55.1923904, 38.4922180, -55.1923904, 38.4922180, -93.6846085, 93.6845932)
24: (-58.7264252, 32.9214249, -58.7264252, 32.9214249, -91.6478424, 91.6478424)
25: (-51.0072670, 44.8982887, -51.0072670, 44.8982887, -95.9055557, 95.9055557)
26: (-86.4337616, 55.4519234, -86.4337616, 55.4519234, -141.8856659, 141.8856812)
27: (-69.3014603, 32.3417053, -69.3014603, 32.3417053, -101.6431656, 101.6431656)
28: (-55.4207039, 44.7213211, -55.4207039, 44.7213211, -100.1420288, 100.1420288)
29: (-70.0397873, 39.1376953, -70.0397873, 39.1376953, -109.1774673, 109.1774750)
30: (-69.2020721, 47.9162865, -69.2020721, 47.9162865, -117.1183472, 117.1183472)
31: (-70.8299103, 33.7554092, -70.8299103, 33.7554092, -104.5853119, 104.5853195)
32: (-65.2864304, 42.6033783, -65.2864304, 42.6033783, -107.8898087, 107.8898087)
33: (-84.8707886, 64.0144119, -84.8707886, 64.0144119, -148.8851929, 148.8851929)
34: (-76.2947235, 53.9513550, -76.2947235, 53.9513550, -130.2460632, 130.2460632)
35: (-67.5015640, 58.4963226, -67.5015640, 58.4963226, -125.9978867, 125.9978867)
36: (-70.7597961, 58.9740944, -70.7597961, 58.9740944, -129.7338867, 129.7338867)
37: (-105.3262100, 49.0449448, -105.3262100, 49.0449448, -154.3711548, 154.3711548)
38: (-97.7891922, 69.7007141, -97.7891922, 69.7007141, -167.4898987, 167.4898987)
39: (-104.8183823, 58.6950645, -104.8183823, 58.6950645, -163.5134277, 163.5134430)
40: (-93.5728149, 39.9540825, -93.5728149, 39.9540825, -133.5269012, 133.5269012)
41: (-67.4933853, 40.4102631, -67.4933853, 40.4102631, -107.9036407, 107.9036484)
42: (-53.2005730, 38.0389557, -53.2005730, 38.0389557, -91.2395325, 91.2395172)

## BASE Result
execution time: IAR + LP analysis = 2.94 + 94.17 = 97.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 13, lower bound: -105.2312717, upper bound: 105.2312717


# Binary Search by BASE starts (time budget: 17902.89 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=139.97683715820312
rel_dist={13: [-98.90839803301242, 98.90839802916994]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=139.97683715820312
rel_dist={13: [-94.2982147551906, 94.2982147548251]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=139.97683715820312
rel_dist={13: [-89.59832754882365, 89.59832755]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=139.97683715820312
rel_dist={13: [-92.17109677126697, 92.17109677124596]}

## Binary Search Result
Binary search time: 417.91 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 17484.98 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 966

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 1725

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -100.0575817, upper bound: 100.1499014
time: 91.08 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -100.1499013, upper bound: 100.1499014
time: 102.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 193.63 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 193.63
Output dim: 13, lower bound: -100.0575817, upper bound: 100.1499014
IS_A2, status: Status.UNKNOWN, split count: 1, time: 193.63
Output dim: 13, lower bound: -100.1499013, upper bound: 100.1499014

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -79.7487793, 61.0640106, -79.8997269, 61.1658821, -140.9146576, 140.9637299
1: -49.9181595, 51.0219803, -50.0083618, 51.0910759, -101.0092316, 101.0303421
2: -42.3431664, 46.4915161, -42.4296417, 46.5923004, -88.9354553, 88.9211578
3: -47.1949654, 57.6961174, -47.3164902, 57.8477783, -105.0427246, 105.0126038
4: -51.5967598, 52.1070290, -51.7534370, 52.2486992, -103.8454590, 103.8604584
5: -50.6425476, 58.3190613, -50.7693825, 58.4971886, -109.1397400, 109.0884323
6: -74.7151260, 37.8987808, -74.8246918, 38.0185699, -112.7336960, 112.7234726
7: -64.8565826, 54.5657806, -64.9716492, 54.6912270, -119.5478058, 119.5374298
8: -59.9942093, 58.6274529, -60.1344833, 58.7905502, -118.7847595, 118.7619324
9: -45.9228210, 49.9003677, -46.0347672, 49.9622116, -95.8850250, 95.9351349
10: -75.1153793, 63.4623795, -75.2360916, 63.5542183, -138.6696014, 138.6984711
11: -77.9448547, 45.8540573, -78.1184387, 46.0079994, -123.9528503, 123.9724960
12: -79.3902588, 54.0684700, -79.5296783, 54.1717987, -133.5620575, 133.5981445
13: -61.8165207, 77.5958405, -62.0321503, 77.8567657, -139.6732788, 139.6279907
14: -119.0646591, 37.8752098, -119.2581711, 37.9679909, -157.0326385, 157.1333771
15: -58.0404930, 61.7140465, -58.1402855, 61.8302231, -119.8707123, 119.8543320
16: -85.3749237, 57.9256363, -85.5031204, 58.0259247, -143.4008484, 143.4287567
17: -119.5854034, 53.8654747, -119.7451172, 53.9711113, -173.5565186, 173.6105804
18: -74.8276825, 43.2151604, -75.0320435, 43.3718033, -118.1994629, 118.2472076
19: -59.3319778, 27.5093632, -59.4881859, 27.6093979, -86.9413757, 86.9975433
20: -50.9427338, 35.1749802, -51.0454674, 35.2586136, -86.2013397, 86.2204437
21: -71.4757538, 35.6804161, -71.6473846, 35.8155136, -107.2912598, 107.3277893
22: -68.5911102, 41.6195793, -68.7393494, 41.7138290, -110.3049393, 110.3589325
23: -55.0032005, 38.2653656, -55.1812630, 38.4351959, -93.4384003, 93.4466248
24: -58.5453110, 32.7553902, -58.7121162, 32.8793716, -91.4246826, 91.4675064
25: -50.8489380, 44.7137146, -50.9942055, 44.8539505, -95.7028809, 95.7079163
26: -86.1758575, 55.2631073, -86.4127350, 55.4068375, -141.5827026, 141.6758423
27: -69.0784912, 32.1684456, -69.2879410, 32.2975769, -101.3760681, 101.4563904
28: -55.2288132, 44.5273552, -55.4106331, 44.6725845, -99.9013977, 99.9379730
29: -69.8836975, 38.9489441, -70.0240555, 39.0904465, -108.9741440, 108.9729996
30: -69.0528107, 47.7474632, -69.1891098, 47.8768272, -116.9296265, 116.9365692
31: -70.6199036, 33.6073875, -70.8163910, 33.7210464, -104.3409500, 104.4237671
32: -65.1272736, 42.4701309, -65.2564240, 42.5878868, -107.7151566, 107.7265549
33: -84.5923157, 63.8433380, -84.8089752, 63.9971352, -148.5894470, 148.6523132
34: -76.0970917, 53.8452415, -76.2534485, 53.9425621, -130.0396576, 130.0986786
35: -67.2969971, 58.3604050, -67.4562988, 58.4840584, -125.7810516, 125.8166962
36: -70.5548553, 58.8514099, -70.7160950, 58.9644051, -129.5192566, 129.5675049
37: -105.0800171, 48.9347000, -105.2761459, 49.0312309, -154.1112366, 154.2108459
38: -97.5279846, 69.5440674, -97.7360306, 69.6889648, -167.2169495, 167.2800903
39: -104.4244690, 58.5167274, -104.7261047, 58.6870461, -163.1115112, 163.2428131
40: -93.3119507, 39.8368034, -93.5187149, 39.9443130, -133.2562561, 133.3555145
41: -67.3457642, 40.3328094, -67.4657669, 40.3976593, -107.7434235, 107.7985764
42: -53.0912361, 37.9546967, -53.1848946, 38.0246811, -91.1159134, 91.1395874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=505, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 599

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9946830, upper bound: 100.1122776
time: 88.84 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -100.0121849, upper bound: 100.1048350
time: 78.10 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -79.9259949, 61.1727066, -79.9358292, 61.1761742, -141.1021729, 141.1085205
1: -50.0214767, 51.0972214, -50.0291519, 51.1005630, -101.1220398, 101.1263733
2: -42.4468842, 46.5979233, -42.4541969, 46.6015625, -89.0484467, 89.0521164
3: -47.3457489, 57.8539467, -47.3529663, 57.8584213, -105.2041702, 105.2069092
4: -51.7908325, 52.2536888, -51.8004990, 52.2570496, -104.0478821, 104.0541840
5: -50.8004227, 58.5069427, -50.8087196, 58.5117035, -109.3121262, 109.3156509
6: -74.8402176, 38.0292740, -74.8474350, 38.0340195, -112.8742218, 112.8767090
7: -64.9918976, 54.6998787, -65.0008392, 54.7043724, -119.6962662, 119.7007141
8: -60.1644554, 58.7993279, -60.1733742, 58.8036880, -118.9681320, 118.9726791
9: -46.0514870, 49.9684982, -46.0599976, 49.9729843, -96.0244675, 96.0284958
10: -75.2520905, 63.5637894, -75.2577667, 63.5708656, -138.8229523, 138.8215637
11: -78.1283188, 46.0436478, -78.1350250, 46.0529671, -124.1812897, 124.1786728
12: -79.5462112, 54.1870537, -79.5523987, 54.1926842, -133.7388916, 133.7394409
13: -62.0870476, 77.8647766, -62.1017113, 77.8690186, -139.9560547, 139.9664917
14: -119.2738800, 37.9789505, -119.2824631, 37.9894714, -157.2633514, 157.2614136
15: -58.1551170, 61.8371696, -58.1604080, 61.8411446, -119.9962463, 119.9975739
16: -85.5190735, 58.0420189, -85.5263367, 58.0519676, -143.5710449, 143.5683441
17: -119.7585602, 53.9816208, -119.7668686, 53.9856873, -173.7442322, 173.7484894
18: -75.0399017, 43.4098816, -75.0450668, 43.4201164, -118.4600067, 118.4549484
19: -59.4921799, 27.6346169, -59.4978561, 27.6406517, -87.1328278, 87.1324692
20: -51.0509300, 35.2731628, -51.0548096, 35.2822647, -86.3331909, 86.3279724
21: -71.6523132, 35.8493690, -71.6588898, 35.8581696, -107.5104828, 107.5082550
22: -68.7488556, 41.7358322, -68.7540741, 41.7423248, -110.4911804, 110.4899063
23: -55.1863670, 38.4782410, -55.1909103, 38.4888496, -93.6752167, 93.6691513
24: -58.7179298, 32.9108086, -58.7243576, 32.9188385, -91.6367493, 91.6351624
25: -50.9988518, 44.8870049, -51.0052452, 44.8955307, -95.8943787, 95.8922501
26: -86.4245071, 55.4399338, -86.4314880, 55.4490051, -141.8735046, 141.8714294
27: -69.2953949, 32.3295975, -69.2999496, 32.3388062, -101.6342010, 101.6295471
28: -55.4152260, 44.7092667, -55.4193649, 44.7184143, -100.1336365, 100.1286163
29: -70.0324402, 39.1259079, -70.0379257, 39.1348343, -109.1672745, 109.1638336
30: -69.1942291, 47.9057159, -69.2001648, 47.9136848, -117.1079102, 117.1058655
31: -70.8217773, 33.7461510, -70.8279419, 33.7526665, -104.5744476, 104.5740967
32: -65.2768860, 42.5949936, -65.2840424, 42.6013222, -107.8782043, 107.8790283
33: -84.8492432, 64.0083923, -84.8655777, 64.0128708, -148.8621216, 148.8739624
34: -76.2750244, 53.9450912, -76.2897491, 53.9498367, -130.2248535, 130.2348328
35: -67.4815979, 58.4920540, -67.4964600, 58.4952545, -125.9768524, 125.9885101
36: -70.7426910, 58.9697609, -70.7556152, 58.9730225, -129.7157135, 129.7253723
37: -105.3041153, 49.0407448, -105.3205032, 49.0438766, -154.3479919, 154.3612518
38: -97.7700119, 69.6954193, -97.7845078, 69.6993866, -167.4693909, 167.4799194
39: -104.7931595, 58.6915627, -104.8122101, 58.6941986, -163.4873505, 163.5037537
40: -93.5573578, 39.9509087, -93.5690536, 39.9532547, -133.5106201, 133.5199585
41: -67.4837570, 40.4042625, -67.4910202, 40.4087791, -107.8925323, 107.8952789
42: -53.1949043, 38.0324821, -53.1991692, 38.0373535, -91.2322464, 91.2316513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=505, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 966

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 599

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9946830, upper bound: 100.1122776
time: 110.37 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -100.1048348, upper bound: 100.1048350
time: 93.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 206.43 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 206.43
Output dim: 13, lower bound: -99.9946830, upper bound: 100.1122776
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 206.43
Output dim: 13, lower bound: -100.0121849, upper bound: 100.1048350
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 206.43
Output dim: 13, lower bound: -99.9946830, upper bound: 100.1122776
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 206.43
Output dim: 13, lower bound: -100.1048348, upper bound: 100.1048350

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -79.7416229, 61.0617485, -79.8124619, 61.1377182, -140.8793335, 140.8742065
1: -49.9147797, 51.0201263, -49.9673882, 51.0681992, -100.9829712, 100.9875183
2: -42.3404617, 46.4892616, -42.3961029, 46.5645370, -88.9049988, 88.8853607
3: -47.1906090, 57.6926003, -47.2621803, 57.8042641, -104.9948730, 104.9547806
4: -51.5946960, 52.1018600, -51.7283325, 52.1871643, -103.7818451, 103.8301926
5: -50.6397057, 58.3153152, -50.7347679, 58.4510536, -109.0907593, 109.0500717
6: -74.7108688, 37.8894196, -74.7721786, 37.9021378, -112.6129913, 112.6615906
7: -64.8522110, 54.5635147, -64.9177399, 54.6631317, -119.5153351, 119.4812469
8: -59.9923935, 58.6230087, -60.1120300, 58.7357063, -118.7281036, 118.7350388
9: -45.9203300, 49.8974876, -46.0042000, 49.9268570, -95.8471832, 95.9016876
10: -75.1126480, 63.4583817, -75.2025986, 63.5048180, -138.6174622, 138.6609802
11: -77.9329529, 45.8518295, -77.9712601, 45.9806671, -123.9135971, 123.8230896
12: -79.3868256, 54.0619736, -79.4871597, 54.0912704, -133.4780884, 133.5491333
13: -61.8137321, 77.5875549, -61.9987144, 77.7597351, -139.5734558, 139.5862732
14: -119.0518646, 37.8733139, -119.0996704, 37.9451370, -156.9969940, 156.9729767
15: -58.0379143, 61.7104492, -58.1080017, 61.7858047, -119.8237152, 119.8184433
16: -85.3674164, 57.9224319, -85.4105682, 57.9865417, -143.3539581, 143.3330078
17: -119.5681686, 53.8625946, -119.5327835, 53.9358826, -173.5040436, 173.3953705
18: -74.8237915, 43.2128830, -74.9837646, 43.3439178, -118.1677017, 118.1966476
19: -59.3210678, 27.5078373, -59.3533669, 27.5905476, -86.9116135, 86.8612061
20: -50.9389153, 35.1727676, -50.9978790, 35.2313232, -86.1702347, 86.1706467
21: -71.4651337, 35.6784782, -71.5153656, 35.7916603, -107.2567749, 107.1938477
22: -68.5807037, 41.6178055, -68.6109009, 41.6918373, -110.2725372, 110.2286987
23: -54.9926414, 38.2638016, -55.0513535, 38.4161606, -93.4087982, 93.3151550
24: -58.5331764, 32.7539368, -58.5613251, 32.8613739, -91.3945465, 91.3152618
25: -50.8400345, 44.7115631, -50.8839951, 44.8275986, -95.6676331, 95.5955582
26: -86.1708984, 55.2594604, -86.3510513, 55.3627014, -141.5335999, 141.6105042
27: -69.0710068, 32.1670837, -69.1966095, 32.2806778, -101.3516846, 101.3636932
28: -55.2154617, 44.5252647, -55.2462692, 44.6475792, -99.8630371, 99.7715302
29: -69.8674698, 38.9471626, -69.8245392, 39.0685577, -108.9360199, 108.7716980
30: -69.0410004, 47.7452507, -69.0447540, 47.8497810, -116.8907700, 116.7899933
31: -70.6077042, 33.6058350, -70.6652679, 33.7017670, -104.3094635, 104.2711029
32: -65.1235809, 42.4623489, -65.2113800, 42.4908180, -107.6143951, 107.6737289
33: -84.5879517, 63.8389359, -84.7555313, 63.9437218, -148.5316772, 148.5944519
34: -76.0932922, 53.8421555, -76.2065811, 53.9045448, -129.9978333, 130.0487366
35: -67.2929840, 58.3582916, -67.4070282, 58.4579544, -125.7509384, 125.7653046
36: -70.5519409, 58.8441658, -70.6805420, 58.8755112, -129.4274445, 129.5247040
37: -105.0745163, 48.9241486, -105.2082291, 48.9076691, -153.9821777, 154.1323853
38: -97.5242462, 69.5288467, -97.6906052, 69.4998093, -167.0240479, 167.2194519
39: -104.4200516, 58.5062866, -104.6720963, 58.5604248, -162.9804688, 163.1783752
40: -93.3081970, 39.8234253, -93.4728394, 39.7784004, -133.0865936, 133.2962646
41: -67.3416977, 40.3258705, -67.4158936, 40.3111954, -107.6528931, 107.7417603
42: -53.0884705, 37.9490891, -53.1506348, 37.9584923, -91.0469513, 91.0997162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 966

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 601

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9486760, upper bound: 100.0416960
time: 113.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9415487, upper bound: 100.0598547
time: 151.44 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -79.7311325, 61.0595245, -79.9622421, 61.2342186, -140.9653473, 141.0217590
1: -49.9105148, 51.0178680, -50.0405006, 51.1571426, -101.0676575, 101.0583649
2: -42.3333588, 46.4876862, -42.4559555, 46.6798477, -89.0131836, 88.9436417
3: -47.1820908, 57.6902390, -47.3504791, 57.9330635, -105.1151428, 105.0407104
4: -51.5918312, 52.0999336, -51.8016281, 52.3122673, -103.9040833, 103.9015579
5: -50.6368446, 58.3133202, -50.8056602, 58.5754929, -109.2123413, 109.1189804
6: -74.7087555, 37.8896561, -75.0429382, 38.0371017, -112.7458572, 112.9325943
7: -64.8408813, 54.5591125, -64.9749451, 54.7889671, -119.6298447, 119.5340576
8: -59.9813461, 58.6198196, -60.1374207, 58.8728485, -118.8541946, 118.7572250
9: -45.9184151, 49.8954773, -46.1604118, 50.0035782, -95.9219971, 96.0558929
10: -75.1102753, 63.4533195, -75.3346100, 63.6321106, -138.7423859, 138.7879333
11: -77.9202194, 45.8450584, -78.1370544, 46.1986923, -124.1189117, 123.9821091
12: -79.3830872, 54.0424652, -79.7879181, 54.1586609, -133.5417480, 133.8303833
13: -61.8088150, 77.5725021, -62.3438377, 77.8597412, -139.6685486, 139.9163361
14: -119.0473862, 37.8701515, -119.3109512, 38.1911850, -157.2385712, 157.1811066
15: -58.0326767, 61.7056236, -58.1713028, 61.8735161, -119.9061890, 119.8769226
16: -85.3601227, 57.9166946, -85.5804291, 58.1467285, -143.5068512, 143.4971313
17: -119.5686264, 53.8532715, -119.8016205, 54.2936172, -173.8622437, 173.6548767
18: -74.8163300, 43.2113342, -75.0833893, 43.5469208, -118.3632507, 118.2947235
19: -59.3234901, 27.5034847, -59.5247612, 27.7718010, -87.0952911, 87.0282440
20: -50.9327774, 35.1704254, -51.0667343, 35.3919563, -86.3247299, 86.2371521
21: -71.4557419, 35.6733589, -71.6780853, 36.0063400, -107.4620819, 107.3514404
22: -68.5782318, 41.6135178, -68.7842102, 41.9005394, -110.4787750, 110.3977203
23: -54.9893990, 38.2597084, -55.2042732, 38.6573334, -93.6467209, 93.4639816
24: -58.5330734, 32.7519493, -58.7597504, 33.1360931, -91.6691437, 91.5117035
25: -50.8394318, 44.7083435, -51.0329819, 45.0734444, -95.9128571, 95.7413101
26: -86.1650696, 55.2555199, -86.4756393, 55.5560913, -141.7211609, 141.7311554
27: -69.0680084, 32.1655235, -69.3284912, 32.5234795, -101.5914917, 101.4939957
28: -55.2183647, 44.5214462, -55.4386864, 44.9195175, -100.1378784, 99.9601288
29: -69.8679428, 38.9401016, -70.0711288, 39.3288078, -109.1967468, 109.0112228
30: -69.0321655, 47.7402649, -69.2087402, 48.1395988, -117.1717682, 116.9490051
31: -70.6096039, 33.6015930, -70.8604202, 33.9191742, -104.5287781, 104.4620132
32: -65.1201859, 42.4622345, -65.5480728, 42.6277237, -107.7479019, 108.0102997
33: -84.5839081, 63.8305321, -85.0624695, 64.0035095, -148.5874176, 148.8930054
34: -76.0881119, 53.8311806, -76.3860931, 53.9455414, -130.0336304, 130.2172699
35: -67.2894135, 58.3515167, -67.6036987, 58.4775009, -125.7669067, 125.9552155
36: -70.5463715, 58.8319588, -70.9374847, 58.9381104, -129.4844818, 129.7694397
37: -105.0685577, 48.9160156, -105.6094666, 49.0225563, -154.0911102, 154.5254822
38: -97.5144043, 69.5139313, -98.0467758, 69.6802673, -167.1946716, 167.5606995
39: -104.4131317, 58.4944839, -105.0968475, 58.6574249, -163.0705566, 163.5913391
40: -93.3008347, 39.8260460, -93.8454132, 39.9561844, -133.2570190, 133.6714630
41: -67.3392487, 40.3253479, -67.7338257, 40.4263878, -107.7656403, 108.0591736
42: -53.0874214, 37.9451180, -53.3653259, 38.0546074, -91.1420288, 91.3104401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 601

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9651289, upper bound: 100.0332456
time: 159.70 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9574407, upper bound: 100.0502275
time: 131.91 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -79.9188538, 61.1704407, -79.8485870, 61.1479568, -141.0668030, 141.0190277
1: -50.0181313, 51.0953369, -49.9881554, 51.0776291, -101.0957413, 101.0834808
2: -42.4441719, 46.5956573, -42.4206657, 46.5737839, -89.0179443, 89.0163269
3: -47.3413811, 57.8504410, -47.2986908, 57.8149414, -105.1563110, 105.1491318
4: -51.7887688, 52.2485199, -51.7753983, 52.1955719, -103.9843445, 104.0239182
5: -50.7975845, 58.5031929, -50.7741318, 58.4656143, -109.2631760, 109.2773209
6: -74.8359451, 38.0199127, -74.7949371, 37.9175873, -112.7535324, 112.8148499
7: -64.9875031, 54.6976204, -64.9469910, 54.6762390, -119.6637421, 119.6446075
8: -60.1626282, 58.7948837, -60.1509247, 58.7488365, -118.9114609, 118.9458084
9: -46.0490112, 49.9656334, -46.0293961, 49.9375992, -95.9866104, 95.9950256
10: -75.2493744, 63.5597763, -75.2243042, 63.5214348, -138.7707977, 138.7840881
11: -78.1164093, 46.0414162, -77.9878387, 46.0256805, -124.1420593, 124.0292435
12: -79.5427856, 54.1805458, -79.5098267, 54.1121216, -133.6549072, 133.6903687
13: -62.0842896, 77.8564606, -62.0682526, 77.7720108, -139.8562927, 139.9247131
14: -119.2610550, 37.9770660, -119.1239243, 37.9666328, -157.2276917, 157.1009827
15: -58.1524963, 61.8335571, -58.1281281, 61.7967262, -119.9492188, 119.9616852
16: -85.5115738, 58.0388222, -85.4337845, 58.0125809, -143.5241547, 143.4725952
17: -119.7413788, 53.9787407, -119.5546265, 53.9504013, -173.6917725, 173.5333557
18: -75.0359802, 43.4076080, -74.9967499, 43.3922157, -118.4281845, 118.4043579
19: -59.4812584, 27.6330795, -59.3630714, 27.6218281, -87.1030807, 86.9961548
20: -51.0471191, 35.2709541, -51.0072098, 35.2549515, -86.3020706, 86.2781677
21: -71.6417084, 35.8474464, -71.5268784, 35.8343277, -107.4760361, 107.3743210
22: -68.7384338, 41.7340393, -68.6256332, 41.7203026, -110.4587402, 110.3596725
23: -55.1757889, 38.4766464, -55.0610199, 38.4698524, -93.6456299, 93.5376663
24: -58.7057724, 32.9093437, -58.5735703, 32.9008484, -91.6066208, 91.4829102
25: -50.9899521, 44.8848534, -50.8950691, 44.8691826, -95.8591309, 95.7799072
26: -86.4195175, 55.4363365, -86.3697739, 55.4048843, -141.8244019, 141.8061066
27: -69.2878876, 32.3282280, -69.2086487, 32.3218727, -101.6097565, 101.5368729
28: -55.4018822, 44.7071953, -55.2550049, 44.6933823, -100.0952606, 99.9622040
29: -70.0162430, 39.1241531, -69.8384552, 39.1129417, -109.1291809, 108.9626007
30: -69.1824188, 47.9035072, -69.0558167, 47.8866692, -117.0690918, 116.9593048
31: -70.8095932, 33.7445946, -70.6768341, 33.7333794, -104.5429688, 104.4214325
32: -65.2732239, 42.5872116, -65.2390289, 42.5042496, -107.7774506, 107.8262405
33: -84.8449097, 64.0040359, -84.8121872, 63.9594421, -148.8043213, 148.8162231
34: -76.2712021, 53.9420128, -76.2428894, 53.9118195, -130.1830139, 130.1848755
35: -67.4776611, 58.4899254, -67.4472046, 58.4691315, -125.9467926, 125.9371338
36: -70.7397842, 58.9625473, -70.7200775, 58.8840790, -129.6238708, 129.6826172
37: -105.2985992, 49.0302086, -105.2526169, 48.9203491, -154.2189484, 154.2828217
38: -97.7662735, 69.6802063, -97.7390671, 69.5102539, -167.2765198, 167.4192810
39: -104.7887192, 58.6810951, -104.7581940, 58.5675964, -163.3563232, 163.4392853
40: -93.5536499, 39.9375114, -93.5232086, 39.7873306, -133.3409729, 133.4607239
41: -67.4797058, 40.3973236, -67.4411469, 40.3222961, -107.8020020, 107.8384628
42: -53.1921387, 38.0268631, -53.1648941, 37.9711838, -91.1633224, 91.1917572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 601

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9486760, upper bound: 100.0416960
time: 84.00 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9415487, upper bound: 100.0598547
time: 92.98 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -79.9083099, 61.1681557, -79.9983215, 61.2444496, -141.1527557, 141.1664734
1: -50.0138626, 51.0930862, -50.0612831, 51.1666183, -101.1804733, 101.1543732
2: -42.4370651, 46.5940819, -42.4804993, 46.6891098, -89.1261749, 89.0745850
3: -47.3328552, 57.8480988, -47.3869820, 57.9437599, -105.2766037, 105.2350769
4: -51.7859421, 52.2466087, -51.8486900, 52.3206253, -104.1065674, 104.0952988
5: -50.7947044, 58.5011978, -50.8450050, 58.5900230, -109.3847275, 109.3462067
6: -74.8338242, 38.0201492, -75.0656967, 38.0525360, -112.8863602, 113.0858459
7: -64.9761810, 54.6932449, -65.0041351, 54.8020935, -119.7782593, 119.6973801
8: -60.1516075, 58.7916985, -60.1763229, 58.8859940, -119.0375977, 118.9680176
9: -46.0470886, 49.9635963, -46.1856384, 50.0143166, -96.0613937, 96.1492310
10: -75.2469711, 63.5547371, -75.3563080, 63.6487160, -138.8956757, 138.9110413
11: -78.1036758, 46.0346184, -78.1536102, 46.2436485, -124.3473206, 124.1882172
12: -79.5389709, 54.1610260, -79.8106079, 54.1795120, -133.7184753, 133.9716339
13: -62.0793839, 77.8414001, -62.4133797, 77.8720016, -139.9513855, 140.2547760
14: -119.2565689, 37.9739304, -119.3351974, 38.2126389, -157.4692078, 157.3091278
15: -58.1472473, 61.8287201, -58.1914062, 61.8843956, -120.0316467, 120.0201111
16: -85.5042801, 58.0330544, -85.6035919, 58.1727448, -143.6770325, 143.6366425
17: -119.7418213, 53.9693909, -119.8234100, 54.3082085, -174.0500183, 173.7927856
18: -75.0285110, 43.4060707, -75.0963898, 43.5952415, -118.6237488, 118.5024490
19: -59.4836922, 27.6287308, -59.5344276, 27.8030586, -87.2867432, 87.1631470
20: -51.0409775, 35.2686005, -51.0760536, 35.4156151, -86.4565887, 86.3446503
21: -71.6323013, 35.8422890, -71.6895676, 36.0490341, -107.6813354, 107.5318527
22: -68.7359772, 41.7297478, -68.7989349, 41.9290543, -110.6650314, 110.5286865
23: -55.1725769, 38.4725342, -55.2139244, 38.7110100, -93.8835831, 93.6864395
24: -58.7056541, 32.9073563, -58.7719841, 33.1755600, -91.8812103, 91.6793365
25: -50.9893494, 44.8816338, -51.0440369, 45.1150513, -96.1043930, 95.9256592
26: -86.4137268, 55.4323654, -86.4943542, 55.5983124, -142.0120239, 141.9267273
27: -69.2849121, 32.3266907, -69.3404999, 32.5646591, -101.8495712, 101.6671906
28: -55.4047852, 44.7033768, -55.4473763, 44.9653282, -100.3701172, 100.1507568
29: -70.0166855, 39.1170883, -70.0849915, 39.3731995, -109.3898849, 109.2020721
30: -69.1736069, 47.8985291, -69.2198334, 48.1765366, -117.3501358, 117.1183624
31: -70.8114929, 33.7403793, -70.8719482, 33.9507942, -104.7622681, 104.6123276
32: -65.2697906, 42.5870781, -65.5757523, 42.6411514, -107.9109421, 108.1628265
33: -84.8408508, 63.9956169, -85.1192932, 64.0192261, -148.8600464, 149.1148987
34: -76.2660370, 53.9310303, -76.4224777, 53.9528465, -130.2188873, 130.3535004
35: -67.4740829, 58.4831467, -67.6439056, 58.4887085, -125.9627914, 126.1270447
36: -70.7342224, 58.9503326, -70.9771042, 58.9466743, -129.6808929, 129.9274292
37: -105.2926102, 49.0220490, -105.6538849, 49.0352478, -154.3278503, 154.6759338
38: -97.7564621, 69.6652756, -98.0952454, 69.6906586, -167.4471130, 167.7604980
39: -104.7817841, 58.6693115, -105.1830292, 58.6645775, -163.4463501, 163.8523407
40: -93.5462799, 39.9401321, -93.8957825, 39.9651070, -133.5113831, 133.8359070
41: -67.4772949, 40.3967743, -67.7590561, 40.4374962, -107.9147797, 108.1558304
42: -53.1910858, 38.0229073, -53.3795967, 38.0672836, -91.2583694, 91.4024811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 966

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 601

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -100.0579080, upper bound: 100.0332456
time: 82.75 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -100.0502274, upper bound: 100.0502275
time: 97.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 182.94 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 182.94
Output dim: 13, lower bound: -99.9486760, upper bound: 100.0416960
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 182.94
Output dim: 13, lower bound: -99.9415487, upper bound: 100.0598547
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 182.94
Output dim: 13, lower bound: -99.9651289, upper bound: 100.0332456
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 182.94
Output dim: 13, lower bound: -99.9574407, upper bound: 100.0502275
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 182.94
Output dim: 13, lower bound: -99.9486760, upper bound: 100.0416960
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 182.94
Output dim: 13, lower bound: -99.9415487, upper bound: 100.0598547
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 182.94
Output dim: 13, lower bound: -100.0579080, upper bound: 100.0332456
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 182.94
Output dim: 13, lower bound: -100.0502274, upper bound: 100.0502275

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -79.6509247, 61.0274391, -79.8052673, 61.1349983, -140.7859192, 140.8327026
1: -49.8613434, 50.9941673, -49.9631844, 51.0661087, -100.9274292, 100.9573517
2: -42.3081322, 46.4642792, -42.3935013, 46.5625267, -88.8706589, 88.8577805
3: -47.1295776, 57.6568985, -47.2573090, 57.8014030, -104.9309845, 104.9141998
4: -51.5585823, 52.0512199, -51.7255249, 52.1831932, -103.7417755, 103.7767487
5: -50.6035118, 58.2740669, -50.7319336, 58.4477234, -109.0512390, 109.0059967
6: -74.6729965, 37.7507248, -74.7691727, 37.8910294, -112.5640259, 112.5198898
7: -64.7843475, 54.5326538, -64.9124908, 54.6606560, -119.4449921, 119.4451294
8: -59.9486389, 58.5724792, -60.1085587, 58.7316017, -118.6802292, 118.6810379
9: -45.8781052, 49.8638535, -46.0009041, 49.9241905, -95.8022919, 95.8647614
10: -75.0702362, 63.4163551, -75.1992798, 63.5015106, -138.5717163, 138.6156311
11: -77.7981415, 45.8212585, -77.9606323, 45.9782562, -123.7763901, 123.7818756
12: -79.3446503, 53.9525871, -79.4838333, 54.0825729, -133.4272156, 133.4364014
13: -61.7745247, 77.4737244, -61.9956284, 77.7504578, -139.5249786, 139.4693604
14: -118.9324799, 37.8486633, -119.0903320, 37.9432564, -156.8757324, 156.9389954
15: -57.9838791, 61.6709099, -58.1037636, 61.7826691, -119.7665482, 119.7746735
16: -85.2843094, 57.8731766, -85.4040527, 57.9826889, -143.2669983, 143.2772217
17: -119.3998032, 53.8288956, -119.5196457, 53.9332199, -173.3330231, 173.3485413
18: -74.7826538, 43.1730881, -74.9805374, 43.3407860, -118.1234436, 118.1536255
19: -59.2080536, 27.4917450, -59.3442459, 27.5892830, -86.7973251, 86.8359833
20: -50.8958054, 35.1436424, -50.9944115, 35.2290306, -86.1248322, 86.1380539
21: -71.3083572, 35.6606369, -71.5029144, 35.7902412, -107.0986023, 107.1635437
22: -68.4420013, 41.5978813, -68.5999908, 41.6902466, -110.1322403, 110.1978683
23: -54.8856430, 38.2449646, -55.0427513, 38.4146767, -93.3003159, 93.2877197
24: -58.3790932, 32.7376175, -58.5491028, 32.8600845, -91.2391815, 91.2867203
25: -50.7157784, 44.6896172, -50.8740120, 44.8258667, -95.5416412, 95.5636292
26: -86.1189575, 55.2165680, -86.3469620, 55.3593826, -141.4783325, 141.5635376
27: -69.0088425, 32.1413231, -69.1916962, 32.2786674, -101.2875061, 101.3330078
28: -55.0989151, 44.4992142, -55.2370262, 44.6454849, -99.7443848, 99.7362366
29: -69.6649017, 38.9187317, -69.8085938, 39.0663185, -108.7312012, 108.7273254
30: -68.8541412, 47.7219391, -69.0299377, 47.8479004, -116.7020416, 116.7518692
31: -70.4814529, 33.5858536, -70.6553955, 33.7001495, -104.1815872, 104.2412491
32: -65.0827789, 42.2847214, -65.2081604, 42.4764633, -107.5592346, 107.4928818
33: -84.5283661, 63.8001556, -84.7508392, 63.9406624, -148.4690094, 148.5509949
34: -76.0547791, 53.7708549, -76.2035217, 53.8989601, -129.9537354, 129.9743652
35: -67.2489166, 58.3187103, -67.4035645, 58.4547729, -125.7036743, 125.7222748
36: -70.5261536, 58.6758194, -70.6785126, 58.8624420, -129.3885956, 129.3543243
37: -105.0175476, 48.8534012, -105.2037277, 48.9021988, -153.9197388, 154.0571289
38: -97.4842148, 69.2389526, -97.6874847, 69.4765091, -166.9607239, 166.9264221
39: -104.3733978, 58.3824806, -104.6683960, 58.5508232, -162.9242249, 163.0508728
40: -93.2671051, 39.6663818, -93.4696198, 39.7659149, -133.0330200, 133.1360016
41: -67.3097534, 40.1964836, -67.4134064, 40.3011017, -107.6108551, 107.6098938
42: -53.0589447, 37.8753586, -53.1482964, 37.9526749, -91.0116196, 91.0236511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 601

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9236663, upper bound: 100.0416960
time: 96.48 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9236663, upper bound: 100.0416960
time: 89.34 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -79.7948227, 61.1338081, -79.7948303, 61.1335258, -140.9283447, 140.9286346
1: -49.9359093, 51.1034012, -49.9565849, 51.0637360, -100.9996490, 101.0599823
2: -42.3534393, 46.5579872, -42.3840256, 46.5610161, -88.9144287, 88.9420166
3: -47.2452087, 57.7791824, -47.2550354, 57.7996216, -105.0448151, 105.0342026
4: -51.6246376, 52.1683426, -51.7224007, 52.1815910, -103.8062286, 103.8907471
5: -50.6660004, 58.3857460, -50.7238388, 58.4460068, -109.1120071, 109.1095810
6: -74.9153519, 37.9090271, -74.7677765, 37.8928680, -112.8082199, 112.6767883
7: -64.8568268, 54.6617470, -64.8995361, 54.6578979, -119.5147247, 119.5612793
8: -59.9976501, 58.7308846, -60.0936394, 58.7288666, -118.7265167, 118.8245239
9: -46.0011940, 49.9255753, -45.9986725, 49.9214325, -95.9226151, 95.9242477
10: -75.2676239, 63.5216980, -75.1976547, 63.4964294, -138.7640381, 138.7193604
11: -77.9868164, 46.0541573, -77.9524307, 45.9740562, -123.9608688, 124.0065918
12: -79.6538239, 54.0391693, -79.4805450, 54.0625992, -133.7164307, 133.5197144
13: -62.1342926, 77.5992737, -61.9907227, 77.7381439, -139.8724365, 139.5899811
14: -119.1364975, 38.0432472, -119.0866470, 37.9412842, -157.0777893, 157.1298828
15: -58.0715675, 61.7927475, -58.1007500, 61.7817650, -119.8533249, 119.8934937
16: -85.4334106, 58.0494156, -85.3958817, 57.9793434, -143.4127502, 143.4452820
17: -119.6426849, 54.1302032, -119.5195541, 53.9269905, -173.5696716, 173.6497498
18: -74.9025040, 43.3698425, -74.9779434, 43.3382416, -118.2407303, 118.3477859
19: -59.3544083, 27.6723366, -59.3440742, 27.5869789, -86.9413910, 87.0164108
20: -50.9786835, 35.2713242, -50.9878693, 35.2274399, -86.2061157, 86.2591934
21: -71.5279999, 35.9136238, -71.5014343, 35.7871017, -107.3151016, 107.4150543
22: -68.6431046, 41.7948608, -68.5996552, 41.6880417, -110.3311462, 110.3945160
23: -55.0261650, 38.4573669, -55.0402527, 38.4130287, -93.4391937, 93.4976120
24: -58.5852394, 32.9975777, -58.5510292, 32.8586121, -91.4438248, 91.5485992
25: -50.8901825, 44.9073105, -50.8748283, 44.8244934, -95.7146759, 95.7821350
26: -86.2590637, 55.3909454, -86.3446121, 55.3563690, -141.6154327, 141.7355499
27: -69.1234589, 32.4018250, -69.1870499, 32.2772446, -101.4007034, 101.5888748
28: -55.2413559, 44.7755852, -55.2362137, 44.6435966, -99.8849411, 100.0117950
29: -69.9180908, 39.1771851, -69.8103409, 39.0621796, -108.9802704, 108.9875107
30: -69.0920868, 48.0605850, -69.0294952, 47.8452950, -116.9373703, 117.0900726
31: -70.6607666, 33.7835846, -70.6534424, 33.6975594, -104.3583221, 104.4370270
32: -65.4678650, 42.4945068, -65.2050629, 42.4799652, -107.9478149, 107.6995621
33: -84.7795639, 63.8908806, -84.7476578, 63.9390068, -148.7185669, 148.6385345
34: -76.2711487, 53.8479729, -76.1994324, 53.8903427, -130.1614838, 130.0473938
35: -67.4342957, 58.3656158, -67.4006348, 58.4482384, -125.8825378, 125.7662506
36: -70.8326111, 58.8164673, -70.6744080, 58.8497391, -129.6823425, 129.4908752
37: -105.2883148, 48.9178505, -105.2004471, 48.8918304, -154.1801453, 154.1183014
38: -97.9753647, 69.5419312, -97.6825867, 69.4753723, -167.4507446, 167.2245178
39: -104.8169937, 58.5044174, -104.6637573, 58.5463676, -163.3633575, 163.1681671
40: -93.5301285, 39.8364563, -93.4659882, 39.7680397, -133.2981720, 133.3024445
41: -67.6130524, 40.3636093, -67.4119186, 40.3034630, -107.9165039, 107.7755280
42: -53.3041534, 37.9875717, -53.1473122, 37.9512329, -91.2553787, 91.1348877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=500, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0301603
time: 112.69 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0227649
time: 100.01 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -79.6404419, 61.0252304, -79.9551392, 61.2315102, -140.8719482, 140.9803772
1: -49.8571739, 50.9919014, -50.0362968, 51.1550751, -101.0122528, 101.0281906
2: -42.3010025, 46.4627151, -42.4533424, 46.6778908, -88.9788971, 88.9160614
3: -47.1211090, 57.6545563, -47.3457451, 57.9302330, -105.0513306, 105.0003052
4: -51.5557709, 52.0492897, -51.7988014, 52.3082085, -103.8639679, 103.8480911
5: -50.6006508, 58.2720757, -50.8028450, 58.5721970, -109.1728516, 109.0749207
6: -74.6708527, 37.7509460, -75.0399399, 38.0259552, -112.6968079, 112.7908783
7: -64.7730560, 54.5282822, -64.9696655, 54.7865028, -119.5595551, 119.4979324
8: -59.9376411, 58.5693207, -60.1339798, 58.8687782, -118.8064194, 118.7033005
9: -45.8762360, 49.8618469, -46.1571579, 50.0009079, -95.8771362, 96.0190048
10: -75.0678864, 63.4113159, -75.3312531, 63.6287918, -138.6966553, 138.7425690
11: -77.7857361, 45.8144989, -78.1264343, 46.1963043, -123.9820404, 123.9409180
12: -79.3408966, 53.9330063, -79.7846298, 54.1499939, -133.4908752, 133.7176361
13: -61.7695885, 77.4583969, -62.3407898, 77.8504333, -139.6200256, 139.7991638
14: -118.9281235, 37.8455086, -119.3016205, 38.1892738, -157.1173706, 157.1471252
15: -57.9786491, 61.6662903, -58.1670227, 61.8703804, -119.8489990, 119.8333130
16: -85.2773056, 57.8673820, -85.5739136, 58.1428566, -143.4201660, 143.4412842
17: -119.4002533, 53.8195419, -119.7883530, 54.2909546, -173.6912079, 173.6078949
18: -74.7752838, 43.1715164, -75.0801392, 43.5438461, -118.3191223, 118.2516556
19: -59.2104645, 27.4873810, -59.5156059, 27.7705288, -86.9809875, 87.0029831
20: -50.8896980, 35.1412964, -51.0633049, 35.3897018, -86.2794037, 86.2045975
21: -71.2990570, 35.6555176, -71.6656494, 36.0049362, -107.3039856, 107.3211670
22: -68.4395294, 41.5936089, -68.7732391, 41.8989563, -110.3384781, 110.3668518
23: -54.8823853, 38.2408409, -55.1956024, 38.6558647, -93.5382309, 93.4364471
24: -58.3789520, 32.7356262, -58.7475052, 33.1347961, -91.5137482, 91.4831314
25: -50.7151222, 44.6863976, -51.0228996, 45.0717163, -95.7868271, 95.7092896
26: -86.1131821, 55.2125778, -86.4715424, 55.5527954, -141.6659851, 141.6841125
27: -69.0059052, 32.1397591, -69.3235931, 32.5214577, -101.5273514, 101.4633484
28: -55.1017876, 44.4953918, -55.4293823, 44.9174728, -100.0192566, 99.9247742
29: -69.6652908, 38.9117432, -70.0550690, 39.3266182, -108.9919052, 108.9667969
30: -68.8452148, 47.7169991, -69.1938629, 48.1378174, -116.9830170, 116.9108582
31: -70.4833221, 33.5816689, -70.8505249, 33.9175949, -104.4009094, 104.4321899
32: -65.0794144, 42.2845840, -65.5448761, 42.6133194, -107.6927338, 107.8294601
33: -84.5243073, 63.7917442, -85.0578384, 64.0004425, -148.5247498, 148.8495636
34: -76.0496292, 53.7599907, -76.3830414, 53.9400215, -129.9896393, 130.1430206
35: -67.2453308, 58.3119965, -67.6002274, 58.4743271, -125.7196579, 125.9122238
36: -70.5206223, 58.6636200, -70.9355011, 58.9250488, -129.4456787, 129.5991211
37: -105.0115738, 48.8453102, -105.6049652, 49.0170975, -154.0286713, 154.4502716
38: -97.4744720, 69.2228394, -98.0436249, 69.6567764, -167.1312256, 167.2664490
39: -104.3664703, 58.3708649, -105.0932541, 58.6477814, -163.0142517, 163.4641113
40: -93.2597504, 39.6689377, -93.8421860, 39.9435959, -133.2033386, 133.5111237
41: -67.3073120, 40.1959076, -67.7313004, 40.4163132, -107.7236252, 107.9272003
42: -53.0578804, 37.8713913, -53.3629990, 38.0487900, -91.1066742, 91.2343903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 601

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9403398, upper bound: 100.0332456
time: 67.38 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9403398, upper bound: 100.0332456
time: 130.03 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -79.7844772, 61.1315804, -79.9445343, 61.2299576, -141.0144348, 141.0761108
1: -49.9318161, 51.1011314, -50.0295105, 51.1527100, -101.0845261, 101.1306458
2: -42.3462753, 46.5563889, -42.4438286, 46.6763611, -89.0226364, 89.0002060
3: -47.2367325, 57.7768478, -47.3432388, 57.9283981, -105.1651306, 105.1200867
4: -51.6217728, 52.1666069, -51.7957764, 52.3067131, -103.9284821, 103.9623871
5: -50.6630745, 58.3836861, -50.7947693, 58.5704498, -109.2335129, 109.1784515
6: -74.9132080, 37.9092216, -75.0384979, 38.0278206, -112.9410248, 112.9477158
7: -64.8456879, 54.6574059, -64.9567871, 54.7837563, -119.6294403, 119.6141815
8: -59.9867783, 58.7277412, -60.1190147, 58.8660316, -118.8528061, 118.8467560
9: -45.9992790, 49.9235573, -46.1549263, 49.9982224, -95.9974976, 96.0784760
10: -75.2652435, 63.5165825, -75.3296432, 63.6239090, -138.8891449, 138.8462219
11: -77.9743195, 46.0474014, -78.1181488, 46.1920090, -124.1663284, 124.1655502
12: -79.6501160, 54.0192261, -79.7813187, 54.1300354, -133.7801514, 133.8005371
13: -62.1293488, 77.5834045, -62.3358650, 77.8379822, -139.9673309, 139.9192657
14: -119.1320038, 38.0400887, -119.2978210, 38.1873322, -157.3193359, 157.3379059
15: -58.0662956, 61.7881889, -58.1639862, 61.8694344, -119.9357147, 119.9521790
16: -85.4263687, 58.0436134, -85.5656738, 58.1395569, -143.5659180, 143.6092834
17: -119.6430588, 54.1208839, -119.7882996, 54.2847137, -173.9277649, 173.9091797
18: -74.8953400, 43.3682861, -75.0775452, 43.5412674, -118.4365997, 118.4458237
19: -59.3567963, 27.6680145, -59.5154457, 27.7682457, -87.1250305, 87.1834564
20: -50.9726028, 35.2689743, -51.0567207, 35.3880844, -86.3606720, 86.3256989
21: -71.5186462, 35.9085007, -71.6641083, 36.0017624, -107.5204086, 107.5726013
22: -68.6405792, 41.7905731, -68.7729568, 41.8967438, -110.5373230, 110.5635223
23: -55.0229034, 38.4532547, -55.1931229, 38.6542206, -93.6771088, 93.6463623
24: -58.5851097, 32.9955750, -58.7494164, 33.1333084, -91.7184143, 91.7449951
25: -50.8895111, 44.9040985, -51.0237808, 45.0703392, -95.9598541, 95.9278717
26: -86.2532730, 55.3869476, -86.4691925, 55.5497704, -141.8030396, 141.8561401
27: -69.1204987, 32.4002838, -69.3187561, 32.5200348, -101.6405334, 101.7190399
28: -55.2442245, 44.7717247, -55.4286575, 44.9155159, -100.1597443, 100.2003708
29: -69.9183807, 39.1701736, -70.0568695, 39.3224258, -109.2407990, 109.2270432
30: -69.0832520, 48.0556183, -69.1935577, 48.1351738, -117.2184143, 117.2491760
31: -70.6625977, 33.7794075, -70.8485489, 33.9149551, -104.5775528, 104.6279526
32: -65.4644928, 42.4943390, -65.5417557, 42.6168137, -108.0812988, 108.0360870
33: -84.7754517, 63.8824844, -85.0546417, 63.9987526, -148.7741852, 148.9371338
34: -76.2659760, 53.8371811, -76.3789062, 53.9313431, -130.1973267, 130.2160950
35: -67.4307251, 58.3589630, -67.5972519, 58.4677696, -125.8984833, 125.9562073
36: -70.8271027, 58.8042679, -70.9313889, 58.9123039, -129.7393951, 129.7356567
37: -105.2822647, 48.9096794, -105.6016769, 49.0064278, -154.2886658, 154.5113525
38: -97.9656448, 69.5261765, -98.0386810, 69.6554413, -167.6210938, 167.5648499
39: -104.8100357, 58.4927711, -105.0885696, 58.6433029, -163.4533386, 163.5813446
40: -93.5227890, 39.8390350, -93.8385162, 39.9457664, -133.4685516, 133.6775513
41: -67.6106033, 40.3629837, -67.7297821, 40.4186363, -108.0292358, 108.0927658
42: -53.3030815, 37.9835777, -53.3619881, 38.0472412, -91.3503113, 91.3455582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=500, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0209140
time: 87.57 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0134171
time: 111.51 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -79.8281097, 61.1360779, -79.8413239, 61.1452255, -140.9733276, 140.9773865
1: -49.9646454, 51.0693970, -49.9839592, 51.0755692, -101.0402145, 101.0533524
2: -42.4118652, 46.5706940, -42.4180374, 46.5717735, -88.9836426, 88.9887238
3: -47.2804031, 57.8146896, -47.2938156, 57.8120537, -105.0924530, 105.1084976
4: -51.7526932, 52.1978951, -51.7725792, 52.1915398, -103.9442291, 103.9704742
5: -50.7614059, 58.4619331, -50.7712822, 58.4622612, -109.2236633, 109.2332153
6: -74.7981262, 37.8812408, -74.7919312, 37.9064713, -112.7045975, 112.6731720
7: -64.9196777, 54.6667290, -64.9416962, 54.6737823, -119.5934448, 119.6084061
8: -60.1189804, 58.7443123, -60.1474838, 58.7447815, -118.8637619, 118.8917923
9: -46.0067940, 49.9319572, -46.0261002, 49.9349327, -95.9416962, 95.9580536
10: -75.2069855, 63.5177689, -75.2209167, 63.5181427, -138.7251282, 138.7386780
11: -77.9815521, 46.0108910, -77.9771881, 46.0232239, -124.0047760, 123.9880753
12: -79.5004425, 54.0711250, -79.5065079, 54.1034470, -133.6038818, 133.5776367
13: -62.0451088, 77.7426300, -62.0651703, 77.7627640, -139.8078613, 139.8078003
14: -119.1416550, 37.9524422, -119.1146317, 37.9647064, -157.1063385, 157.0670776
15: -58.0984573, 61.7939339, -58.1238747, 61.7936020, -119.8920517, 119.9178085
16: -85.4284058, 57.9895210, -85.4272308, 58.0086746, -143.4370728, 143.4167480
17: -119.5728531, 53.9450569, -119.5413971, 53.9478111, -173.5206604, 173.4864502
18: -74.9948425, 43.3678246, -74.9935150, 43.3890915, -118.3839340, 118.3613358
19: -59.3682518, 27.6170082, -59.3539314, 27.6205406, -86.9887848, 86.9709396
20: -51.0039673, 35.2418518, -51.0037575, 35.2526779, -86.2566376, 86.2456055
21: -71.4849243, 35.8296700, -71.5144577, 35.8329239, -107.3178406, 107.3441162
22: -68.5997467, 41.7141647, -68.6147003, 41.7187347, -110.3184814, 110.3288651
23: -55.0688210, 38.4578590, -55.0523911, 38.4683456, -93.5371552, 93.5102539
24: -58.5516739, 32.8930359, -58.5613441, 32.8995743, -91.4512177, 91.4543762
25: -50.8656845, 44.8628998, -50.8850555, 44.8674393, -95.7331238, 95.7479553
26: -86.3675537, 55.3934288, -86.3656769, 55.4015465, -141.7691040, 141.7590942
27: -69.2256927, 32.3024521, -69.2037506, 32.3198471, -101.5455246, 101.5062027
28: -55.2853470, 44.6812134, -55.2457695, 44.6913300, -99.9766617, 99.9269867
29: -69.8136444, 39.0957489, -69.8224792, 39.1107101, -108.9243469, 108.9182281
30: -68.9955444, 47.8802032, -69.0409851, 47.8848038, -116.8803406, 116.9211884
31: -70.6833191, 33.7246437, -70.6669464, 33.7317810, -104.4151001, 104.3915863
32: -65.2324219, 42.4095535, -65.2358246, 42.4898987, -107.7223206, 107.6453781
33: -84.7853546, 63.9651604, -84.8075027, 63.9563789, -148.7417297, 148.7726593
34: -76.2327271, 53.8707008, -76.2398834, 53.9062881, -130.1390076, 130.1105804
35: -67.4335632, 58.4503479, -67.4437027, 58.4659805, -125.8995438, 125.8940430
36: -70.7140427, 58.7941818, -70.7180328, 58.8710785, -129.5851135, 129.5122070
37: -105.2416153, 48.9594193, -105.2481308, 48.9148750, -154.1564941, 154.2075500
38: -97.7262650, 69.3903198, -97.7359161, 69.4869385, -167.2131958, 167.1262360
39: -104.7420807, 58.5572929, -104.7544785, 58.5579529, -163.3000183, 163.3117676
40: -93.5125656, 39.7804642, -93.5199814, 39.7748337, -133.2873840, 133.3004456
41: -67.4477997, 40.2678833, -67.4386292, 40.3122253, -107.7600250, 107.7065125
42: -53.1625862, 37.9531250, -53.1625557, 37.9653549, -91.1279449, 91.1156769

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 601

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9236663, upper bound: 100.0416960
time: 105.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9236663, upper bound: 100.0416960
time: 95.38 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -79.9720154, 61.2424049, -79.8309555, 61.1436844, -141.1156921, 141.0733643
1: -50.0392570, 51.1786232, -49.9773560, 51.0732117, -101.1124725, 101.1559753
2: -42.4571533, 46.6643181, -42.4085884, 46.5702782, -89.0274353, 89.0728989
3: -47.3959961, 57.9369545, -47.2915115, 57.8102684, -105.2062683, 105.2284698
4: -51.8187943, 52.3149681, -51.7694664, 52.1899719, -104.0087662, 104.0844193
5: -50.8239250, 58.5735397, -50.7631950, 58.4605446, -109.2844696, 109.3367310
6: -75.0404892, 38.0394821, -74.7905273, 37.9083557, -112.9488373, 112.8300018
7: -64.9922333, 54.7957840, -64.9287415, 54.6710396, -119.6632690, 119.7245255
8: -60.1680412, 58.9026527, -60.1325455, 58.7420197, -118.9100647, 119.0352020
9: -46.1298790, 49.9937172, -46.0238495, 49.9321938, -96.0620575, 96.0175629
10: -75.4043045, 63.6230507, -75.2193069, 63.5130424, -138.9173431, 138.8423615
11: -78.1701050, 46.2437668, -77.9689941, 46.0190125, -124.1891174, 124.2127533
12: -79.8096390, 54.1576805, -79.5031891, 54.0834503, -133.8930969, 133.6608734
13: -62.4049377, 77.8681564, -62.0602417, 77.7504196, -140.1553650, 139.9284058
14: -119.3455811, 38.1470299, -119.1108932, 37.9627762, -157.3083496, 157.2579193
15: -58.1861229, 61.9157906, -58.1209030, 61.7926674, -119.9787827, 120.0366821
16: -85.5774384, 58.1658554, -85.4190674, 58.0053482, -143.5827789, 143.5849304
17: -119.8157501, 54.2463379, -119.5413818, 53.9414978, -173.7572327, 173.7877197
18: -75.1146851, 43.5646362, -74.9909210, 43.3865547, -118.5012360, 118.5555496
19: -59.5145683, 27.7976208, -59.3537674, 27.6182384, -87.1328049, 87.1513901
20: -51.0868187, 35.3695831, -50.9972153, 35.2510910, -86.3379059, 86.3667831
21: -71.7045288, 36.0826492, -71.5129547, 35.8297653, -107.5342941, 107.5956039
22: -68.8007889, 41.9111328, -68.6143875, 41.7165184, -110.5173035, 110.5255127
23: -55.2093506, 38.6702728, -55.0499115, 38.4667244, -93.6760712, 93.7201843
24: -58.7577820, 33.1529770, -58.5632896, 32.8980827, -91.6558533, 91.7162628
25: -51.0400581, 45.0806427, -50.8858833, 44.8660736, -95.9061279, 95.9665146
26: -86.5076447, 55.5678291, -86.3633804, 55.3985825, -141.9062042, 141.9312134
27: -69.3403015, 32.5629959, -69.1990967, 32.3184395, -101.6587372, 101.7620926
28: -55.4277191, 44.9575577, -55.2449570, 44.6894150, -100.1171341, 100.2025146
29: -70.0667572, 39.3542137, -69.8242645, 39.1065331, -109.1732941, 109.1784668
30: -69.2334442, 48.2188759, -69.0405579, 47.8821793, -117.1156235, 117.2594299
31: -70.8625946, 33.9224129, -70.6650085, 33.7291565, -104.5917511, 104.5874176
32: -65.6175232, 42.6192741, -65.2327271, 42.4933701, -108.1108856, 107.8519974
33: -85.0366974, 64.0559235, -84.8043518, 63.9547310, -148.9914093, 148.8602600
34: -76.4492188, 53.9478340, -76.2357330, 53.8976288, -130.3468475, 130.1835632
35: -67.6190186, 58.4972801, -67.4407806, 58.4594231, -126.0784454, 125.9380646
36: -71.0205383, 58.9347305, -70.7139587, 58.8583069, -129.8788452, 129.6486816
37: -105.5124664, 49.0238304, -105.2448730, 48.9045029, -154.4169617, 154.2687073
38: -98.2174835, 69.6932144, -97.7309952, 69.4857635, -167.7032471, 167.4242096
39: -105.1858521, 58.6791992, -104.7498627, 58.5534859, -163.7393341, 163.4290619
40: -93.7756500, 39.9505081, -93.5163345, 39.7769737, -133.5526276, 133.4668427
41: -67.7510986, 40.4349823, -67.4371414, 40.3145790, -108.0656586, 107.8721237
42: -53.4078064, 38.0652809, -53.1615562, 37.9638977, -91.3717041, 91.2268372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0301603
time: 85.12 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0227649
time: 89.94 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -79.8176575, 61.1338501, -79.9912415, 61.2416992, -141.0593567, 141.1250916
1: -49.9604988, 51.0671043, -50.0570679, 51.1645432, -101.1250458, 101.1241760
2: -42.4047394, 46.5691185, -42.4778976, 46.6871185, -89.0918427, 89.0470123
3: -47.2719040, 57.8123512, -47.3822021, 57.9408760, -105.2127838, 105.1945496
4: -51.7498550, 52.1959648, -51.8458786, 52.3165703, -104.0664139, 104.0418320
5: -50.7585297, 58.4599876, -50.8422089, 58.5867310, -109.3452606, 109.3022003
6: -74.7959900, 37.8814468, -75.0627060, 38.0414162, -112.8374023, 112.9441528
7: -64.9084167, 54.6623802, -64.9988556, 54.7996330, -119.7080383, 119.6612396
8: -60.1079941, 58.7411804, -60.1728897, 58.8819580, -118.9899521, 118.9140701
9: -46.0048828, 49.9299355, -46.1823692, 50.0116615, -96.0165405, 96.1122971
10: -75.2046051, 63.5126572, -75.3529358, 63.6454048, -138.8499908, 138.8656006
11: -77.9691315, 46.0041161, -78.1430054, 46.2412834, -124.2104187, 124.1471252
12: -79.4967194, 54.0515633, -79.8073044, 54.1708145, -133.6675415, 133.8588715
13: -62.0401764, 77.7272873, -62.4103088, 77.8627014, -139.9028778, 140.1376038
14: -119.1372681, 37.9492722, -119.3258362, 38.2107086, -157.3479767, 157.2751160
15: -58.0931816, 61.7893448, -58.1871338, 61.8812675, -119.9744492, 119.9764786
16: -85.4214478, 57.9837532, -85.5970764, 58.1688652, -143.5903168, 143.5808258
17: -119.5733261, 53.9356842, -119.8101730, 54.3055725, -173.8788757, 173.7458496
18: -74.9874420, 43.3662605, -75.0931244, 43.5921936, -118.5796204, 118.4593811
19: -59.3706551, 27.6126633, -59.5252876, 27.8018131, -87.1724701, 87.1379471
20: -50.9978561, 35.2395172, -51.0726242, 35.4133415, -86.4111862, 86.3121338
21: -71.4756165, 35.8245277, -71.6771469, 36.0476189, -107.5232391, 107.5016785
22: -68.5972443, 41.7098846, -68.7879486, 41.9274635, -110.5247040, 110.4978333
23: -55.0655518, 38.4537201, -55.2052574, 38.7095337, -93.7750702, 93.6589813
24: -58.5515442, 32.8910446, -58.7597198, 33.1742859, -91.7258148, 91.6507645
25: -50.8650093, 44.8597031, -51.0339355, 45.1133270, -95.9783325, 95.8936386
26: -86.3617783, 55.3895035, -86.4902344, 55.5950012, -141.9567871, 141.8797302
27: -69.2227936, 32.3009224, -69.3356018, 32.5626602, -101.7854538, 101.6365204
28: -55.2882042, 44.6773529, -55.4381027, 44.9633064, -100.2515106, 100.1154404
29: -69.8140182, 39.0887375, -70.0689545, 39.3709831, -109.1849976, 109.1576843
30: -68.9866028, 47.8752708, -69.2049255, 48.1747017, -117.1613007, 117.0801926
31: -70.6851959, 33.7204437, -70.8620682, 33.9492073, -104.6343918, 104.5825119
32: -65.2290421, 42.4094315, -65.5725479, 42.6267509, -107.8557892, 107.9819794
33: -84.7812576, 63.9567719, -85.1146393, 64.0162048, -148.7974548, 149.0714111
34: -76.2275696, 53.8598251, -76.4194031, 53.9473152, -130.1748657, 130.2792358
35: -67.4299774, 58.4436302, -67.6404190, 58.4855270, -125.9155045, 126.0840454
36: -70.7084579, 58.7819748, -70.9750443, 58.9336014, -129.6420593, 129.7570190
37: -105.2356873, 48.9513168, -105.6494141, 49.0297928, -154.2654724, 154.6007385
38: -97.7165146, 69.3741455, -98.0921097, 69.6671906, -167.3836975, 167.4662476
39: -104.7351685, 58.5456543, -105.1794357, 58.6549339, -163.3901062, 163.7250671
40: -93.5052338, 39.7830162, -93.8925934, 39.9525299, -133.4577484, 133.6755981
41: -67.4453659, 40.2673340, -67.7565308, 40.4274063, -107.8727722, 108.0238647
42: -53.1615334, 37.9491653, -53.3772583, 38.0614243, -91.2229462, 91.3264236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 601

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -100.0332453, upper bound: 100.0332456
time: 207.93 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -100.0332453, upper bound: 100.0332456
time: 92.20 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -79.9617004, 61.2401657, -79.9806213, 61.2401848, -141.2018890, 141.2207947
1: -50.0351448, 51.1763306, -50.0502815, 51.1621666, -101.1973114, 101.2266083
2: -42.4499855, 46.6627426, -42.4683876, 46.6856308, -89.1356201, 89.1311340
3: -47.3875313, 57.9345932, -47.3797150, 57.9390640, -105.3265839, 105.3143082
4: -51.8158951, 52.3131981, -51.8428497, 52.3150673, -104.1309586, 104.1560440
5: -50.8209648, 58.5715218, -50.8340950, 58.5849953, -109.4059448, 109.4056168
6: -75.0383301, 38.0396461, -75.0612488, 38.0432739, -113.0816040, 113.1008911
7: -64.9810333, 54.7914505, -64.9859619, 54.7968903, -119.7779083, 119.7774048
8: -60.1571655, 58.8995018, -60.1579552, 58.8791771, -119.0363235, 119.0574570
9: -46.1279411, 49.9916840, -46.1801262, 50.0089722, -96.1369171, 96.1717987
10: -75.4019394, 63.6179848, -75.3513184, 63.6405449, -139.0424805, 138.9692993
11: -78.1576157, 46.2370300, -78.1347198, 46.2370224, -124.3946228, 124.3717499
12: -79.8059082, 54.1377640, -79.8039932, 54.1508904, -133.9568024, 133.9417419
13: -62.3999901, 77.8523178, -62.4053955, 77.8502121, -140.2501984, 140.2577057
14: -119.3410721, 38.1438560, -119.3220901, 38.2088089, -157.5498810, 157.4659424
15: -58.1808510, 61.9112053, -58.1841354, 61.8803139, -120.0611649, 120.0953369
16: -85.5703735, 58.1600571, -85.5888443, 58.1655579, -143.7359314, 143.7488861
17: -119.8161469, 54.2369995, -119.8100891, 54.2992859, -174.1154327, 174.0470886
18: -75.1075287, 43.5630531, -75.0905457, 43.5896492, -118.6971664, 118.6535873
19: -59.5169678, 27.7932816, -59.5251198, 27.7995205, -87.3164673, 87.3183975
20: -51.0807381, 35.3672333, -51.0660591, 35.4117508, -86.4924774, 86.4332886
21: -71.6952057, 36.0775223, -71.6755981, 36.0444603, -107.7396698, 107.7531204
22: -68.7982483, 41.9068298, -68.7876587, 41.9252472, -110.7234955, 110.6944885
23: -55.2060852, 38.6661720, -55.2027740, 38.7078552, -93.9139404, 93.8689346
24: -58.7576485, 33.1509781, -58.7616196, 33.1727905, -91.9304352, 91.9125977
25: -51.0393715, 45.0774231, -51.0347977, 45.1119194, -96.1512909, 96.1122208
26: -86.5018082, 55.5638390, -86.4878769, 55.5919724, -142.0937805, 142.0517120
27: -69.3373260, 32.5614395, -69.3307724, 32.5612602, -101.8985825, 101.8922119
28: -55.4305878, 44.9537201, -55.4373703, 44.9613647, -100.3919525, 100.3910904
29: -70.0670853, 39.3471756, -70.0707474, 39.3668022, -109.4338837, 109.4179230
30: -69.2246246, 48.2139359, -69.2046204, 48.1720734, -117.3966751, 117.4185562
31: -70.8644714, 33.9182167, -70.8600998, 33.9465866, -104.8110580, 104.7783203
32: -65.6141434, 42.6191254, -65.5694275, 42.6302338, -108.2443771, 108.1885529
33: -85.0326309, 64.0474854, -85.1115112, 64.0144653, -149.0470886, 149.1589966
34: -76.4440460, 53.9369812, -76.4152374, 53.9386978, -130.3827362, 130.3522186
35: -67.6154327, 58.4906082, -67.6374435, 58.4789772, -126.0944061, 126.1280518
36: -71.0149918, 58.9225540, -70.9709625, 58.9208603, -129.9358521, 129.8935242
37: -105.5064316, 49.0156555, -105.6461487, 49.0191040, -154.5255280, 154.6618042
38: -98.2077637, 69.6773834, -98.0871582, 69.6658173, -167.8735809, 167.7645416
39: -105.1788940, 58.6675224, -105.1747513, 58.6504250, -163.8293152, 163.8422699
40: -93.7683716, 39.9530563, -93.8889160, 39.9546776, -133.7230225, 133.8419800
41: -67.7486572, 40.4343643, -67.7550507, 40.4297562, -108.1784134, 108.1894150
42: -53.4067345, 38.0612946, -53.3762665, 38.0598984, -91.4666290, 91.4375534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -100.0134170, upper bound: 100.0209140
time: 83.64 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0134171
time: 133.89 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 220.02 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -99.9236663, upper bound: 100.0416960
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -99.9236663, upper bound: 100.0416960
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0301603
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0227649
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -99.9403398, upper bound: 100.0332456
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -99.9403398, upper bound: 100.0332456
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0209140
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0134171
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -99.9236663, upper bound: 100.0416960
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -99.9236663, upper bound: 100.0416960
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0301603
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0227649
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -100.0332453, upper bound: 100.0332456
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -100.0332453, upper bound: 100.0332456
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -100.0134170, upper bound: 100.0209140
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 220.02
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0134171

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -79.6509247, 61.0274391, -79.7222748, 61.1033821, -140.7543030, 140.7497101
1: -49.8613434, 50.9941673, -49.9141731, 51.0421524, -100.9034958, 100.9083405
2: -42.3081322, 46.4642792, -42.3638916, 46.5393066, -88.8474426, 88.8281708
3: -47.1295776, 57.6568985, -47.2014694, 57.7682419, -104.8978195, 104.8583603
4: -51.5585823, 52.0512199, -51.6922531, 52.1369858, -103.6955643, 103.7434692
5: -50.6035118, 58.2740669, -50.6986542, 58.4092331, -109.0127411, 108.9727173
6: -74.6729965, 37.7507248, -74.7343826, 37.7638283, -112.4368286, 112.4851074
7: -64.7843475, 54.5326538, -64.8500671, 54.6321335, -119.4164810, 119.3827209
8: -59.9486389, 58.5724792, -60.0683556, 58.6843185, -118.6329575, 118.6408234
9: -45.8781052, 49.8638535, -45.9617462, 49.8932762, -95.7713776, 95.8255997
10: -75.0702362, 63.4163551, -75.1601410, 63.4628563, -138.5330811, 138.5764923
11: -77.7981415, 45.8212585, -77.8365936, 45.9500847, -123.7482300, 123.6578445
12: -79.3446503, 53.9525871, -79.4448395, 53.9819641, -133.3265991, 133.3974152
13: -61.7745247, 77.4737244, -61.9595070, 77.6462860, -139.4208069, 139.4332275
14: -118.9324799, 37.8486633, -118.9806976, 37.9204788, -156.8529663, 156.8293610
15: -57.9838791, 61.6709099, -58.0537033, 61.7462044, -119.7300720, 119.7246094
16: -85.2843094, 57.8731766, -85.3274536, 57.9370537, -143.2213440, 143.2005920
17: -119.3998032, 53.8288956, -119.3650513, 53.9022064, -173.3020020, 173.1939392
18: -74.7826538, 43.1730881, -74.9426727, 43.3038979, -118.0865479, 118.1157532
19: -59.2080536, 27.4917450, -59.2407532, 27.5745049, -86.7825623, 86.7324982
20: -50.8958054, 35.1436424, -50.9549026, 35.2022018, -86.0980072, 86.0985413
21: -71.3083572, 35.6606369, -71.3589325, 35.7738647, -107.0822144, 107.0195618
22: -68.4420013, 41.5978813, -68.4726562, 41.6719742, -110.1139755, 110.0705414
23: -54.8856430, 38.2449646, -54.9449043, 38.3973198, -93.2829514, 93.1898651
24: -58.3790932, 32.7376175, -58.4077034, 32.8450546, -91.2241516, 91.1453247
25: -50.7157784, 44.6896172, -50.7609253, 44.8056755, -95.5214386, 95.4505310
26: -86.1189575, 55.2165680, -86.2992401, 55.3197937, -141.4387512, 141.5158081
27: -69.0088425, 32.1413231, -69.1346741, 32.2547836, -101.2636261, 101.2759857
28: -55.0989151, 44.4992142, -55.1304855, 44.6215363, -99.7204514, 99.6296921
29: -69.6649017, 38.9187317, -69.6228027, 39.0401764, -108.7050781, 108.5415344
30: -68.8541412, 47.7219391, -68.8582764, 47.8263855, -116.6805267, 116.5802155
31: -70.4814529, 33.5858536, -70.5395203, 33.6818085, -104.1632614, 104.1253662
32: -65.0827789, 42.2847214, -65.1706696, 42.3134537, -107.3962250, 107.4553909
33: -84.5283661, 63.8001556, -84.6959152, 63.9049759, -148.4333496, 148.4960632
34: -76.0547791, 53.7708549, -76.1679993, 53.8332748, -129.8880463, 129.9388580
35: -67.2489166, 58.3187103, -67.3627167, 58.4185295, -125.6674423, 125.6814270
36: -70.5261536, 58.6758194, -70.6548157, 58.7076378, -129.2337952, 129.3306274
37: -105.0175476, 48.8534012, -105.1511459, 48.8370895, -153.8546295, 154.0045471
38: -97.4842148, 69.2389526, -97.6506424, 69.2103348, -166.6945496, 166.8895874
39: -104.3733978, 58.3824806, -104.6255188, 58.4369049, -162.8103027, 163.0079956
40: -93.2671051, 39.6663818, -93.4317780, 39.6219101, -132.8890076, 133.0981598
41: -67.3097534, 40.1964836, -67.3840179, 40.1819611, -107.4917145, 107.5804977
42: -53.0589447, 37.8753586, -53.1211395, 37.8851204, -90.9440613, 90.9964981

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 966

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 599

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9482609, upper bound: 100.0416960
time: 82.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9482609, upper bound: 100.0416960
time: 79.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -79.6509247, 61.0274391, -79.8659668, 61.2097282, -140.8606567, 140.8934021
1: -49.8613434, 50.9941673, -49.9886398, 51.1513824, -101.0127258, 100.9828033
2: -42.3081322, 46.4642792, -42.4090881, 46.6334457, -88.9415741, 88.8733597
3: -47.1295776, 57.6568985, -47.3168907, 57.8911285, -105.0207062, 104.9737778
4: -51.5585823, 52.0512199, -51.7582474, 52.2548256, -103.8134079, 103.8094635
5: -50.6035118, 58.2740669, -50.7610092, 58.5223503, -109.1258621, 109.0350723
6: -74.6729965, 37.7507248, -74.9767609, 37.9216919, -112.5946884, 112.7274857
7: -64.7843475, 54.5326538, -64.9225006, 54.7614632, -119.5458069, 119.4551544
8: -59.9486389, 58.5724792, -60.1175270, 58.8440056, -118.7926331, 118.6900024
9: -45.8781052, 49.8638535, -46.0853577, 49.9548454, -95.8329468, 95.9492111
10: -75.0702362, 63.4163551, -75.3575897, 63.5680161, -138.6382446, 138.7739258
11: -77.7981415, 45.8212585, -78.0246277, 46.1830177, -123.9811554, 123.8458710
12: -79.3446503, 53.9525871, -79.7539825, 54.0686417, -133.4132843, 133.7065735
13: -61.7745247, 77.4737244, -62.3195457, 77.7716370, -139.5461578, 139.7932739
14: -118.9324799, 37.8486633, -119.1845169, 38.1153069, -157.0477905, 157.0331726
15: -57.9838791, 61.6709099, -58.1413994, 61.8683586, -119.8522339, 119.8123093
16: -85.2843094, 57.8731766, -85.4768524, 58.1132088, -143.3974915, 143.3500061
17: -119.3998032, 53.8288956, -119.6076050, 54.2036133, -173.6034241, 173.4364929
18: -74.7826538, 43.1730881, -75.0626907, 43.5007858, -118.2834396, 118.2357635
19: -59.2080536, 27.4917450, -59.3867874, 27.7552032, -86.9632416, 86.8785248
20: -50.8958054, 35.1436424, -51.0376816, 35.3299484, -86.2257538, 86.1813202
21: -71.3083572, 35.6606369, -71.5779724, 36.0269623, -107.3353119, 107.2386093
22: -68.4420013, 41.5978813, -68.6734161, 41.8689423, -110.3109436, 110.2712936
23: -54.8856430, 38.2449646, -55.0850029, 38.6098862, -93.4955292, 93.3299713
24: -58.3790932, 32.7376175, -58.6135864, 33.1051178, -91.4842072, 91.3512039
25: -50.7157784, 44.6896172, -50.9345093, 45.0234756, -95.7392426, 95.6241302
26: -86.1189575, 55.2165680, -86.4393463, 55.4945412, -141.6134949, 141.6559143
27: -69.0088425, 32.1413231, -69.2495880, 32.5155640, -101.5244064, 101.3909073
28: -55.0989151, 44.4992142, -55.2722359, 44.8979416, -99.9968567, 99.7714539
29: -69.6649017, 38.9187317, -69.8753052, 39.2986832, -108.9635773, 108.7940369
30: -68.8541412, 47.7219391, -69.0956421, 48.1652603, -117.0194016, 116.8175812
31: -70.4814529, 33.5858536, -70.7184143, 33.8795929, -104.3610458, 104.3042679
32: -65.0827789, 42.2847214, -65.5558701, 42.5229301, -107.6056976, 107.8405914
33: -84.5283661, 63.8001556, -84.9474335, 63.9956589, -148.5240173, 148.7475891
34: -76.0547791, 53.7708549, -76.3844910, 53.9104004, -129.9651794, 130.1553345
35: -67.2489166, 58.3187103, -67.5485306, 58.4653549, -125.7142639, 125.8672409
36: -70.5261536, 58.6758194, -70.9614563, 58.8476944, -129.3738403, 129.6372681
37: -105.0175476, 48.8534012, -105.4222565, 48.9017181, -153.9192352, 154.2756653
38: -97.4842148, 69.2389526, -98.1419373, 69.5124054, -166.9966125, 167.3808746
39: -104.3733978, 58.3824806, -105.0695877, 58.5585136, -162.9319153, 163.4520721
40: -93.2671051, 39.6663818, -93.6949387, 39.7913666, -133.0584717, 133.3613281
41: -67.3097534, 40.1964836, -67.6874084, 40.3490067, -107.6587601, 107.8838882
42: -53.0589447, 37.8753586, -53.3663902, 37.9972649, -91.0562057, 91.2417450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=501, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 599

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9486760, upper bound: 100.0416960
time: 88.40 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9486760, upper bound: 100.0416960
time: 89.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -79.7922974, 61.1305580, -79.7625046, 61.0924797, -140.8847656, 140.8930664
1: -49.9342499, 51.0999146, -49.9350357, 51.0200500, -100.9542999, 101.0349503
2: -42.3522110, 46.5539017, -42.3681221, 46.5099945, -88.8621979, 88.9220276
3: -47.2437897, 57.7739792, -47.2370529, 57.7336464, -104.9774323, 105.0110245
4: -51.6236649, 52.1660233, -51.7097588, 52.1529541, -103.7766113, 103.8757782
5: -50.6650352, 58.3803139, -50.7115898, 58.3776817, -109.0427094, 109.0919037
6: -74.9132233, 37.9033432, -74.7413177, 37.8203659, -112.7335815, 112.6446609
7: -64.8545303, 54.6523094, -64.8697968, 54.5374413, -119.3919678, 119.5220947
8: -59.9965973, 58.7260551, -60.0802155, 58.6692276, -118.6658096, 118.8062668
9: -45.9994659, 49.9233704, -45.9775429, 49.8931313, -95.8925934, 95.9009094
10: -75.2652130, 63.5185051, -75.1684265, 63.4567375, -138.7219543, 138.6869354
11: -77.9825439, 46.0493011, -77.8988876, 45.9169960, -123.8995209, 123.9481888
12: -79.6376038, 54.0371971, -79.2777557, 54.0376854, -133.6752930, 133.3149414
13: -62.1285210, 77.5963745, -61.9228821, 77.7014999, -139.8300018, 139.5192566
14: -119.1229935, 38.0419006, -118.9141998, 37.9245224, -157.0475006, 156.9561005
15: -58.0606384, 61.7902069, -57.9616928, 61.7499237, -119.8105621, 119.7518997
16: -85.4297028, 58.0375786, -85.3502350, 57.8386040, -143.2682953, 143.3878174
17: -119.6270676, 54.1281815, -119.3195496, 53.9023666, -173.5294342, 173.4477234
18: -74.8962173, 43.3676300, -74.8974915, 43.3114357, -118.2076569, 118.2651138
19: -59.3522530, 27.6712360, -59.3175774, 27.5730133, -86.9252625, 86.9888153
20: -50.9755020, 35.2699585, -50.9476852, 35.2099800, -86.1854630, 86.2176437
21: -71.5250931, 35.9117928, -71.4652100, 35.7637978, -107.2888870, 107.3769989
22: -68.6288910, 41.7928963, -68.4179840, 41.6635818, -110.2924728, 110.2108765
23: -55.0233345, 38.4554291, -55.0044289, 38.3887024, -93.4120331, 93.4598541
24: -58.5827141, 32.9955597, -58.5184975, 32.8335800, -91.4162903, 91.5140457
25: -50.8849220, 44.9056625, -50.8078499, 44.8036652, -95.6885834, 95.7135162
26: -86.2405853, 55.3887482, -86.1175461, 55.3298569, -141.5704346, 141.5062866
27: -69.1205139, 32.4002075, -69.1501999, 32.2571945, -101.3777084, 101.5504074
28: -55.2387848, 44.7744331, -55.2041321, 44.6291771, -99.8679504, 99.9785614
29: -69.9076538, 39.1759796, -69.6791229, 39.0467453, -108.9543991, 108.8551025
30: -69.0892029, 48.0505943, -68.9929047, 47.7204514, -116.8096466, 117.0435028
31: -70.6583939, 33.7816162, -70.6240311, 33.6725693, -104.3309631, 104.4056473
32: -65.4632568, 42.4925308, -65.1471710, 42.4544334, -107.9176941, 107.6397018
33: -84.7767639, 63.8865547, -84.7127838, 63.8837547, -148.6605225, 148.5993347
34: -76.2682495, 53.8454170, -76.1631470, 53.8579330, -130.1261902, 130.0085449
35: -67.4315033, 58.3634491, -67.3660126, 58.4207840, -125.8522873, 125.7294464
36: -70.8257446, 58.8151588, -70.5879288, 58.8336105, -129.6593475, 129.4030762
37: -105.2837982, 48.9163437, -105.1441574, 48.8726578, -154.1564636, 154.0605011
38: -97.9661865, 69.5398254, -97.5737991, 69.4491272, -167.4153137, 167.1136169
39: -104.8137589, 58.5015297, -104.6234970, 58.5093765, -163.3231354, 163.1250305
40: -93.5274811, 39.8255119, -93.4325180, 39.6320152, -133.1594849, 133.2580261
41: -67.6105652, 40.3602715, -67.3812866, 40.2640305, -107.8745880, 107.7415543
42: -53.3018227, 37.9832687, -53.1183090, 37.8975601, -91.1993866, 91.1015778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=500, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 599

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0301603
time: 89.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0301603
time: 87.98 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -79.7893372, 61.1215096, -80.0072937, 61.1525040, -140.9418335, 141.1287994
1: -49.9319344, 51.0884438, -50.1040878, 51.0782623, -101.0101929, 101.1925278
2: -42.3499603, 46.5452347, -42.5958557, 46.5737686, -88.9237289, 89.1410828
3: -47.2416763, 57.7633934, -47.4850693, 57.8213158, -105.0629883, 105.2484589
4: -51.6211052, 52.1631470, -51.8069229, 52.2592735, -103.8803787, 103.9700699
5: -50.6630478, 58.3721085, -50.9585419, 58.4687958, -109.1318436, 109.3306427
6: -74.9093857, 37.8825302, -74.9180603, 37.9120331, -112.8214188, 112.8005829
7: -64.8505249, 54.6220169, -65.2633362, 54.6105423, -119.4610672, 119.8853531
8: -59.9944000, 58.7155838, -60.3456650, 58.7635880, -118.7579880, 119.0612488
9: -45.9942169, 49.9214630, -46.0578728, 50.0140457, -96.0082550, 95.9793396
10: -75.2606735, 63.5161896, -75.2896729, 63.7288132, -138.9894867, 138.8058624
11: -77.9765778, 46.0032234, -78.1362610, 45.9838219, -123.9603882, 124.1394806
12: -79.6378784, 54.0359764, -79.5300903, 54.6968536, -134.3347168, 133.5660706
13: -62.1032143, 77.5934677, -62.0260849, 78.0719757, -140.1751862, 139.6195374
14: -119.1214447, 38.0401688, -119.2425461, 38.3719025, -157.4933472, 157.2827148
15: -58.0245209, 61.7855530, -58.1131592, 62.1289902, -120.1535034, 119.8987122
16: -85.4248734, 58.0001450, -85.7859039, 57.9566040, -143.3814545, 143.7860413
17: -119.6298523, 54.1226883, -119.6194458, 54.4490051, -174.0788574, 173.7421265
18: -74.8911133, 43.3622246, -75.0436554, 43.4907684, -118.3818817, 118.4058762
19: -59.3496475, 27.6697083, -59.3943405, 27.6615334, -87.0111847, 87.0640488
20: -50.9721985, 35.2681236, -51.0576935, 35.3422623, -86.3144608, 86.3258209
21: -71.5218353, 35.9029808, -71.5678711, 35.8789215, -107.4007492, 107.4708481
22: -68.5988464, 41.7881165, -68.5996399, 42.1936226, -110.7924500, 110.3877563
23: -55.0219917, 38.4455643, -55.1225586, 38.4360504, -93.4580383, 93.5681229
24: -58.5799980, 32.9862061, -58.6838608, 32.8865356, -91.4665375, 91.6700592
25: -50.8717003, 44.9011841, -50.9018364, 44.9952812, -95.8669815, 95.8030243
26: -86.2277908, 55.3839226, -86.4052277, 55.9711113, -142.1988983, 141.7891541
27: -69.1182556, 32.3871536, -69.3116913, 32.2975693, -101.4158173, 101.6988449
28: -55.2375526, 44.7722092, -55.3231087, 44.6945648, -99.9321136, 100.0953217
29: -69.9036713, 39.1738205, -69.8872299, 39.5142746, -109.4179459, 109.0610504
30: -69.0843582, 48.0386505, -69.3193512, 47.8827705, -116.9671326, 117.3580017
31: -70.6559601, 33.7725906, -70.7630920, 33.7719688, -104.4279327, 104.5356827
32: -65.4596634, 42.4901428, -65.2643433, 42.6332970, -108.0929413, 107.7544785
33: -84.7731400, 63.8827019, -84.9434280, 64.0278778, -148.8010254, 148.8261108
34: -76.2647171, 53.8417091, -76.3495941, 53.9425125, -130.2072296, 130.1912842
35: -67.4269409, 58.3611412, -67.4874496, 58.4912720, -125.9182129, 125.8485870
36: -70.8047028, 58.8140717, -70.6856537, 59.1670990, -129.9718018, 129.4997253
37: -105.2765274, 48.9141769, -105.2979813, 48.9662132, -154.2427368, 154.2121582
38: -97.9608536, 69.5371552, -97.7655334, 69.7908020, -167.7516479, 167.3026733
39: -104.8014297, 58.4998131, -104.7016449, 58.6134682, -163.4149017, 163.2014618
40: -93.5238419, 39.8283119, -93.7991791, 39.8035393, -133.3273773, 133.6274872
41: -67.6074524, 40.3471680, -67.5319138, 40.3528976, -107.9603424, 107.8790741
42: -53.2995682, 37.9710426, -53.2511024, 38.0105095, -91.3100739, 91.2221451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=500, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 966

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 599

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0227649
time: 129.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0227649
time: 84.61 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 216.88 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 216.88
Output dim: 13, lower bound: -99.9482609, upper bound: 100.0416960
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 216.88
Output dim: 13, lower bound: -99.9482609, upper bound: 100.0416960
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 216.88
Output dim: 13, lower bound: -99.9486760, upper bound: 100.0416960
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 216.88
Output dim: 13, lower bound: -99.9486760, upper bound: 100.0416960
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 216.88
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0301603
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 216.88
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0301603
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 216.88
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0227649
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 216.88
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0227649
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 216.88
Output dim: 13, lower bound: -99.9403398, upper bound: 100.0332456
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 216.88
Output dim: 13, lower bound: -99.9403398, upper bound: 100.0332456
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 216.88
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0209140
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 216.88
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0134171
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 216.88
Output dim: 13, lower bound: -99.9236663, upper bound: 100.0416960
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 216.88
Output dim: 13, lower bound: -99.9236663, upper bound: 100.0416960
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 216.88
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0301603
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 216.88
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0227649
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 216.88
Output dim: 13, lower bound: -100.0332453, upper bound: 100.0332456
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 216.88
Output dim: 13, lower bound: -100.0332453, upper bound: 100.0332456
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 216.88
Output dim: 13, lower bound: -100.0134170, upper bound: 100.0209140
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 216.88
Output dim: 13, lower bound: -99.9047529, upper bound: 100.0134171
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=139.97683715820312
rel_dist={13: [-100.15777163255231, 100.15777161498153]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 966

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1725

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.9765608, upper bound: 96.0393283
time: 84.00 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.9765608, upper bound: 96.0413672
time: 85.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 170.03 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 170.03
Output dim: 13, lower bound: -95.9765608, upper bound: 96.0393283
IS_A2, status: Status.UNKNOWN, split count: 1, time: 170.03
Output dim: 13, lower bound: -95.9765608, upper bound: 96.0413672

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -79.7487793, 61.0640106, -79.8687210, 61.1569901, -140.9057617, 140.9327240
1: -49.9181595, 51.0219803, -49.9900436, 51.0828209, -101.0009766, 101.0120239
2: -42.3431664, 46.4915161, -42.4083099, 46.5840912, -88.9272614, 88.8998184
3: -47.1949654, 57.6961174, -47.2858047, 57.8382263, -105.0331726, 104.9819183
4: -51.5967598, 52.1070290, -51.7139282, 52.2412109, -103.8379669, 103.8209457
5: -50.6425476, 58.3190613, -50.7361183, 58.4845238, -109.1270523, 109.0551682
6: -74.7151260, 37.8987808, -74.8050613, 38.0051956, -112.7203217, 112.7038422
7: -64.8565826, 54.5657806, -64.9466629, 54.6797981, -119.5363770, 119.5124435
8: -59.9942093, 58.6274529, -60.1014023, 58.7790947, -118.7733002, 118.7288513
9: -45.9228210, 49.9003677, -46.0126495, 49.9526749, -95.8754883, 95.9130173
10: -75.1153793, 63.4623795, -75.2174683, 63.5394211, -138.6548004, 138.6798401
11: -77.9448547, 45.8540573, -78.1036987, 45.9701920, -123.9150467, 123.9577560
12: -79.3902588, 54.0684700, -79.5102539, 54.1538315, -133.5440979, 133.5787201
13: -61.8165207, 77.5958405, -61.9737320, 77.8459625, -139.6624756, 139.5695496
14: -119.0646591, 37.8752098, -119.2368774, 37.9484520, -157.0131073, 157.1120911
15: -58.0404930, 61.7140465, -58.1230164, 61.8205719, -119.8610611, 119.8370667
16: -85.3749237, 57.9256363, -85.4829559, 58.0030174, -143.3779297, 143.4085999
17: -119.5854034, 53.8654747, -119.7258530, 53.9585648, -173.5439606, 173.5913239
18: -74.8276825, 43.2151604, -75.0205078, 43.3310471, -118.1587219, 118.2356720
19: -59.3319778, 27.5093632, -59.4791298, 27.5831699, -86.9151459, 86.9884872
20: -50.9427338, 35.1749802, -51.0371017, 35.2376442, -86.1803741, 86.2120819
21: -71.4757538, 35.6804161, -71.6366425, 35.7796021, -107.2553558, 107.3170547
22: -68.5911102, 41.6195793, -68.7263870, 41.6897125, -110.2808228, 110.3459625
23: -55.0032005, 38.2653656, -55.1725960, 38.3901062, -93.3933029, 93.4379578
24: -58.5453110, 32.7553902, -58.7009659, 32.8461189, -91.3914261, 91.4563599
25: -50.8489380, 44.7137146, -50.9839783, 44.8189316, -95.6678696, 95.6976929
26: -86.1758575, 55.2631073, -86.3962173, 55.3712692, -141.5471191, 141.6593323
27: -69.0784912, 32.1684456, -69.2772369, 32.2627296, -101.3412170, 101.4456787
28: -55.2288132, 44.5273552, -55.4027100, 44.6340828, -99.8628998, 99.9300613
29: -69.8836975, 38.9489441, -70.0116425, 39.0531502, -108.9368439, 108.9605713
30: -69.0528107, 47.7474632, -69.1789322, 47.8457298, -116.8985443, 116.9263916
31: -70.6199036, 33.6073875, -70.8057404, 33.6938972, -104.3137970, 104.4131317
32: -65.1272736, 42.4701309, -65.2329254, 42.5757523, -107.7030258, 107.7030487
33: -84.5923157, 63.8433380, -84.7604370, 63.9834747, -148.5757751, 148.6037750
34: -76.0970917, 53.8452415, -76.2211304, 53.9356194, -130.0327148, 130.0663757
35: -67.2969971, 58.3604050, -67.4207993, 58.4743195, -125.7713165, 125.7812042
36: -70.5548553, 58.8514099, -70.6823425, 58.9568176, -129.5116730, 129.5337524
37: -105.0800171, 48.9347000, -105.2369232, 49.0203667, -154.1003876, 154.1716309
38: -97.5279846, 69.5440674, -97.6948242, 69.6797333, -167.2077026, 167.2388916
39: -104.4244690, 58.5167274, -104.6537247, 58.6807327, -163.1051636, 163.1704407
40: -93.3119507, 39.8368034, -93.4761887, 39.9366493, -133.2485962, 133.3129883
41: -67.3457642, 40.3328094, -67.4441986, 40.3877487, -107.7335129, 107.7770081
42: -53.0912361, 37.9546967, -53.1724930, 38.0135002, -91.1047287, 91.1271896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=505, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 599

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.9135111, upper bound: 95.9974173
time: 84.75 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.9346687, upper bound: 95.9974173
time: 87.15 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -79.9259949, 61.1727066, -79.9322968, 61.1749992, -141.1009827, 141.1049957
1: -50.0214767, 51.0972214, -50.0264587, 51.0994186, -101.1208954, 101.1236801
2: -42.4468842, 46.5979233, -42.4515839, 46.6003189, -89.0472031, 89.0495071
3: -47.3457489, 57.8539467, -47.3504753, 57.8569221, -105.2026672, 105.2044220
4: -51.7908325, 52.2536888, -51.7971687, 52.2559128, -104.0467453, 104.0508499
5: -50.8004227, 58.5069427, -50.8058548, 58.5101509, -109.3105621, 109.3127975
6: -74.8402176, 38.0292740, -74.8450012, 38.0323639, -112.8725739, 112.8742752
7: -64.9918976, 54.6998787, -64.9978027, 54.7028618, -119.6947632, 119.6976776
8: -60.1644554, 58.7993279, -60.1701736, 58.8022690, -118.9667206, 118.9694824
9: -46.0514870, 49.9684982, -46.0570450, 49.9714699, -96.0229568, 96.0255432
10: -75.2520905, 63.5637894, -75.2558441, 63.5684547, -138.8205414, 138.8196411
11: -78.1283188, 46.0436478, -78.1327515, 46.0498619, -124.1781616, 124.1763992
12: -79.5462112, 54.1870537, -79.5503387, 54.1906891, -133.7369080, 133.7373810
13: -62.0870476, 77.8647766, -62.0968399, 77.8675690, -139.9545898, 139.9616089
14: -119.2738800, 37.9789505, -119.2795563, 37.9859467, -157.2598114, 157.2584991
15: -58.1551170, 61.8371696, -58.1585426, 61.8398018, -119.9949188, 119.9957047
16: -85.5190735, 58.0420189, -85.5238647, 58.0485306, -143.5675964, 143.5658875
17: -119.7585602, 53.9816208, -119.7640457, 53.9843140, -173.7428741, 173.7456665
18: -75.0399017, 43.4098816, -75.0433502, 43.4166412, -118.4565353, 118.4532318
19: -59.4921799, 27.6346169, -59.4959450, 27.6386223, -87.1307983, 87.1305618
20: -51.0509300, 35.2731628, -51.0534897, 35.2792168, -86.3301468, 86.3266525
21: -71.6523132, 35.8493690, -71.6566620, 35.8552094, -107.5075226, 107.5060272
22: -68.7488556, 41.7358322, -68.7523041, 41.7401581, -110.4889984, 110.4881363
23: -55.1863670, 38.4782410, -55.1893845, 38.4853020, -93.6716690, 93.6676025
24: -58.7179298, 32.9108086, -58.7222176, 32.9161415, -91.6340561, 91.6330185
25: -50.9988518, 44.8870049, -51.0030861, 44.8926697, -95.8915253, 95.8900909
26: -86.4245071, 55.4399338, -86.4291458, 55.4459991, -141.8705139, 141.8690796
27: -69.2953949, 32.3295975, -69.2984467, 32.3357048, -101.6310883, 101.6280365
28: -55.4152260, 44.7092667, -55.4179688, 44.7153473, -100.1305695, 100.1272202
29: -70.0324402, 39.1259079, -70.0360794, 39.1318436, -109.1642838, 109.1619873
30: -69.1942291, 47.9057159, -69.1981964, 47.9109879, -117.1052170, 117.1039124
31: -70.8217773, 33.7461510, -70.8258820, 33.7503815, -104.5721588, 104.5720367
32: -65.2768860, 42.5949936, -65.2816010, 42.5991631, -107.8760529, 107.8765869
33: -84.8492432, 64.0083923, -84.8601227, 64.0112915, -148.8605347, 148.8685150
34: -76.2750244, 53.9450912, -76.2847748, 53.9482727, -130.2232971, 130.2298584
35: -67.4815979, 58.4920540, -67.4911652, 58.4941788, -125.9757767, 125.9832153
36: -70.7426910, 58.9697609, -70.7513123, 58.9719086, -129.7145996, 129.7210693
37: -105.3041153, 49.0407448, -105.3149109, 49.0427704, -154.3468933, 154.3556519
38: -97.7700119, 69.6954193, -97.7796326, 69.6980591, -167.4680634, 167.4750519
39: -104.7931595, 58.6915627, -104.8058701, 58.6932755, -163.4864197, 163.4974213
40: -93.5573578, 39.9509087, -93.5651703, 39.9524155, -133.5097656, 133.5160828
41: -67.4837570, 40.4042625, -67.4885941, 40.4072113, -107.8909607, 107.8928452
42: -53.1949043, 38.0324821, -53.1977272, 38.0357170, -91.2306213, 91.2302094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=505, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 966

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 599

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.9135111, upper bound: 95.9994131
time: 131.59 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.9994129, upper bound: 95.9994131
time: 100.14 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 234.18 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 234.18
Output dim: 13, lower bound: -95.9135111, upper bound: 95.9974173
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 234.18
Output dim: 13, lower bound: -95.9346687, upper bound: 95.9974173
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 234.18
Output dim: 13, lower bound: -95.9135111, upper bound: 95.9994131
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 234.18
Output dim: 13, lower bound: -95.9994129, upper bound: 95.9994131

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -79.7131958, 61.0527382, -79.7814636, 61.1287804, -140.8419647, 140.8341980
1: -49.9014893, 51.0128326, -49.9490662, 51.0598679, -100.9613495, 100.9618988
2: -42.3296509, 46.4803085, -42.3747864, 46.5563354, -88.8859787, 88.8550949
3: -47.1732254, 57.6786499, -47.2314911, 57.7947578, -104.9679642, 104.9101410
4: -51.5865173, 52.0816383, -51.6888008, 52.1797028, -103.7662048, 103.7704391
5: -50.6285019, 58.3005142, -50.7015419, 58.4384270, -109.0669250, 109.0020599
6: -74.6939163, 37.8522720, -74.7525406, 37.8887711, -112.5826645, 112.6048126
7: -64.8348236, 54.5544968, -64.8927612, 54.6516647, -119.4864883, 119.4472580
8: -59.9851799, 58.6053200, -60.0789566, 58.7242699, -118.7094421, 118.6842804
9: -45.9104958, 49.8861656, -45.9820976, 49.9173088, -95.8277969, 95.8682632
10: -75.1018829, 63.4424286, -75.1839752, 63.4900055, -138.5918884, 138.6264038
11: -77.8855972, 45.8430290, -77.9565125, 45.9429016, -123.8284988, 123.7995453
12: -79.3731384, 54.0359955, -79.4676971, 54.0733032, -133.4464417, 133.5036926
13: -61.8028755, 77.5544891, -61.9402542, 77.7489166, -139.5517883, 139.4947510
14: -119.0008850, 37.8659363, -119.0783920, 37.9256363, -156.9265137, 156.9443207
15: -58.0275650, 61.6961327, -58.0907936, 61.7761650, -119.8037262, 119.7869263
16: -85.3375854, 57.9098206, -85.3904114, 57.9636459, -143.3012238, 143.3002319
17: -119.4998932, 53.8512001, -119.5135498, 53.9233208, -173.4232025, 173.3647461
18: -74.8081818, 43.2039146, -74.9722366, 43.3031464, -118.1113281, 118.1761398
19: -59.2776337, 27.5017490, -59.3443336, 27.5643139, -86.8419495, 86.8460846
20: -50.9236984, 35.1639786, -50.9895210, 35.2103348, -86.1340332, 86.1535034
21: -71.4229584, 35.6708107, -71.5046082, 35.7557373, -107.1786957, 107.1754150
22: -68.5395508, 41.6106796, -68.5979462, 41.6676979, -110.2072449, 110.2086258
23: -54.9507027, 38.2576523, -55.0426788, 38.3710556, -93.3217621, 93.3003311
24: -58.4849548, 32.7481232, -58.5501671, 32.8281097, -91.3130493, 91.2982864
25: -50.8048325, 44.7029533, -50.8737793, 44.7925835, -95.5974121, 95.5767365
26: -86.1511612, 55.2452126, -86.3345032, 55.3271027, -141.4782410, 141.5797119
27: -69.0413284, 32.1616287, -69.1859283, 32.2458305, -101.2871552, 101.3475571
28: -55.1623383, 44.5171127, -55.2383270, 44.6090546, -99.7713928, 99.7554321
29: -69.8032990, 38.9400253, -69.8121643, 39.0312347, -108.8345337, 108.7521820
30: -68.9949188, 47.7365265, -69.0345764, 47.8186989, -116.8136139, 116.7711029
31: -70.5591583, 33.5995941, -70.6546326, 33.6746063, -104.2337646, 104.2542267
32: -65.1090012, 42.4315376, -65.1879120, 42.4786911, -107.5876923, 107.6194458
33: -84.5707092, 63.8215103, -84.7069397, 63.9300842, -148.5007782, 148.5284424
34: -76.0781097, 53.8299141, -76.1742783, 53.8976135, -129.9757080, 130.0041809
35: -67.2770996, 58.3498535, -67.3715210, 58.4482346, -125.7253342, 125.7213745
36: -70.5403290, 58.8152313, -70.6467667, 58.8678665, -129.4081879, 129.4620056
37: -105.0526886, 48.8835907, -105.1690216, 48.8967972, -153.9494781, 154.0526123
38: -97.5094376, 69.4682541, -97.6493607, 69.4906006, -167.0000305, 167.1176147
39: -104.4025650, 58.4652328, -104.5996399, 58.5541153, -162.9566803, 163.0648804
40: -93.2934570, 39.7702942, -93.4303360, 39.7707214, -133.0641785, 133.2006226
41: -67.3255615, 40.2983856, -67.3943024, 40.3012848, -107.6268311, 107.6926880
42: -53.0774612, 37.9268951, -53.1382332, 37.9473114, -91.0247726, 91.0651245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 966

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 601

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8726391, upper bound: 95.9343257
time: 82.57 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8726391, upper bound: 95.9554049
time: 79.85 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -79.7208328, 61.0568275, -79.9312210, 61.2252808, -140.9461060, 140.9880371
1: -49.9059029, 51.0153503, -50.0221710, 51.1488647, -101.0547638, 101.0375214
2: -42.3276825, 46.4854736, -42.4346008, 46.6716690, -88.9993515, 88.9200745
3: -47.1737671, 57.6869011, -47.3197784, 57.9235725, -105.0973206, 105.0066757
4: -51.5887680, 52.0958595, -51.7620850, 52.3048019, -103.8935699, 103.8579407
5: -50.6332970, 58.3098946, -50.7724228, 58.5628395, -109.1961365, 109.0823059
6: -74.7051849, 37.8844223, -75.0233078, 38.0237427, -112.7289124, 112.9077301
7: -64.8308563, 54.5552635, -64.9499359, 54.7775116, -119.6083679, 119.5051880
8: -59.9729805, 58.6156273, -60.1043053, 58.8614464, -118.8344269, 118.7199326
9: -45.9156494, 49.8926430, -46.1383171, 49.9940262, -95.9096680, 96.0309601
10: -75.1071854, 63.4477425, -75.3159943, 63.6172791, -138.7244568, 138.7637329
11: -77.9048462, 45.8392372, -78.1222992, 46.1608887, -124.0657349, 123.9615326
12: -79.3788605, 54.0265198, -79.7684402, 54.1407242, -133.5195923, 133.7949524
13: -61.8043404, 77.5618134, -62.2853508, 77.8489685, -139.6533051, 139.8471680
14: -119.0370865, 37.8670197, -119.2896881, 38.1716309, -157.2087097, 157.1567078
15: -58.0280800, 61.7005501, -58.1540489, 61.8638687, -119.8919525, 119.8545990
16: -85.3510590, 57.9114075, -85.5602798, 58.1238174, -143.4748840, 143.4716797
17: -119.5589142, 53.8454895, -119.7823486, 54.2810364, -173.8399506, 173.6278381
18: -74.8092194, 43.2091141, -75.0718842, 43.5061646, -118.3153687, 118.2809982
19: -59.3184776, 27.4997101, -59.5157471, 27.7455559, -87.0640259, 87.0154572
20: -50.9264297, 35.1677170, -51.0583649, 35.3709412, -86.2973709, 86.2260818
21: -71.4453201, 35.6688232, -71.6673584, 35.9704208, -107.4157410, 107.3361816
22: -68.5708923, 41.6096649, -68.7712555, 41.8764114, -110.4473038, 110.3809128
23: -54.9823837, 38.2560921, -55.1955528, 38.6122208, -93.5946045, 93.4516449
24: -58.5263214, 32.7497787, -58.7486153, 33.1028023, -91.6291046, 91.4983826
25: -50.8340378, 44.7050247, -51.0227547, 45.0384293, -95.8724670, 95.7277832
26: -86.1589584, 55.2511063, -86.4591370, 55.5204964, -141.6794586, 141.7102356
27: -69.0619583, 32.1638184, -69.3178177, 32.4885788, -101.5505295, 101.4816360
28: -55.2122459, 44.5179138, -55.4307785, 44.8810196, -100.0932617, 99.9486923
29: -69.8590927, 38.9343987, -70.0587463, 39.2914734, -109.1505661, 108.9931412
30: -69.0208969, 47.7357483, -69.1986237, 48.1085434, -117.1294403, 116.9343719
31: -70.6037598, 33.5979309, -70.8498077, 33.8920059, -104.4957428, 104.4477386
32: -65.1160278, 42.4575806, -65.5245895, 42.6156425, -107.7316742, 107.9821625
33: -84.5788727, 63.8224792, -85.0138245, 63.9898720, -148.5687408, 148.8363037
34: -76.0827332, 53.8227882, -76.3537827, 53.9386444, -130.0213776, 130.1765442
35: -67.2850418, 58.3458252, -67.5681458, 58.4677811, -125.7528229, 125.9139633
36: -70.5411835, 58.8202744, -70.9037323, 58.9304848, -129.4716644, 129.7239990
37: -105.0616684, 48.9057617, -105.5702057, 49.0117455, -154.0734100, 154.4759674
38: -97.5059357, 69.4990921, -98.0055389, 69.6710892, -167.1770172, 167.5046234
39: -104.4062653, 58.4802170, -105.0243759, 58.6511612, -163.0574341, 163.5045929
40: -93.2938232, 39.8201256, -93.8028412, 39.9485207, -133.2423248, 133.6229553
41: -67.3356171, 40.3209152, -67.7122040, 40.4165115, -107.7521286, 108.0331192
42: -53.0851669, 37.9401474, -53.3529129, 38.0434570, -91.1286240, 91.2930527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 966

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 601

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8924629, upper bound: 95.9341877
time: 82.91 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8924629, upper bound: 95.9551997
time: 94.07 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -79.8903885, 61.1614075, -79.8450546, 61.1467896, -141.0371704, 141.0064545
1: -50.0048065, 51.0880013, -49.9854660, 51.0764999, -101.0813065, 101.0734634
2: -42.4333534, 46.5867195, -42.4180603, 46.5725479, -89.0058975, 89.0047760
3: -47.3240128, 57.8364983, -47.2961617, 57.8134537, -105.1374664, 105.1326599
4: -51.7806396, 52.2283287, -51.7720566, 52.1944351, -103.9750748, 104.0003815
5: -50.7863693, 58.4884071, -50.7712746, 58.4640274, -109.2503967, 109.2596741
6: -74.8190231, 37.9827957, -74.7924805, 37.9159470, -112.7349701, 112.7752762
7: -64.9701538, 54.6885719, -64.9439087, 54.6747437, -119.6448975, 119.6324768
8: -60.1554375, 58.7772293, -60.1477280, 58.7474060, -118.9028473, 118.9249496
9: -46.0391731, 49.9542961, -46.0264206, 49.9360886, -95.9752579, 95.9807129
10: -75.2385864, 63.5438232, -75.2223053, 63.5190468, -138.7576294, 138.7661285
11: -78.0690536, 46.0326233, -77.9855957, 46.0225372, -124.0915833, 124.0182190
12: -79.5290070, 54.1545448, -79.5077515, 54.1101608, -133.6391602, 133.6622925
13: -62.0734329, 77.8234253, -62.0633736, 77.7705383, -139.8439636, 139.8867950
14: -119.2100906, 37.9697266, -119.1210403, 37.9630737, -157.1731567, 157.0907593
15: -58.1421509, 61.8192444, -58.1262970, 61.7953644, -119.9375153, 119.9455261
16: -85.4816742, 58.0261917, -85.4313049, 58.0091324, -143.4908142, 143.4574890
17: -119.6730118, 53.9673195, -119.5517654, 53.9490700, -173.6220703, 173.5190735
18: -75.0203781, 43.3986511, -74.9950562, 43.3887520, -118.4091263, 118.3936996
19: -59.4378395, 27.6270008, -59.3611450, 27.6198044, -87.0576477, 86.9881439
20: -51.0319138, 35.2621613, -51.0059052, 35.2519341, -86.2838440, 86.2680664
21: -71.5995178, 35.8397980, -71.5246506, 35.8313637, -107.4308777, 107.3644409
22: -68.6972656, 41.7269516, -68.6238632, 41.7181282, -110.4153900, 110.3508148
23: -55.1338654, 38.4705124, -55.0594902, 38.4662781, -93.6001434, 93.5299988
24: -58.6575432, 32.9035225, -58.5714302, 32.8981552, -91.5556946, 91.4749451
25: -50.9547539, 44.8762665, -50.8929367, 44.8663254, -95.8210754, 95.7692032
26: -86.3997879, 55.4220314, -86.3674393, 55.4018288, -141.8016052, 141.7894592
27: -69.2581940, 32.3227768, -69.2071075, 32.3187866, -101.5769806, 101.5298767
28: -55.3487740, 44.6990967, -55.2536201, 44.6903267, -100.0390854, 99.9527130
29: -69.9520721, 39.1170311, -69.8365860, 39.1099739, -109.0620422, 108.9536133
30: -69.1363525, 47.8947792, -69.0538406, 47.8839684, -117.0203247, 116.9486160
31: -70.7610321, 33.7383652, -70.6747818, 33.7310944, -104.4921265, 104.4131317
32: -65.2586365, 42.5563736, -65.2365570, 42.5020752, -107.7607117, 107.7929230
33: -84.8276672, 63.9865837, -84.8067093, 63.9578629, -148.7855225, 148.7932892
34: -76.2560730, 53.9297485, -76.2379456, 53.9102592, -130.1663361, 130.1676941
35: -67.4617920, 58.4815025, -67.4419022, 58.4680710, -125.9298477, 125.9234009
36: -70.7281952, 58.9336014, -70.7157440, 58.8829498, -129.6111450, 129.6493378
37: -105.2768097, 48.9896507, -105.2469940, 48.9192734, -154.1960754, 154.2366486
38: -97.7514954, 69.6195984, -97.7342224, 69.5088654, -167.2603607, 167.3538208
39: -104.7712555, 58.6400566, -104.7518311, 58.5666809, -163.3379059, 163.3918915
40: -93.5388794, 39.8843842, -93.5193176, 39.7865143, -133.3253937, 133.4036865
41: -67.4635925, 40.3698044, -67.4386749, 40.3207550, -107.7843475, 107.8084793
42: -53.1811256, 38.0046654, -53.1634407, 37.9695358, -91.1506500, 91.1681061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 966

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 601

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8726391, upper bound: 95.9364468
time: 135.24 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.9375095, upper bound: 95.9574287
time: 91.45 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -79.8980103, 61.1655273, -79.9948120, 61.2432938, -141.1412964, 141.1603394
1: -50.0092621, 51.0905914, -50.0585632, 51.1654854, -101.1747284, 101.1491394
2: -42.4314041, 46.5918961, -42.4778824, 46.6878815, -89.1192856, 89.0697784
3: -47.3245506, 57.8447533, -47.3844719, 57.9422607, -105.2668076, 105.2292252
4: -51.7828674, 52.2425270, -51.8453178, 52.3194923, -104.1023560, 104.0878448
5: -50.7911758, 58.4978065, -50.8421898, 58.5884476, -109.3796234, 109.3399963
6: -74.8302612, 38.0149078, -75.0632782, 38.0509033, -112.8811646, 113.0781784
7: -64.9661560, 54.6893234, -65.0010834, 54.8005905, -119.7667465, 119.6904068
8: -60.1432877, 58.7875099, -60.1731377, 58.8845291, -119.0278015, 118.9606476
9: -46.0443573, 49.9607697, -46.1826630, 50.0127907, -96.0571442, 96.1434326
10: -75.2438965, 63.5491409, -75.3543549, 63.6463318, -138.8902283, 138.9034882
11: -78.0883026, 46.0288353, -78.1513519, 46.2405472, -124.3288422, 124.1801910
12: -79.5347595, 54.1450806, -79.8085556, 54.1775551, -133.7123108, 133.9536285
13: -62.0748901, 77.8307190, -62.4084816, 77.8705597, -139.9454346, 140.2391968
14: -119.2463455, 37.9708138, -119.3323288, 38.2091331, -157.4554596, 157.3031311
15: -58.1426315, 61.8236809, -58.1895370, 61.8830376, -120.0256653, 120.0132141
16: -85.4952316, 58.0278130, -85.6011276, 58.1693153, -143.6645508, 143.6289368
17: -119.7320786, 53.9616089, -119.8205490, 54.3068314, -174.0389099, 173.7821655
18: -75.0214462, 43.4038544, -75.0946503, 43.5918159, -118.6132660, 118.4984970
19: -59.4786644, 27.6249733, -59.5325050, 27.8010464, -87.2797089, 87.1574783
20: -51.0346336, 35.2659073, -51.0747452, 35.4125900, -86.4472198, 86.3406525
21: -71.6218872, 35.8378029, -71.6873627, 36.0460815, -107.6679688, 107.5251617
22: -68.7286148, 41.7258949, -68.7971725, 41.9268494, -110.6554642, 110.5230713
23: -55.1655350, 38.4689484, -55.2123756, 38.7074814, -93.8730164, 93.6813202
24: -58.6988983, 32.9051895, -58.7698250, 33.1728592, -91.8717499, 91.6750031
25: -50.9839401, 44.8783264, -51.0418816, 45.1121750, -96.0961151, 95.9202042
26: -86.4075928, 55.4279480, -86.4920197, 55.5952835, -142.0028687, 141.9199677
27: -69.2788544, 32.3249588, -69.3389740, 32.5615845, -101.8404388, 101.6639252
28: -55.3986702, 44.6998749, -55.4459801, 44.9622803, -100.3609467, 100.1458588
29: -70.0078583, 39.1113892, -70.0831451, 39.3702278, -109.3780823, 109.1945190
30: -69.1623383, 47.8939972, -69.2178345, 48.1738510, -117.3361893, 117.1118317
31: -70.8056793, 33.7367058, -70.8699036, 33.9485359, -104.7542114, 104.6066132
32: -65.2656708, 42.5824356, -65.5732727, 42.6389847, -107.9046478, 108.1557083
33: -84.8358002, 63.9875259, -85.1138229, 64.0176620, -148.8534241, 149.1013489
34: -76.2606354, 53.9226875, -76.4175262, 53.9512215, -130.2118530, 130.3402100
35: -67.4697266, 58.4774628, -67.6386719, 58.4876099, -125.9573364, 126.1161346
36: -70.7290955, 58.9386444, -70.9727707, 58.9455795, -129.6746826, 129.9114075
37: -105.2857361, 49.0118256, -105.6482925, 49.0341568, -154.3198853, 154.6601105
38: -97.7479935, 69.6504059, -98.0904388, 69.6893234, -167.4373169, 167.7408447
39: -104.7749329, 58.6550407, -105.1767120, 58.6636810, -163.4386139, 163.8317413
40: -93.5392609, 39.9341965, -93.8919220, 39.9642715, -133.5035095, 133.8261108
41: -67.4736176, 40.3923416, -67.7566071, 40.4359436, -107.9095459, 108.1489487
42: -53.1888275, 38.0179329, -53.3781128, 38.0656433, -91.2544708, 91.3960419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 966

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 601

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.9572234, upper bound: 95.9362924
time: 117.43 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8726391, upper bound: 95.9572235
time: 110.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 230.73 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 230.73
Output dim: 13, lower bound: -95.8726391, upper bound: 95.9343257
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 230.73
Output dim: 13, lower bound: -95.8726391, upper bound: 95.9554049
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 230.73
Output dim: 13, lower bound: -95.8924629, upper bound: 95.9341877
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 230.73
Output dim: 13, lower bound: -95.8924629, upper bound: 95.9551997
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 230.73
Output dim: 13, lower bound: -95.8726391, upper bound: 95.9364468
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 230.73
Output dim: 13, lower bound: -95.9375095, upper bound: 95.9574287
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 230.73
Output dim: 13, lower bound: -95.9572234, upper bound: 95.9362924
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 230.73
Output dim: 13, lower bound: -95.8726391, upper bound: 95.9572235

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -79.6226730, 61.0184174, -79.7450943, 61.1148872, -140.7375641, 140.7635193
1: -49.8480682, 50.9867935, -49.9275551, 51.0493889, -100.8974609, 100.9143524
2: -42.2973633, 46.4552460, -42.3613586, 46.5461121, -88.8434753, 88.8166046
3: -47.1123276, 57.6428223, -47.2069702, 57.7801590, -104.8924866, 104.8497849
4: -51.5504532, 52.0314827, -51.6744041, 52.1592751, -103.7097321, 103.7058868
5: -50.5923386, 58.2591133, -50.6870575, 58.4213333, -109.0136719, 108.9461670
6: -74.6560669, 37.7137375, -74.7372589, 37.8322945, -112.4883499, 112.4509888
7: -64.7670441, 54.5236053, -64.8657227, 54.6391716, -119.4062195, 119.3893280
8: -59.9414406, 58.5545959, -60.0612717, 58.7034073, -118.6448517, 118.6158600
9: -45.8682175, 49.8525352, -45.9652290, 49.9037094, -95.7719269, 95.8177643
10: -75.0594482, 63.4004097, -75.1669464, 63.4731407, -138.5325623, 138.5673523
11: -77.7508774, 45.8124275, -77.9020844, 45.9305573, -123.6814346, 123.7145081
12: -79.3309021, 53.9266357, -79.4507599, 54.0287018, -133.3596039, 133.3773804
13: -61.7636833, 77.4408569, -61.9245110, 77.7023926, -139.4660797, 139.3653717
14: -118.8816757, 37.8412743, -119.0306778, 37.9157791, -156.7974548, 156.8719482
15: -57.9734573, 61.6565895, -58.0691261, 61.7602119, -119.7336731, 119.7257156
16: -85.2545166, 57.8604660, -85.3570175, 57.9437981, -143.1983032, 143.2174683
17: -119.3317719, 53.8174973, -119.4459915, 53.9097672, -173.2415161, 173.2634888
18: -74.7670746, 43.1640549, -74.9557495, 43.2871933, -118.0542603, 118.1197968
19: -59.1647453, 27.4856682, -59.2976723, 27.5578156, -86.7225494, 86.7833405
20: -50.8806458, 35.1348495, -50.9720535, 35.1986618, -86.0792999, 86.1069031
21: -71.2662811, 35.6530151, -71.4412003, 35.7485733, -107.0148544, 107.0942154
22: -68.4009933, 41.5907898, -68.5421982, 41.6596832, -110.0606766, 110.1329880
23: -54.8438835, 38.2388039, -54.9986725, 38.3634720, -93.2073517, 93.2374649
24: -58.3310165, 32.7318153, -58.4881248, 32.8215332, -91.1525497, 91.2199402
25: -50.6810455, 44.6810417, -50.8228302, 44.7837296, -95.4647675, 95.5038757
26: -86.0992279, 55.2022896, -86.3136520, 55.3099174, -141.4091492, 141.5159454
27: -68.9792175, 32.1358261, -69.1608810, 32.2354813, -101.2146988, 101.2967072
28: -55.0459633, 44.4911041, -55.1911736, 44.5985603, -99.6445236, 99.6822662
29: -69.6010208, 38.9116287, -69.7306900, 39.0198135, -108.6208344, 108.6423187
30: -68.8082428, 47.7131882, -68.9591370, 47.8092346, -116.6174698, 116.6723251
31: -70.4330368, 33.5796471, -70.6040268, 33.6665344, -104.0995712, 104.1836700
32: -65.0682373, 42.2539520, -65.1715012, 42.4061203, -107.4743576, 107.4254532
33: -84.5111237, 63.7827034, -84.6830444, 63.9145393, -148.4256592, 148.4657440
34: -76.0395966, 53.7586327, -76.1587219, 53.8692284, -129.9088135, 129.9173584
35: -67.2329407, 58.3103294, -67.3537598, 58.4322968, -125.6652374, 125.6640854
36: -70.5145798, 58.6468658, -70.6364059, 58.8010788, -129.3156586, 129.2832642
37: -104.9956741, 48.8129272, -105.1460266, 48.8687515, -153.8644257, 153.9589539
38: -97.4694366, 69.1784744, -97.6332779, 69.3717651, -166.8412018, 166.8117371
39: -104.3559799, 58.3416939, -104.5808182, 58.5047798, -162.8607178, 162.9225159
40: -93.2523499, 39.6134644, -93.4137955, 39.7068291, -132.9591827, 133.0272522
41: -67.2936249, 40.1690369, -67.3814774, 40.2497101, -107.5433273, 107.5505142
42: -53.0479317, 37.8532562, -53.1263161, 37.9175911, -90.9655228, 90.9795685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9019028
time: 85.65 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8371354, upper bound: 95.8988186
time: 97.31 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -79.7666016, 61.1248169, -79.7529373, 61.1222153, -140.8888245, 140.8777466
1: -49.9226074, 51.0960770, -49.9316940, 51.0528412, -100.9754333, 101.0277710
2: -42.3426132, 46.5491371, -42.3553238, 46.5508575, -88.8934631, 88.9044571
3: -47.2278557, 57.7653008, -47.2198639, 57.7875061, -105.0153656, 104.9851608
4: -51.6164474, 52.1488457, -51.6791954, 52.1710587, -103.7875061, 103.8280334
5: -50.6548004, 58.3713112, -50.6837654, 58.4306564, -109.0854568, 109.0550766
6: -74.8984528, 37.8718948, -74.7455978, 37.8745613, -112.7730103, 112.6174927
7: -64.8394928, 54.6528168, -64.8627396, 54.6434250, -119.4829102, 119.5155487
8: -59.9903984, 58.7134705, -60.0508194, 58.7136116, -118.7040100, 118.7642899
9: -45.9914665, 49.9142075, -45.9731979, 49.9087448, -95.9002075, 95.8874054
10: -75.2568665, 63.5056305, -75.1760635, 63.4765396, -138.7333984, 138.6817017
11: -77.9393463, 46.0453720, -77.9275742, 45.9320831, -123.8714294, 123.9729462
12: -79.6401291, 54.0132103, -79.4571381, 54.0260010, -133.6661224, 133.4703522
13: -62.1235352, 77.5663452, -61.9273376, 77.7139740, -139.8375092, 139.4936829
14: -119.0855865, 38.0359459, -119.0576630, 37.9193802, -157.0049744, 157.0936127
15: -58.0611725, 61.7785416, -58.0791283, 61.7698708, -119.8310394, 119.8576660
16: -85.4037399, 58.0366592, -85.3664551, 57.9523048, -143.3560486, 143.4031067
17: -119.5745773, 54.1187973, -119.4926071, 53.9087448, -173.4833221, 173.6114044
18: -74.8869781, 43.3608475, -74.9629059, 43.2940674, -118.1810455, 118.3237457
19: -59.3110428, 27.6663094, -59.3296471, 27.5585384, -86.8695831, 86.9959564
20: -50.9634857, 35.2625618, -50.9734497, 35.2041168, -86.1676025, 86.2359924
21: -71.4857330, 35.9060211, -71.4823914, 35.7482643, -107.2339859, 107.3884125
22: -68.6020050, 41.7877502, -68.5805283, 41.6615753, -110.2635803, 110.3682785
23: -54.9842987, 38.4512405, -55.0265427, 38.3660431, -93.3503418, 93.4777832
24: -58.5371017, 32.9918022, -58.5343552, 32.8236160, -91.3607178, 91.5261536
25: -50.8551750, 44.8987503, -50.8597260, 44.7876205, -95.6427917, 95.7584763
26: -86.2394257, 55.3767853, -86.3243256, 55.3170280, -141.5564575, 141.7011108
27: -69.0938110, 32.3964539, -69.1708527, 32.2403183, -101.3341293, 101.5672913
28: -55.1882668, 44.7674713, -55.2231636, 44.6027222, -99.7909851, 99.9906311
29: -69.8539581, 39.1700974, -69.7900696, 39.0209007, -108.8748627, 108.9601669
30: -69.0460358, 48.0519142, -69.0115051, 47.8114243, -116.8574524, 117.0634079
31: -70.6123047, 33.7773857, -70.6368942, 33.6677704, -104.2800522, 104.4142761
32: -65.4533691, 42.4637146, -65.1778946, 42.4613914, -107.9147339, 107.6416092
33: -84.7624207, 63.8734856, -84.6945038, 63.9225159, -148.6849060, 148.5679932
34: -76.2559814, 53.8357925, -76.1627350, 53.8742332, -130.1302185, 129.9985352
35: -67.4184570, 58.3572235, -67.3612823, 58.4329376, -125.8513947, 125.7184982
36: -70.8211060, 58.7873764, -70.6368866, 58.8268661, -129.6479645, 129.4242554
37: -105.2665100, 48.8776169, -105.1566620, 48.8719254, -154.1384277, 154.0342712
38: -97.9606476, 69.4812393, -97.6364212, 69.4545441, -167.4151917, 167.1176605
39: -104.7997055, 58.4634666, -104.5863800, 58.5318871, -163.3315887, 163.0498505
40: -93.5154037, 39.7833939, -93.4191895, 39.7550507, -133.2704468, 133.2025757
41: -67.5969543, 40.3361359, -67.3879776, 40.2892799, -107.8862152, 107.7241135
42: -53.2931786, 37.9654846, -53.1329956, 37.9360924, -91.2292709, 91.0984802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=500, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9231478
time: 77.54 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9200687
time: 87.51 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -79.6302948, 61.0225372, -79.8950195, 61.2114563, -140.8417358, 140.9175568
1: -49.8526459, 50.9894257, -50.0006485, 51.1383896, -100.9910355, 100.9900742
2: -42.2953568, 46.4604721, -42.4212799, 46.6615677, -88.9569244, 88.8817444
3: -47.1128235, 57.6512032, -47.2957993, 57.9089851, -105.0218048, 104.9470062
4: -51.5527344, 52.0453033, -51.7477226, 52.2840767, -103.8368073, 103.7930145
5: -50.5971413, 58.2686920, -50.7579956, 58.5459480, -109.1430893, 109.0266876
6: -74.6673050, 37.7456970, -75.0080185, 37.9671783, -112.6344757, 112.7537155
7: -64.7630844, 54.5243530, -64.9229279, 54.7650337, -119.5281219, 119.4472809
8: -59.9292984, 58.5650864, -60.0867424, 58.8407173, -118.7700119, 118.6518250
9: -45.8734818, 49.8589973, -46.1216278, 49.9804726, -95.8539581, 95.9806213
10: -75.0648499, 63.4057426, -75.2989349, 63.6003952, -138.6652527, 138.7046814
11: -77.7705841, 45.8087196, -78.0679932, 46.1486893, -123.9192581, 123.8767090
12: -79.3366852, 53.9171066, -79.7515259, 54.0961761, -133.4328613, 133.6686401
13: -61.7650909, 77.4476547, -62.2698288, 77.8021088, -139.5671997, 139.7174835
14: -118.9179077, 37.8423996, -119.2418060, 38.1618576, -157.0797577, 157.0841980
15: -57.9739647, 61.6613655, -58.1323700, 61.8478775, -119.8218384, 119.7937317
16: -85.2684555, 57.8620529, -85.5269928, 58.1039238, -143.3723755, 143.3890381
17: -119.3905792, 53.8118210, -119.7146988, 54.2675476, -173.6581116, 173.5265198
18: -74.7682343, 43.1692963, -75.0552368, 43.4903755, -118.2586060, 118.2245178
19: -59.2054596, 27.4836445, -59.4689980, 27.7391415, -86.9446030, 86.9526367
20: -50.8833694, 35.1386108, -51.0408859, 35.3593063, -86.2426758, 86.1794968
21: -71.2883148, 35.6509895, -71.6036987, 35.9632530, -107.2515640, 107.2546844
22: -68.4321594, 41.5897675, -68.7153015, 41.8683853, -110.3005447, 110.3050537
23: -54.8751068, 38.2372627, -55.1512985, 38.6046753, -93.4797821, 93.3885651
24: -58.3722191, 32.7334595, -58.6864243, 33.0962677, -91.4684753, 91.4198837
25: -50.7097664, 44.6830826, -50.9714470, 45.0296097, -95.7393646, 95.6545258
26: -86.1070328, 55.2081757, -86.4382095, 55.5034447, -141.6104736, 141.6463928
27: -68.9998779, 32.1380348, -69.2926865, 32.4783554, -101.4782333, 101.4307251
28: -55.0956879, 44.4918442, -55.3832588, 44.8705482, -99.9662323, 99.8750992
29: -69.6564255, 38.9060669, -69.9768753, 39.2800980, -108.9365158, 108.8829422
30: -68.8338776, 47.7124481, -69.1227722, 48.0992165, -116.9330750, 116.8351898
31: -70.4775009, 33.5779953, -70.7990875, 33.8839493, -104.3614502, 104.3770752
32: -65.0752869, 42.2799911, -65.5082474, 42.5428848, -107.6181717, 107.7882309
33: -84.5192108, 63.7837296, -84.9901428, 63.9742889, -148.4934845, 148.7738647
34: -76.0442123, 53.7515869, -76.3381195, 53.9102783, -129.9544983, 130.0897064
35: -67.2409821, 58.3063431, -67.5502930, 58.4517899, -125.6927567, 125.8566284
36: -70.5154419, 58.6520157, -70.8934174, 58.8633919, -129.3788300, 129.5454407
37: -105.0046997, 48.8350906, -105.5473175, 48.9836121, -153.9882965, 154.3824158
38: -97.4660034, 69.2080383, -97.9894562, 69.5515900, -167.0175781, 167.1974945
39: -104.3595810, 58.3566971, -105.0058670, 58.6016693, -162.9612427, 163.3625641
40: -93.2527618, 39.6630058, -93.7863159, 39.8843002, -133.1370544, 133.4493103
41: -67.3036499, 40.1915092, -67.6993332, 40.3648605, -107.6685104, 107.8908386
42: -53.0556221, 37.8663177, -53.3409882, 38.0136414, -91.0692596, 91.2073059

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8569587, upper bound: 95.9018063
time: 83.66 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8569587, upper bound: 95.8987007
time: 91.13 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -79.7742233, 61.1288414, -79.9026642, 61.2186584, -140.9928589, 141.0315094
1: -49.9273338, 51.0986481, -50.0046234, 51.1417770, -101.0691071, 101.1032715
2: -42.3406754, 46.5541687, -42.4151764, 46.6661911, -89.0068665, 88.9693451
3: -47.2284966, 57.7734795, -47.3081017, 57.9162979, -105.1447906, 105.0815811
4: -51.6186790, 52.1626396, -51.7525978, 52.2961960, -103.9148712, 103.9152298
5: -50.6595764, 58.3803253, -50.7547226, 58.5550880, -109.2146606, 109.1350479
6: -74.9096222, 37.9039459, -75.0163345, 38.0095253, -112.9191360, 112.9202805
7: -64.8357849, 54.6534805, -64.9200516, 54.7692833, -119.6050720, 119.5735321
8: -59.9785271, 58.7235947, -60.0760994, 58.8507996, -118.8293228, 118.7996902
9: -45.9965134, 49.9206924, -46.1294708, 49.9855766, -95.9820862, 96.0501633
10: -75.2622070, 63.5109940, -75.3080597, 63.6041374, -138.8663483, 138.8190613
11: -77.9590454, 46.0415840, -78.0932617, 46.1500092, -124.1090546, 124.1348419
12: -79.6458893, 54.0034065, -79.7578583, 54.0934753, -133.7393646, 133.7612610
13: -62.1248322, 77.5727234, -62.2724915, 77.8136749, -139.9385071, 139.8452148
14: -119.1217499, 38.0369415, -119.2688293, 38.1653862, -157.2871399, 157.3057709
15: -58.0615997, 61.7833214, -58.1423340, 61.8575020, -119.9190979, 119.9256592
16: -85.4174957, 58.0382919, -85.5362244, 58.1124420, -143.5299225, 143.5745239
17: -119.6332855, 54.1131325, -119.7613297, 54.2664719, -173.8997498, 173.8744507
18: -74.8884659, 43.3660545, -75.0625000, 43.4971123, -118.3855743, 118.4285583
19: -59.3517647, 27.6642723, -59.5009995, 27.7397976, -87.0915604, 87.1652679
20: -50.9662971, 35.2662735, -51.0423279, 35.3647156, -86.3310013, 86.3086014
21: -71.5079041, 35.9039917, -71.6450729, 35.9629555, -107.4708557, 107.5490570
22: -68.6331787, 41.7867432, -68.7538223, 41.8702736, -110.5034485, 110.5405655
23: -55.0157471, 38.4496765, -55.1793404, 38.6071854, -93.6229172, 93.6290131
24: -58.5783768, 32.9934006, -58.7327003, 33.0982971, -91.6766739, 91.7260895
25: -50.8841209, 44.9007492, -51.0086365, 45.0334167, -95.9175415, 95.9093857
26: -86.2469635, 55.3825607, -86.4488754, 55.5104446, -141.7574158, 141.8314362
27: -69.1144409, 32.3985672, -69.3025665, 32.4831238, -101.5975494, 101.7011337
28: -55.2381058, 44.7681770, -55.4155693, 44.8746643, -100.1127701, 100.1837463
29: -69.9094849, 39.1644745, -70.0365753, 39.2811279, -109.1906052, 109.2010498
30: -69.0719452, 48.0511017, -69.1755219, 48.1012802, -117.1732254, 117.2266083
31: -70.6567688, 33.7757263, -70.8320007, 33.8851624, -104.5419312, 104.6077271
32: -65.4603882, 42.4896889, -65.5145569, 42.5982475, -108.0586243, 108.0042419
33: -84.7704163, 63.8745079, -85.0013580, 63.9823227, -148.7527161, 148.8758545
34: -76.2606049, 53.8287811, -76.3421783, 53.9152641, -130.1758575, 130.1709595
35: -67.4263153, 58.3533173, -67.5578995, 58.4524612, -125.8787689, 125.9112167
36: -70.8219147, 58.7926025, -70.8937988, 58.8893967, -129.7113037, 129.6864014
37: -105.2754059, 48.8994217, -105.5577545, 48.9865379, -154.2619476, 154.4571838
38: -97.9571991, 69.5112457, -97.9924698, 69.6344757, -167.5916748, 167.5037231
39: -104.8031921, 58.4785652, -105.0111008, 58.6288605, -163.4320526, 163.4896545
40: -93.5158615, 39.8330612, -93.7917633, 39.9327927, -133.4486542, 133.6248169
41: -67.6069412, 40.3585815, -67.7058258, 40.4044800, -108.0113907, 108.0644073
42: -53.3008194, 37.9782410, -53.3476562, 38.0319824, -91.3328018, 91.3258972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=500, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9229435
time: 84.59 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9198874
time: 91.93 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -79.7998810, 61.1270561, -79.8087082, 61.1328697, -140.9327545, 140.9357605
1: -49.9513702, 51.0620422, -49.9639130, 51.0660172, -101.0173874, 101.0259552
2: -42.4010696, 46.5616570, -42.4046440, 46.5623322, -88.9633942, 88.9662933
3: -47.2631493, 57.8006439, -47.2716217, 57.7987900, -105.0619354, 105.0722656
4: -51.7445335, 52.1781387, -51.7576752, 52.1739807, -103.9185181, 103.9358139
5: -50.7502403, 58.4469757, -50.7568169, 58.4469414, -109.1971741, 109.2037964
6: -74.7811966, 37.8442230, -74.7772217, 37.8594742, -112.6406708, 112.6214371
7: -64.9023590, 54.6577225, -64.9168854, 54.6622849, -119.5646362, 119.5746078
8: -60.1117973, 58.7264137, -60.1300468, 58.7265053, -118.8383026, 118.8564606
9: -45.9968643, 49.9206314, -46.0095482, 49.9224968, -95.9193573, 95.9301758
10: -75.1961212, 63.5018196, -75.2053070, 63.5022011, -138.6983185, 138.7071228
11: -77.9342651, 46.0020676, -77.9311829, 46.0102234, -123.9444885, 123.9332504
12: -79.4867020, 54.0451889, -79.4907684, 54.0655708, -133.5522461, 133.5359497
13: -62.0342636, 77.7097321, -62.0476074, 77.7239990, -139.7582703, 139.7573395
14: -119.0908203, 37.9450760, -119.0733795, 37.9533005, -157.0441132, 157.0184631
15: -58.0880051, 61.7795868, -58.1046371, 61.7794304, -119.8674316, 119.8842239
16: -85.3985443, 57.9768791, -85.3979034, 57.9893036, -143.3878479, 143.3747711
17: -119.5048370, 53.9336624, -119.4842300, 53.9355431, -173.4403839, 173.4178772
18: -74.9792480, 43.3587875, -74.9785538, 43.3728256, -118.3520737, 118.3373413
19: -59.3249588, 27.6109562, -59.3144913, 27.6133022, -86.9382477, 86.9254456
20: -50.9887886, 35.2330856, -50.9884262, 35.2402573, -86.2290497, 86.2215118
21: -71.4428329, 35.8220367, -71.4612732, 35.8241730, -107.2670059, 107.2833099
22: -68.5587082, 41.7070847, -68.5681000, 41.7101402, -110.2688446, 110.2751846
23: -55.0270615, 38.4517136, -55.0155029, 38.4586945, -93.4857483, 93.4672089
24: -58.5036087, 32.8872299, -58.5093994, 32.8915787, -91.3951797, 91.3966293
25: -50.8309250, 44.8543320, -50.8419838, 44.8574600, -95.6883850, 95.6963043
26: -86.3478622, 55.3791962, -86.3465881, 55.3847122, -141.7325745, 141.7257843
27: -69.1960754, 32.2969589, -69.1820679, 32.3084526, -101.5045242, 101.4790268
28: -55.2323914, 44.6730652, -55.2064476, 44.6798439, -99.9122314, 99.8795166
29: -69.7497711, 39.0886650, -69.7551270, 39.0985603, -108.8483200, 108.8437958
30: -68.9496460, 47.8714714, -68.9783630, 47.8745193, -116.8241653, 116.8498383
31: -70.6348877, 33.7184486, -70.6241455, 33.7230377, -104.3579254, 104.3425903
32: -65.2178802, 42.3787766, -65.2201385, 42.4294930, -107.6473694, 107.5989151
33: -84.7681351, 63.9477577, -84.7828293, 63.9423332, -148.7104492, 148.7305908
34: -76.2175598, 53.8584862, -76.2224197, 53.8818474, -130.0993958, 130.0809021
35: -67.4176178, 58.4419746, -67.4241486, 58.4521294, -125.8697510, 125.8661194
36: -70.7024384, 58.7652512, -70.7053986, 58.8161850, -129.5186157, 129.4706421
37: -105.2197495, 48.9189339, -105.2240524, 48.8912201, -154.1109619, 154.1429901
38: -97.7115402, 69.3298264, -97.7181015, 69.3900681, -167.1016083, 167.0479279
39: -104.7246399, 58.5164680, -104.7330246, 58.5173531, -163.2419891, 163.2494965
40: -93.4978333, 39.7275543, -93.5027695, 39.7226105, -133.2204437, 133.2303162
41: -67.4316711, 40.2404633, -67.4258499, 40.2691879, -107.7008591, 107.6663132
42: -53.1515770, 37.9310074, -53.1515312, 37.9398232, -91.0914001, 91.0825348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9040957
time: 81.60 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9011311
time: 117.92 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -79.9437714, 61.2334137, -79.8165207, 61.1401634, -141.0839386, 141.0499268
1: -50.0259132, 51.1712799, -49.9680862, 51.0694427, -101.0953522, 101.1393661
2: -42.4463272, 46.6554909, -42.3985977, 46.5670662, -89.0133972, 89.0540924
3: -47.3786697, 57.9230766, -47.2845573, 57.8061829, -105.1848526, 105.2076187
4: -51.8105965, 52.2954254, -51.7624130, 52.1857910, -103.9963837, 104.0578308
5: -50.8126984, 58.5590858, -50.7535095, 58.4562683, -109.2689514, 109.3125839
6: -75.0235901, 38.0023689, -74.7855835, 37.9017715, -112.9253540, 112.7879257
7: -64.9748688, 54.7868385, -64.9138641, 54.6665077, -119.6413574, 119.7006989
8: -60.1607971, 58.8852272, -60.1195869, 58.7367325, -118.8975220, 119.0048141
9: -46.1201248, 49.9823227, -46.0175209, 49.9275169, -96.0476379, 95.9998474
10: -75.3935242, 63.6070328, -75.2143936, 63.5056000, -138.8991089, 138.8214264
11: -78.1226730, 46.2349930, -77.9566498, 46.0117798, -124.1344452, 124.1916428
12: -79.7959290, 54.1317215, -79.4971542, 54.0628433, -133.8587646, 133.6288757
13: -62.3941765, 77.8352203, -62.0504303, 77.7356262, -140.1298065, 139.8856506
14: -119.2946548, 38.1397095, -119.1003342, 37.9568558, -157.2515106, 157.2400513
15: -58.1757240, 61.9015350, -58.1146355, 61.7890663, -119.9647827, 120.0161667
16: -85.5476837, 58.1530914, -85.4073486, 57.9977875, -143.5454712, 143.5604248
17: -119.7476044, 54.2349472, -119.5308533, 53.9344940, -173.6820984, 173.7657776
18: -75.0991364, 43.5556068, -74.9857025, 43.3796616, -118.4787979, 118.5413055
19: -59.4711914, 27.7915764, -59.3464508, 27.6140213, -87.0852127, 87.1380310
20: -51.0716400, 35.3608246, -50.9898376, 35.2456894, -86.3173294, 86.3506622
21: -71.6622543, 36.0750504, -71.5024261, 35.8239212, -107.4861755, 107.5774689
22: -68.7596970, 41.9040260, -68.6064682, 41.7119980, -110.4716949, 110.5104980
23: -55.1674500, 38.6641769, -55.0433769, 38.4612503, -93.6287003, 93.7075500
24: -58.7096367, 33.1472321, -58.5556068, 32.8936615, -91.6033020, 91.7028351
25: -51.0050316, 45.0720825, -50.8788872, 44.8613319, -95.8663559, 95.9509735
26: -86.4879456, 55.5536728, -86.3572388, 55.3917618, -141.8796997, 141.9109192
27: -69.3106232, 32.5576057, -69.1920471, 32.3132858, -101.6239014, 101.7496490
28: -55.3746376, 44.9494591, -55.2384300, 44.6840019, -100.0586319, 100.1878891
29: -70.0026550, 39.3471184, -69.8144913, 39.0996628, -109.1023102, 109.1616058
30: -69.1873779, 48.2102013, -69.0307388, 47.8766823, -117.0640488, 117.2409363
31: -70.8141403, 33.9162140, -70.6570282, 33.7242661, -104.5384064, 104.5732422
32: -65.6030273, 42.5884476, -65.2265472, 42.4848099, -108.0878296, 107.8149872
33: -85.0196075, 64.0384521, -84.7942581, 63.9503212, -148.9699249, 148.8327026
34: -76.4340515, 53.9356079, -76.2264175, 53.8868370, -130.3208771, 130.1620178
35: -67.6032028, 58.4888535, -67.4317017, 58.4527588, -126.0559616, 125.9205399
36: -71.0090485, 58.9056625, -70.7058792, 58.8419762, -129.8510132, 129.6115417
37: -105.4907227, 48.9836082, -105.2346344, 48.8943748, -154.3851013, 154.2182312
38: -98.2027435, 69.6324310, -97.7212524, 69.4729156, -167.6756592, 167.3536835
39: -105.1685333, 58.6382599, -104.7386017, 58.5444679, -163.7129822, 163.3768616
40: -93.7609406, 39.8973885, -93.5082245, 39.7708282, -133.5317688, 133.4056091
41: -67.7350006, 40.4075241, -67.4323578, 40.3087311, -108.0437241, 107.8398666
42: -53.3968315, 38.0431938, -53.1581955, 37.9583359, -91.3551559, 91.2013855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9253123
time: 89.51 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9222829
time: 81.87 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -79.8075104, 61.1312141, -79.9586258, 61.2294388, -141.0369415, 141.0898132
1: -49.9559669, 51.0646286, -50.0370064, 51.1550102, -101.1109772, 101.1016388
2: -42.3990936, 46.5668869, -42.4645691, 46.6777725, -89.0768661, 89.0314560
3: -47.2636070, 57.8089905, -47.3604813, 57.9276619, -105.1912613, 105.1694641
4: -51.7468376, 52.1919479, -51.8309708, 52.2987671, -104.0456085, 104.0229187
5: -50.7550354, 58.4565582, -50.8277817, 58.5715599, -109.3265991, 109.2843323
6: -74.7924042, 37.8762360, -75.0479584, 37.9943466, -112.7867508, 112.9241867
7: -64.8984528, 54.6584167, -64.9740753, 54.7881088, -119.6865463, 119.6324921
8: -60.0996437, 58.7369614, -60.1555214, 58.8638306, -118.9634705, 118.8924713
9: -46.0021667, 49.9271088, -46.1659355, 49.9992485, -96.0014191, 96.0930481
10: -75.2015457, 63.5071411, -75.3372803, 63.6294441, -138.8309784, 138.8444214
11: -77.9539795, 45.9983215, -78.0970459, 46.2283440, -124.1823273, 124.0953674
12: -79.4924927, 54.0356598, -79.7916412, 54.1329765, -133.6254578, 133.8273010
13: -62.0356827, 77.7165604, -62.3929634, 77.8237000, -139.8593750, 140.1095276
14: -119.1270523, 37.9461136, -119.2844391, 38.1993370, -157.3263855, 157.2305603
15: -58.0885277, 61.7843933, -58.1678772, 61.8670235, -119.9555511, 119.9522705
16: -85.4125824, 57.9784546, -85.5678101, 58.1494637, -143.5620422, 143.5462646
17: -119.5636520, 53.9279938, -119.7528839, 54.2933197, -173.8569641, 173.6808777
18: -74.9804153, 43.3640366, -75.0780334, 43.5760612, -118.5564728, 118.4420700
19: -59.3656616, 27.6089363, -59.4857941, 27.7946091, -87.1602631, 87.0947266
20: -50.9915352, 35.2368317, -51.0572662, 35.4009247, -86.3924561, 86.2940826
21: -71.4648666, 35.8199844, -71.6237030, 36.0389099, -107.5037766, 107.4436722
22: -68.5898666, 41.7060280, -68.7411652, 41.9188461, -110.5087051, 110.4471893
23: -55.0582581, 38.4501495, -55.1680984, 38.6999283, -93.7581787, 93.6182480
24: -58.5448074, 32.8888931, -58.7076149, 33.1663284, -91.7111359, 91.5965042
25: -50.8596573, 44.8563843, -50.9905586, 45.1033401, -95.9629974, 95.8469391
26: -86.3555984, 55.3850670, -86.4710846, 55.5782166, -141.9338074, 141.8561401
27: -69.2167282, 32.2991753, -69.3138199, 32.5513496, -101.7680817, 101.6129913
28: -55.2820969, 44.6738358, -55.3984833, 44.9518585, -100.2339554, 100.0723038
29: -69.8051834, 39.0830727, -70.0012741, 39.3588600, -109.1640472, 109.0843277
30: -68.9752655, 47.8706894, -69.1419983, 48.1645279, -117.1397858, 117.0126877
31: -70.6793747, 33.7167625, -70.8192139, 33.9404755, -104.6198502, 104.5359802
32: -65.2249451, 42.4048080, -65.5569458, 42.5662270, -107.7911682, 107.9617538
33: -84.7762299, 63.9487762, -85.0901642, 64.0020905, -148.7783051, 149.0389252
34: -76.2221527, 53.8514366, -76.4018478, 53.9229012, -130.1450500, 130.2532806
35: -67.4256287, 58.4379959, -67.6207809, 58.4716339, -125.8972626, 126.0587769
36: -70.7033234, 58.7703629, -70.9624405, 58.8784637, -129.5817871, 129.7328033
37: -105.2287598, 48.9410591, -105.6254349, 49.0060463, -154.2348022, 154.5664978
38: -97.7080841, 69.3593750, -98.0743332, 69.5698471, -167.2779236, 167.4337158
39: -104.7283096, 58.5315247, -105.1582031, 58.6142540, -163.3425598, 163.6897278
40: -93.4982529, 39.7770844, -93.8754272, 39.9000359, -133.3982849, 133.6525116
41: -67.4416809, 40.2629089, -67.7437439, 40.3843269, -107.8260040, 108.0066528
42: -53.1592903, 37.9440575, -53.3662148, 38.0358200, -91.1950989, 91.3102646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.9220892, upper bound: 95.9040122
time: 88.51 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.9220892, upper bound: 95.9010258
time: 87.45 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -79.9514389, 61.2374420, -79.9662476, 61.2366409, -141.1880798, 141.2036896
1: -50.0306396, 51.1738625, -50.0410118, 51.1584129, -101.1890564, 101.2148743
2: -42.4444046, 46.6605263, -42.4584808, 46.6824036, -89.1268082, 89.1190033
3: -47.3792877, 57.9312325, -47.3727875, 57.9349976, -105.3142853, 105.3040161
4: -51.8128204, 52.3092651, -51.8358612, 52.3108673, -104.1236877, 104.1451263
5: -50.8174667, 58.5681458, -50.8244667, 58.5807304, -109.3981934, 109.3926086
6: -75.0347672, 38.0344009, -75.0563049, 38.0366936, -113.0714569, 113.0907059
7: -64.9711456, 54.7875595, -64.9712143, 54.7923203, -119.7634659, 119.7587738
8: -60.1489334, 58.8953133, -60.1448975, 58.8739014, -119.0228271, 119.0402069
9: -46.1251907, 49.9888382, -46.1737900, 50.0043793, -96.1295700, 96.1626282
10: -75.3988800, 63.6123848, -75.3464127, 63.6331711, -139.0320435, 138.9588013
11: -78.1423798, 46.2312317, -78.1223373, 46.2296791, -124.3720551, 124.3535690
12: -79.8016510, 54.1219826, -79.7979507, 54.1302872, -133.9319458, 133.9199371
13: -62.3955040, 77.8416138, -62.3956413, 77.8352966, -140.2308044, 140.2372437
14: -119.3308182, 38.1407318, -119.3114395, 38.2028694, -157.5336914, 157.4521790
15: -58.1761475, 61.9063377, -58.1778374, 61.8767014, -120.0528488, 120.0841751
16: -85.5615005, 58.1547585, -85.5770950, 58.1579819, -143.7194824, 143.7318573
17: -119.8063812, 54.2292824, -119.7994843, 54.2922974, -174.0986633, 174.0287476
18: -75.1006317, 43.5608330, -75.0853119, 43.5827560, -118.6833725, 118.6461487
19: -59.5119324, 27.7895508, -59.5177841, 27.7952785, -87.3072052, 87.3073349
20: -51.0744438, 35.3645401, -51.0587082, 35.4063606, -86.4807892, 86.4232407
21: -71.6844482, 36.0729980, -71.6650543, 36.0385895, -107.7230377, 107.7380524
22: -68.7908478, 41.9029961, -68.7797089, 41.9207039, -110.7115479, 110.6827087
23: -55.1989098, 38.6625977, -55.1961670, 38.7024155, -93.9013214, 93.8587646
24: -58.7509079, 33.1488152, -58.7539253, 33.1683540, -91.9192657, 91.9027405
25: -51.0339928, 45.0740967, -51.0277596, 45.1071701, -96.1411591, 96.1018524
26: -86.4954987, 55.5594559, -86.4817581, 55.5851898, -142.0806885, 142.0412140
27: -69.3312836, 32.5597229, -69.3237152, 32.5561142, -101.8873978, 101.8834381
28: -55.4244614, 44.9501801, -55.4308014, 44.9559441, -100.3804016, 100.3809814
29: -70.0581894, 39.3415146, -70.0609741, 39.3598824, -109.4180679, 109.4024811
30: -69.2132721, 48.2093964, -69.1947327, 48.1666069, -117.3798828, 117.4041290
31: -70.8586273, 33.9145470, -70.8521271, 33.9416733, -104.8002930, 104.7666779
32: -65.6100464, 42.6144409, -65.5632782, 42.6216049, -108.2316437, 108.1777191
33: -85.0276108, 64.0395203, -85.1013641, 64.0100708, -149.0376892, 149.1408691
34: -76.4386749, 53.9286118, -76.4058838, 53.9278984, -130.3665466, 130.3345032
35: -67.6110840, 58.4849854, -67.6283646, 58.4723129, -126.0833893, 126.1133423
36: -71.0098343, 58.9109192, -70.9628296, 58.9045067, -129.9143372, 129.8737488
37: -105.4996185, 49.0054131, -105.6358414, 49.0089989, -154.5086212, 154.6412506
38: -98.1993332, 69.6624603, -98.0773621, 69.6527710, -167.8521118, 167.7398224
39: -105.1720428, 58.6533165, -105.1634369, 58.6413536, -163.8133850, 163.8167419
40: -93.7613983, 39.9470787, -93.8808365, 39.9485054, -133.7098999, 133.8279114
41: -67.7449646, 40.4299469, -67.7502441, 40.4238968, -108.1688538, 108.1801834
42: -53.4044876, 38.0559769, -53.3728981, 38.0541992, -91.4586868, 91.4288635

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.9220892, upper bound: 95.9251204
time: 803.56 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9220892
time: 98.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 904.60 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9019028
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.8371354, upper bound: 95.8988186
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9231478
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9200687
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.8569587, upper bound: 95.9018063
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.8569587, upper bound: 95.8987007
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9229435
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9198874
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9040957
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9011311
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9253123
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9222829
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.9220892, upper bound: 95.9040122
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.9220892, upper bound: 95.9010258
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.9220892, upper bound: 95.9251204
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 904.60
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9220892

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -79.6098633, 61.0020180, -79.7127457, 61.0738678, -140.6837158, 140.7147675
1: -49.8395424, 50.9692726, -49.9059868, 51.0056763, -100.8452148, 100.8752594
2: -42.2910385, 46.4346848, -42.3454399, 46.4950905, -88.7861328, 88.7801208
3: -47.1051216, 57.6165810, -47.1889877, 57.7141418, -104.8192596, 104.8055725
4: -51.5454178, 52.0199013, -51.6617546, 52.1306458, -103.6760559, 103.6816559
5: -50.5874519, 58.2315826, -50.6748199, 58.3529854, -108.9404373, 108.9063950
6: -74.6453552, 37.6848068, -74.7107849, 37.7597656, -112.4051056, 112.3955841
7: -64.7551727, 54.4756889, -64.8360138, 54.5187035, -119.2738571, 119.3116989
8: -59.9360733, 58.5306168, -60.0478477, 58.6437950, -118.5798645, 118.5784531
9: -45.8596001, 49.8413086, -45.9440918, 49.8754120, -95.7350082, 95.7854004
10: -75.0475845, 63.3844528, -75.1377487, 63.4334488, -138.4810333, 138.5221863
11: -77.7294617, 45.7880287, -77.8485947, 45.8734970, -123.6029587, 123.6366272
12: -79.2485809, 53.9166489, -79.2479477, 54.0038223, -133.2523956, 133.1645813
13: -61.7354736, 77.4262085, -61.8566895, 77.6657562, -139.4012299, 139.2828979
14: -118.8132248, 37.8345146, -118.8581848, 37.8990974, -156.7123108, 156.6926880
15: -57.9182243, 61.6437798, -57.9300575, 61.7284203, -119.6466446, 119.5738373
16: -85.2359924, 57.8040237, -85.3113403, 57.8030815, -143.0390625, 143.1153564
17: -119.2524872, 53.8075409, -119.2459564, 53.8851814, -173.1376648, 173.0534973
18: -74.7350540, 43.1530533, -74.8752823, 43.2603683, -117.9954224, 118.0283356
19: -59.1540489, 27.4800739, -59.2711754, 27.5438614, -86.6979065, 86.7512512
20: -50.8645744, 35.1279068, -50.9318657, 35.1811867, -86.0457611, 86.0597687
21: -71.2517166, 35.6437454, -71.4050064, 35.7252846, -106.9770050, 107.0487366
22: -68.3288879, 41.5808983, -68.3604965, 41.6352272, -109.9641113, 109.9413910
23: -54.8295708, 38.2290955, -54.9628563, 38.3391266, -93.1687012, 93.1919556
24: -58.3181496, 32.7216835, -58.4555931, 32.7965088, -91.1146469, 91.1772766
25: -50.6543922, 44.6726646, -50.7558365, 44.7628784, -95.4172668, 95.4284973
26: -86.0057755, 55.1915283, -86.0865326, 55.2834129, -141.2891846, 141.2780457
27: -68.9643707, 32.1277618, -69.1240387, 32.2154541, -101.1798248, 101.2518005
28: -55.0330162, 44.4853821, -55.1590767, 44.5841522, -99.6171722, 99.6444550
29: -69.5483856, 38.9054718, -69.5994949, 39.0043488, -108.5527344, 108.5049667
30: -68.7935944, 47.6630211, -68.9225769, 47.6843567, -116.4779129, 116.5855942
31: -70.4212723, 33.5696259, -70.5746460, 33.6415482, -104.0628204, 104.1442719
32: -65.0449829, 42.2437439, -65.1135941, 42.3805809, -107.4255676, 107.3573227
33: -84.4971848, 63.7606964, -84.6482086, 63.8592987, -148.3564758, 148.4089050
34: -76.0249634, 53.7456245, -76.1224823, 53.8368263, -129.8617859, 129.8681030
35: -67.2189636, 58.2993889, -67.3191376, 58.4048691, -125.6238251, 125.6185303
36: -70.4798889, 58.6404877, -70.5499268, 58.7849655, -129.2648621, 129.1904144
37: -104.9730759, 48.8053207, -105.0897675, 48.8495827, -153.8226624, 153.8950806
38: -97.4245148, 69.1679230, -97.5244751, 69.3455429, -166.7700500, 166.6923981
39: -104.3396149, 58.3269997, -104.5405502, 58.4677963, -162.8074036, 162.8675385
40: -93.2389145, 39.5587006, -93.3803406, 39.5707855, -132.8096924, 132.9390259
41: -67.2811890, 40.1524734, -67.3508606, 40.2102394, -107.4914246, 107.5033340
42: -53.0362625, 37.8313904, -53.0973091, 37.8638916, -90.9001541, 90.9286957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 966

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 648

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8014168, upper bound: 95.8709096
time: 108.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8014168, upper bound: 95.8706619
time: 88.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -79.6140060, 60.9983864, -79.9575500, 61.1339111, -140.7479248, 140.9559326
1: -49.8416214, 50.9624977, -50.0750427, 51.0638809, -100.9055023, 101.0375366
2: -42.2917480, 46.4346275, -42.5731735, 46.5588684, -88.8506165, 89.0077972
3: -47.1066780, 57.6175919, -47.4369965, 57.8018379, -104.9085159, 105.0545807
4: -51.5446510, 52.0231743, -51.7589645, 52.2369690, -103.7816162, 103.7821350
5: -50.5876656, 58.2368431, -50.9217796, 58.4441528, -109.0318146, 109.1586227
6: -74.6463470, 37.6710587, -74.8875275, 37.8514671, -112.4978180, 112.5585785
7: -64.7566376, 54.4579811, -65.2295685, 54.5918083, -119.3484497, 119.6875458
8: -59.9361839, 58.5302620, -60.3132935, 58.7381477, -118.6743317, 118.8435516
9: -45.8569984, 49.8457832, -46.0244484, 49.9963188, -95.8533173, 95.8702316
10: -75.0482941, 63.3914757, -75.2590179, 63.7055435, -138.7538300, 138.6504822
11: -77.7342682, 45.7294388, -78.0860291, 45.9403801, -123.6746292, 123.8154678
12: -79.3048325, 53.9216614, -79.5002289, 54.6630096, -133.9678345, 133.4218903
13: -61.7133102, 77.4313965, -61.9598923, 78.0361862, -139.7494812, 139.3912811
14: -118.8571396, 37.8362427, -119.1864853, 38.3464355, -157.2035828, 157.0227203
15: -57.8960190, 61.6448555, -58.0815086, 62.1074486, -120.0034485, 119.7263641
16: -85.2408600, 57.7792625, -85.7470245, 57.9210663, -143.1619110, 143.5262909
17: -119.3111725, 53.8053780, -119.5458145, 54.4318695, -173.7430420, 173.3511963
18: -74.7483749, 43.1517601, -75.0214539, 43.4397087, -118.1880798, 118.1732101
19: -59.1571274, 27.4813900, -59.3479691, 27.6323833, -86.7895050, 86.8293533
20: -50.8702049, 35.1296234, -51.0417976, 35.3134270, -86.1836319, 86.1714172
21: -71.2563095, 35.6355247, -71.5076447, 35.8403854, -107.0966949, 107.1431656
22: -68.3298569, 41.5797844, -68.5420837, 42.1652603, -110.4951172, 110.1218567
23: -54.8372231, 38.2193336, -55.0809669, 38.3864975, -93.2237244, 93.3003006
24: -58.3224373, 32.7136192, -58.6209106, 32.8494759, -91.1719131, 91.3345337
25: -50.6505928, 44.6710358, -50.8497581, 44.9545288, -95.6051178, 95.5207977
26: -86.0489044, 55.1909943, -86.3741379, 55.9246902, -141.9735870, 141.5651245
27: -68.9707413, 32.1119308, -69.2854385, 32.2558098, -101.2265472, 101.3973541
28: -55.0399284, 44.4856606, -55.2780342, 44.6495209, -99.6894455, 99.7636871
29: -69.5785370, 38.9060516, -69.8075180, 39.4719048, -109.0504456, 108.7135696
30: -68.7956390, 47.6802673, -69.2490463, 47.8467064, -116.6423340, 116.9293137
31: -70.4253235, 33.5617752, -70.7137146, 33.7409592, -104.1662750, 104.2754898
32: -65.0551147, 42.2468987, -65.2307892, 42.5594635, -107.6145630, 107.4776764
33: -84.5007324, 63.7700272, -84.8788147, 64.0034409, -148.5041504, 148.6488342
34: -76.0291824, 53.7487411, -76.3089447, 53.9213448, -129.9505310, 130.0576782
35: -67.2210007, 58.3031998, -67.4405975, 58.4753342, -125.6963348, 125.7437973
36: -70.4688110, 58.6430550, -70.6476288, 59.1184273, -129.5872192, 129.2906799
37: -104.9765854, 48.8070908, -105.2436295, 48.9431229, -153.9197083, 154.0507202
38: -97.4456100, 69.1708755, -97.7162094, 69.6872025, -167.1328125, 166.8870850
39: -104.3302078, 58.3345490, -104.6187286, 58.5718918, -162.9020996, 162.9532776
40: -93.2420502, 39.6003914, -93.7469940, 39.7423019, -132.9843445, 133.3473816
41: -67.2845306, 40.1433868, -67.5014954, 40.2991180, -107.5836487, 107.6448746
42: -53.0405083, 37.8261795, -53.2301102, 37.9768753, -91.0173798, 91.0562897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=503, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 648

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8014168, upper bound: 95.8692518
time: 97.27 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -95.8014168, upper bound: 95.8692518
time: 251.09 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 350.77 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 350.77
Output dim: 13, lower bound: -95.8014168, upper bound: 95.8709096
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 350.77
Output dim: 13, lower bound: -95.8014168, upper bound: 95.8706619
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 350.77
Output dim: 13, lower bound: -95.8014168, upper bound: 95.8692518
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 350.77
Output dim: 13, lower bound: -95.8014168, upper bound: 95.8692518
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 350.77
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9231478
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 350.77
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9200687
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 350.77
Output dim: 13, lower bound: -95.8569587, upper bound: 95.9018063
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 350.77
Output dim: 13, lower bound: -95.8569587, upper bound: 95.8987007
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 350.77
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9229435
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 350.77
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9198874
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 350.77
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9040957
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 350.77
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9011311
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 350.77
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9253123
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 350.77
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9222829
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 350.77
Output dim: 13, lower bound: -95.9220892, upper bound: 95.9040122
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 350.77
Output dim: 13, lower bound: -95.9220892, upper bound: 95.9010258
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 350.77
Output dim: 13, lower bound: -95.9220892, upper bound: 95.9251204
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 350.77
Output dim: 13, lower bound: -95.8371354, upper bound: 95.9220892
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=139.97683715820312
rel_dist={13: [-96.04775687840494, 96.04775687390503]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1725
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 1725

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -94.2374280, upper bound: 94.2896431
time: 88.86 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -94.2374280, upper bound: 94.2914374
time: 78.42 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 167.42 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 167.42
Output dim: 13, lower bound: -94.2374280, upper bound: 94.2896431
IS_A2, status: Status.UNKNOWN, split count: 1, time: 167.42
Output dim: 13, lower bound: -94.2374280, upper bound: 94.2914374

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -79.7487793, 61.0640106, -79.8558960, 61.1533394, -140.9021149, 140.9199066
1: -49.9181595, 51.0219803, -49.9825211, 51.0793953, -100.9975433, 101.0045013
2: -42.3431664, 46.4915161, -42.3993988, 46.5807953, -88.9239578, 88.8909073
3: -47.1949654, 57.6961174, -47.2733154, 57.8343544, -105.0293198, 104.9694366
4: -51.5967598, 52.1070290, -51.6974945, 52.2381973, -103.8349609, 103.8045120
5: -50.6425476, 58.3190613, -50.7223816, 58.4793167, -109.1218643, 109.0414276
6: -74.7151260, 37.8987808, -74.7971268, 37.9996948, -112.7148056, 112.6959076
7: -64.8565826, 54.5657806, -64.9364471, 54.6752434, -119.5318298, 119.5022278
8: -59.9942093, 58.6274529, -60.0879517, 58.7744942, -118.7687073, 118.7154083
9: -45.9228210, 49.9003677, -46.0034332, 49.9487686, -95.8715744, 95.9037933
10: -75.1153793, 63.4623795, -75.2099838, 63.5333481, -138.6487274, 138.6723633
11: -77.9448547, 45.8540573, -78.0975876, 45.9545708, -123.8993988, 123.9516373
12: -79.3902588, 54.0684700, -79.5025635, 54.1464310, -133.5366821, 133.5710297
13: -61.8165207, 77.5958405, -61.9506493, 77.8415222, -139.6580505, 139.5464935
14: -119.0646591, 37.8752098, -119.2283020, 37.9407196, -157.0053711, 157.1035156
15: -58.0404930, 61.7140465, -58.1158791, 61.8166466, -119.8571320, 119.8299255
16: -85.3749237, 57.9256363, -85.4745941, 57.9934425, -143.3683624, 143.4002380
17: -119.5854034, 53.8654747, -119.7181244, 53.9532814, -173.5386810, 173.5835876
18: -74.8276825, 43.2151604, -75.0158768, 43.3150826, -118.1427536, 118.2310333
19: -59.3319778, 27.5093632, -59.4754257, 27.5723686, -86.9043427, 86.9847870
20: -50.9427338, 35.1749802, -51.0337029, 35.2301750, -86.1729126, 86.2086792
21: -71.4757538, 35.6804161, -71.6322327, 35.7651405, -107.2408905, 107.3126373
22: -68.5911102, 41.6195793, -68.7210388, 41.6799927, -110.2711029, 110.3406219
23: -55.0032005, 38.2653656, -55.1690712, 38.3715210, -93.3747253, 93.4344330
24: -58.5453110, 32.7553902, -58.6964912, 32.8328476, -91.3781586, 91.4518814
25: -50.8489380, 44.7137146, -50.9799614, 44.8044128, -95.6533432, 95.6936798
26: -86.1758575, 55.2631073, -86.3893738, 55.3574448, -141.5332947, 141.6524811
27: -69.0784912, 32.1684456, -69.2728882, 32.2493744, -101.3278656, 101.4413300
28: -55.2288132, 44.5273552, -55.3994637, 44.6187210, -99.8475266, 99.9268112
29: -69.8836975, 38.9489441, -70.0065308, 39.0374908, -108.9211884, 108.9554749
30: -69.0528107, 47.7474632, -69.1748505, 47.8337517, -116.8865662, 116.9223022
31: -70.6199036, 33.6073875, -70.8013992, 33.6826210, -104.3025208, 104.4087830
32: -65.1272736, 42.4701309, -65.2234344, 42.5709229, -107.6981964, 107.6935654
33: -84.5923157, 63.8433380, -84.7405701, 63.9777756, -148.5700684, 148.5838928
34: -76.0970917, 53.8452415, -76.2080536, 53.9327545, -130.0298462, 130.0532837
35: -67.2969971, 58.3604050, -67.4065704, 58.4702835, -125.7672806, 125.7669678
36: -70.5548553, 58.8514099, -70.6685638, 58.9535828, -129.5084381, 129.5199738
37: -105.0800171, 48.9347000, -105.2213593, 49.0158844, -154.0958862, 154.1560669
38: -97.5279846, 69.5440674, -97.6783829, 69.6759567, -167.2039185, 167.2224426
39: -104.4244690, 58.5167274, -104.6246796, 58.6781082, -163.1025696, 163.1413879
40: -93.3119507, 39.8368034, -93.4582672, 39.9335175, -133.2454529, 133.2950745
41: -67.3457642, 40.3328094, -67.4355927, 40.3837814, -107.7295456, 107.7684021
42: -53.0912361, 37.9546967, -53.1677895, 38.0090446, -91.1002808, 91.1224823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=505, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=501, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 966

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 599

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -94.1821696, upper bound: 94.2554917
time: 88.54 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -94.1821696, upper bound: 94.2554917
time: 88.73 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -79.9259949, 61.1727066, -79.9309921, 61.1745033, -141.1004944, 141.1036987
1: -50.0214767, 51.0972214, -50.0254059, 51.0989914, -101.1204681, 101.1226273
2: -42.4468842, 46.5979233, -42.4505310, 46.5998383, -89.0467224, 89.0484543
3: -47.3457489, 57.8539467, -47.3495026, 57.8563309, -105.2020645, 105.2034454
4: -51.7908325, 52.2536888, -51.7958412, 52.2554665, -104.0462952, 104.0495300
5: -50.8004227, 58.5069427, -50.8047562, 58.5094604, -109.3098755, 109.3116913
6: -74.8402176, 38.0292740, -74.8440247, 38.0317383, -112.8719482, 112.8732986
7: -64.9918976, 54.6998787, -64.9965668, 54.7022476, -119.6941452, 119.6964417
8: -60.1644554, 58.7993279, -60.1689606, 58.8016586, -118.9661102, 118.9682770
9: -46.0514870, 49.9684982, -46.0558815, 49.9708710, -96.0223541, 96.0243759
10: -75.2520905, 63.5637894, -75.2550583, 63.5675201, -138.8195953, 138.8188477
11: -78.1283188, 46.0436478, -78.1318436, 46.0486031, -124.1768951, 124.1754913
12: -79.5462112, 54.1870537, -79.5494843, 54.1899414, -133.7361450, 133.7365417
13: -62.0870476, 77.8647766, -62.0948334, 77.8670044, -139.9540405, 139.9595947
14: -119.2738800, 37.9789505, -119.2783813, 37.9845543, -157.2584381, 157.2573242
15: -58.1551170, 61.8371696, -58.1578331, 61.8392487, -119.9943542, 119.9950027
16: -85.5190735, 58.0420189, -85.5228653, 58.0471420, -143.5662231, 143.5648804
17: -119.7585602, 53.9816208, -119.7629242, 53.9837723, -173.7423401, 173.7445374
18: -75.0399017, 43.4098816, -75.0426331, 43.4152641, -118.4551697, 118.4525070
19: -59.4921799, 27.6346169, -59.4951630, 27.6378212, -87.1299973, 87.1297760
20: -51.0509300, 35.2731628, -51.0529709, 35.2779655, -86.3288956, 86.3261261
21: -71.6523132, 35.8493690, -71.6557846, 35.8540230, -107.5063248, 107.5051575
22: -68.7488556, 41.7358322, -68.7515869, 41.7392769, -110.4881287, 110.4874115
23: -55.1863670, 38.4782410, -55.1887703, 38.4838524, -93.6702194, 93.6670074
24: -58.7179298, 32.9108086, -58.7213326, 32.9150429, -91.6329651, 91.6321411
25: -50.9988518, 44.8870049, -51.0022316, 44.8915100, -95.8903656, 95.8892365
26: -86.4245071, 55.4399338, -86.4281769, 55.4447746, -141.8692780, 141.8681030
27: -69.2953949, 32.3295975, -69.2977982, 32.3344574, -101.6298523, 101.6273956
28: -55.4152260, 44.7092667, -55.4173889, 44.7140999, -100.1293259, 100.1266479
29: -70.0324402, 39.1259079, -70.0353241, 39.1306572, -109.1631012, 109.1612244
30: -69.1942291, 47.9057159, -69.1973724, 47.9099083, -117.1041260, 117.1030884
31: -70.8217773, 33.7461510, -70.8250504, 33.7495041, -104.5712814, 104.5711975
32: -65.2768860, 42.5949936, -65.2806091, 42.5982857, -107.8751526, 107.8755951
33: -84.8492432, 64.0083923, -84.8578644, 64.0106812, -148.8599243, 148.8662567
34: -76.2750244, 53.9450912, -76.2827454, 53.9476280, -130.2226562, 130.2278290
35: -67.4815979, 58.4920540, -67.4892120, 58.4937439, -125.9753418, 125.9812622
36: -70.7426910, 58.9697609, -70.7495270, 58.9714432, -129.7141418, 129.7192841
37: -105.3041153, 49.0407448, -105.3126678, 49.0423965, -154.3464966, 154.3534088
38: -97.7700119, 69.6954193, -97.7776184, 69.6975250, -167.4675293, 167.4730377
39: -104.7931595, 58.6915627, -104.8032532, 58.6929283, -163.4860840, 163.4947968
40: -93.5573578, 39.9509087, -93.5635376, 39.9520988, -133.5094452, 133.5144501
41: -67.4837570, 40.4042625, -67.4875641, 40.4066238, -107.8903809, 107.8918304
42: -53.1949043, 38.0324821, -53.1971397, 38.0350494, -91.2299423, 91.2296143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=505, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 599
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 966

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 599

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -94.1821696, upper bound: 94.2554917
time: 90.23 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -94.2574506, upper bound: 94.2574508
time: 104.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 197.29 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 197.29
Output dim: 13, lower bound: -94.1821696, upper bound: 94.2554917
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 197.29
Output dim: 13, lower bound: -94.1821696, upper bound: 94.2554917
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 197.29
Output dim: 13, lower bound: -94.1821696, upper bound: 94.2554917
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 197.29
Output dim: 13, lower bound: -94.2574506, upper bound: 94.2574508

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -79.7021332, 61.0491486, -79.7686462, 61.1251411, -140.8272552, 140.8177948
1: -49.8962212, 51.0099030, -49.9415550, 51.0564995, -100.9527130, 100.9514618
2: -42.3253555, 46.4767761, -42.3658829, 46.5530396, -88.8783951, 88.8426590
3: -47.1663284, 57.6730995, -47.2189941, 57.7909088, -104.9572372, 104.8920898
4: -51.5833168, 52.0738754, -51.6723824, 52.1766777, -103.7599869, 103.7462616
5: -50.6240768, 58.2946625, -50.6877975, 58.4332008, -109.0572662, 108.9824524
6: -74.6872101, 37.8375053, -74.7445984, 37.8832703, -112.5704727, 112.5820847
7: -64.8279724, 54.5509224, -64.8825378, 54.6471329, -119.4750977, 119.4334488
8: -59.9822922, 58.5984077, -60.0654755, 58.7197075, -118.7019806, 118.6638794
9: -45.9065895, 49.8816490, -45.9728279, 49.9134102, -95.8199844, 95.8544769
10: -75.0976028, 63.4361115, -75.1764984, 63.4839630, -138.5815582, 138.6126099
11: -77.8668671, 45.8395348, -77.9504089, 45.9272537, -123.7941208, 123.7899475
12: -79.3677750, 54.0255623, -79.4600220, 54.0659027, -133.4336853, 133.4855804
13: -61.7986755, 77.5418549, -61.9172058, 77.7444916, -139.5431671, 139.4590607
14: -118.9806671, 37.8629608, -119.0697479, 37.9178658, -156.8985291, 156.9327087
15: -58.0234795, 61.6904449, -58.0836639, 61.7722244, -119.7957001, 119.7741089
16: -85.3258209, 57.9048615, -85.3820724, 57.9540787, -143.2799072, 143.2869263
17: -119.4729004, 53.8466949, -119.5058060, 53.9181175, -173.3910217, 173.3524933
18: -74.8020172, 43.2003632, -74.9676208, 43.2871704, -118.0891800, 118.1679840
19: -59.2603607, 27.4993401, -59.3406143, 27.5535355, -86.8138885, 86.8399506
20: -50.9176521, 35.1604919, -50.9861183, 35.2028770, -86.1205292, 86.1466064
21: -71.4061127, 35.6677628, -71.5002365, 35.7413063, -107.1474152, 107.1679993
22: -68.5232239, 41.6078796, -68.5926056, 41.6580048, -110.1812286, 110.2004852
23: -54.9342079, 38.2552223, -55.0391235, 38.3524513, -93.2866592, 93.2943420
24: -58.4657516, 32.7458344, -58.5457230, 32.8148575, -91.2806015, 91.2915497
25: -50.7908287, 44.6995964, -50.8697472, 44.7780495, -95.5688782, 95.5693436
26: -86.1433029, 55.2395782, -86.3277054, 55.3132668, -141.4565735, 141.5672913
27: -69.0296707, 32.1594963, -69.1815491, 32.2324600, -101.2621307, 101.3410492
28: -55.1412506, 44.5139275, -55.2351036, 44.5936890, -99.7349243, 99.7490311
29: -69.7778854, 38.9372444, -69.8070374, 39.0155487, -108.7934341, 108.7442780
30: -68.9765625, 47.7330704, -69.0304794, 47.8066978, -116.7832642, 116.7635498
31: -70.5398102, 33.5971451, -70.6502991, 33.6633453, -104.2031555, 104.2474442
32: -65.1032486, 42.4191551, -65.1783905, 42.4738464, -107.5770874, 107.5975342
33: -84.5639191, 63.8147240, -84.6871643, 63.9243546, -148.4882507, 148.5018921
34: -76.0721588, 53.8251076, -76.1611328, 53.8947372, -129.9668884, 129.9862366
35: -67.2708664, 58.3465195, -67.3572845, 58.4441948, -125.7150574, 125.7038040
36: -70.5358276, 58.8036041, -70.6330261, 58.8646202, -129.4004517, 129.4366302
37: -105.0440292, 48.8680801, -105.1534424, 48.8922768, -153.9362946, 154.0215149
38: -97.5036545, 69.4440689, -97.6329422, 69.4868164, -166.9904633, 167.0769958
39: -104.3957214, 58.4492111, -104.5706406, 58.5514488, -162.9471741, 163.0198364
40: -93.2876129, 39.7492256, -93.4124374, 39.7675934, -133.0552063, 133.1616669
41: -67.3191986, 40.2873840, -67.3857269, 40.2973404, -107.6165314, 107.6731110
42: -53.0730972, 37.9184723, -53.1335411, 37.9428406, -91.0159378, 91.0520172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=501, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 601

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -94.1473596, upper bound: 94.2008854
time: 82.33 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -94.1473596, upper bound: 94.2205507
time: 99.40 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -79.7162704, 61.0557327, -79.9183960, 61.2216492, -140.9379272, 140.9741211
1: -49.9039192, 51.0143433, -50.0146637, 51.1454811, -101.0494003, 101.0290070
2: -42.3251419, 46.4845390, -42.4256935, 46.6683807, -88.9935150, 88.9102325
3: -47.1700363, 57.6854439, -47.3072968, 57.9196968, -105.0897064, 104.9927368
4: -51.5874252, 52.0941010, -51.7456474, 52.3018112, -103.8892288, 103.8397446
5: -50.6317177, 58.3084488, -50.7586746, 58.5576134, -109.1893158, 109.0671158
6: -74.7036133, 37.8821869, -75.0153580, 38.0182495, -112.7218399, 112.8975449
7: -64.8263779, 54.5536423, -64.9397278, 54.7729721, -119.5993500, 119.4933624
8: -59.9692307, 58.6138611, -60.0908623, 58.8568459, -118.8260803, 118.7047272
9: -45.9144745, 49.8913918, -46.1290398, 49.9901314, -95.9046021, 96.0204239
10: -75.1058807, 63.4452591, -75.3084869, 63.6112251, -138.7171021, 138.7537537
11: -77.8980103, 45.8366966, -78.1162109, 46.1452332, -124.0432434, 123.9529114
12: -79.3771057, 54.0193100, -79.7607574, 54.1333237, -133.5104218, 133.7800598
13: -61.8024826, 77.5571060, -62.2622910, 77.8445587, -139.6470337, 139.8193970
14: -119.0327454, 37.8656235, -119.2811279, 38.1638489, -157.1965790, 157.1467438
15: -58.0262260, 61.6983376, -58.1469345, 61.8599014, -119.8861237, 119.8452759
16: -85.3470612, 57.9092255, -85.5519409, 58.1142578, -143.4613190, 143.4611664
17: -119.5548401, 53.8420525, -119.7746277, 54.2758102, -173.8306274, 173.6166840
18: -74.8061600, 43.2081947, -75.0672302, 43.4901886, -118.2963409, 118.2754211
19: -59.3162537, 27.4980888, -59.5120239, 27.7347946, -87.0510406, 87.0101013
20: -50.9236565, 35.1665726, -51.0549698, 35.3634644, -86.2871246, 86.2215424
21: -71.4414978, 35.6668396, -71.6629791, 35.9559669, -107.3974609, 107.3298187
22: -68.5678101, 41.6079865, -68.7659454, 41.8667068, -110.4345169, 110.3739319
23: -54.9792862, 38.2544937, -55.1920242, 38.5936279, -93.5729141, 93.4465179
24: -58.5233612, 32.7488441, -58.7441597, 33.0895424, -91.6128922, 91.4930038
25: -50.8316994, 44.7035789, -51.0187263, 45.0238914, -95.8555908, 95.7223053
26: -86.1565094, 55.2491760, -86.4523315, 55.5066719, -141.6631775, 141.7015076
27: -69.0593414, 32.1630936, -69.3134766, 32.4752274, -101.5345688, 101.4765701
28: -55.2097206, 44.5163460, -55.4275398, 44.8655930, -100.0753098, 99.9438858
29: -69.8554077, 38.9318542, -70.0536499, 39.2758179, -109.1312256, 108.9855042
30: -69.0161896, 47.7337227, -69.1945190, 48.0965462, -117.1127319, 116.9282303
31: -70.6011581, 33.5963669, -70.8454437, 33.8807182, -104.4818726, 104.4418106
32: -65.1142654, 42.4557724, -65.5150604, 42.6107712, -107.7250290, 107.9708252
33: -84.5766678, 63.8189392, -84.9939346, 63.9841957, -148.5608673, 148.8128662
34: -76.0803680, 53.8191833, -76.3406677, 53.9357834, -130.0161438, 130.1598511
35: -67.2832108, 58.3433037, -67.5539246, 58.4637413, -125.7469254, 125.8972244
36: -70.5389862, 58.8150215, -70.8899612, 58.9272614, -129.4662476, 129.7049866
37: -105.0586777, 48.9015617, -105.5546188, 49.0072479, -154.0659027, 154.4561768
38: -97.5021973, 69.4926758, -97.9890518, 69.6672974, -167.1694946, 167.4817200
39: -104.4032288, 58.4738159, -104.9952774, 58.6484985, -163.0517273, 163.4690857
40: -93.2907486, 39.8176308, -93.7849579, 39.9454041, -133.2361450, 133.6025848
41: -67.3339844, 40.3190460, -67.7035980, 40.4125214, -107.7464905, 108.0226440
42: -53.0841866, 37.9380722, -53.3482056, 38.0389977, -91.1231689, 91.2862778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 966

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 601

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -94.1473596, upper bound: 94.2008854
time: 103.65 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -94.1473596, upper bound: 94.2205507
time: 420.34 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -79.8793106, 61.1578064, -79.8437576, 61.1463356, -141.0256500, 141.0015564
1: -49.9995346, 51.0851059, -49.9844170, 51.0760422, -101.0755692, 101.0695190
2: -42.4290543, 46.5831680, -42.4170303, 46.5720558, -89.0011139, 89.0001984
3: -47.3171272, 57.8309250, -47.2952080, 57.8128395, -105.1299667, 105.1261292
4: -51.7774124, 52.2205582, -51.7707481, 52.1939735, -103.9713898, 103.9913025
5: -50.7819519, 58.4825783, -50.7701645, 58.4633598, -109.2453156, 109.2527390
6: -74.8123245, 37.9679756, -74.7915192, 37.9153137, -112.7276382, 112.7594910
7: -64.9632874, 54.6850243, -64.9427185, 54.6741447, -119.6374283, 119.6277466
8: -60.1525803, 58.7702866, -60.1464844, 58.7468262, -118.8994064, 118.9167633
9: -46.0352783, 49.9497871, -46.0252762, 49.9354858, -95.9707489, 95.9750671
10: -75.2342987, 63.5375443, -75.2215652, 63.5180893, -138.7523804, 138.7591095
11: -78.0503235, 46.0291519, -77.9846649, 46.0212593, -124.0715790, 124.0138168
12: -79.5236511, 54.1441536, -79.5068970, 54.1094055, -133.6330566, 133.6510468
13: -62.0692368, 77.8107605, -62.0613632, 77.7699509, -139.8391876, 139.8721313
14: -119.1898727, 37.9667511, -119.1198730, 37.9616699, -157.1515350, 157.0866241
15: -58.1380653, 61.8135567, -58.1255569, 61.7948036, -119.9328690, 119.9391022
16: -85.4699173, 58.0212212, -85.4303284, 58.0077133, -143.4776154, 143.4515533
17: -119.6460419, 53.9628143, -119.5506210, 53.9485130, -173.5945435, 173.5134277
18: -75.0142288, 43.3951035, -74.9943466, 43.3873901, -118.4015961, 118.3894501
19: -59.4205551, 27.6245918, -59.3603668, 27.6189461, -87.0395050, 86.9849548
20: -51.0258408, 35.2586861, -51.0053749, 35.2506790, -86.2765198, 86.2640610
21: -71.5827026, 35.8367424, -71.5237503, 35.8302078, -107.4129028, 107.3604889
22: -68.6809769, 41.7241364, -68.6231461, 41.7172470, -110.3982162, 110.3472824
23: -55.1173782, 38.4680862, -55.0588722, 38.4648209, -93.5821915, 93.5269547
24: -58.6383400, 32.9012527, -58.5705681, 32.8970490, -91.5353851, 91.4718170
25: -50.9407616, 44.8728790, -50.8920479, 44.8651428, -95.8059082, 95.7649231
26: -86.3919373, 55.4164124, -86.3665009, 55.4006081, -141.7925415, 141.7829132
27: -69.2465439, 32.3206253, -69.2064667, 32.3175468, -101.5640869, 101.5270920
28: -55.3276978, 44.6958885, -55.2530594, 44.6890793, -100.0167694, 99.9489441
29: -69.9266357, 39.1142311, -69.8358383, 39.1087646, -109.0354004, 108.9500732
30: -69.1180115, 47.8913383, -69.0530090, 47.8828926, -117.0008926, 116.9443436
31: -70.7416992, 33.7359276, -70.6739349, 33.7302399, -104.4719391, 104.4098663
32: -65.2528839, 42.5439949, -65.2355957, 42.5012207, -107.7541046, 107.7795868
33: -84.8209381, 63.9797554, -84.8044968, 63.9572792, -148.7782135, 148.7842407
34: -76.2500992, 53.9249725, -76.2358856, 53.9095535, -130.1596375, 130.1608582
35: -67.4555206, 58.4781876, -67.4399490, 58.4676285, -125.9231415, 125.9181366
36: -70.7236786, 58.9219513, -70.7139740, 58.8825302, -129.6062012, 129.6359253
37: -105.2681351, 48.9741440, -105.2447433, 48.9188423, -154.1869812, 154.2188873
38: -97.7457581, 69.5954132, -97.7322159, 69.5083466, -167.2541046, 167.3276367
39: -104.7644119, 58.6240730, -104.7492065, 58.5663261, -163.3307190, 163.3732758
40: -93.5330505, 39.8633118, -93.5177307, 39.7862091, -133.3192596, 133.3810425
41: -67.4572220, 40.3588028, -67.4376984, 40.3201447, -107.7773666, 107.7964935
42: -53.1767578, 37.9962463, -53.1628571, 37.9688759, -91.1456299, 91.1591034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 601

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -94.1473596, upper bound: 94.2023413
time: 77.61 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -94.1473596, upper bound: 94.2220770
time: 96.26 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -79.8934479, 61.1643829, -79.9934998, 61.2428246, -141.1362610, 141.1578827
1: -50.0072212, 51.0895691, -50.0575142, 51.1650467, -101.1722717, 101.1470795
2: -42.4288788, 46.5909500, -42.4768448, 46.6873894, -89.1162720, 89.0677948
3: -47.3208008, 57.8432999, -47.3835106, 57.9416351, -105.2624359, 105.2268066
4: -51.7815514, 52.2407913, -51.8440323, 52.3190308, -104.1005859, 104.0848160
5: -50.7896042, 58.4963531, -50.8410530, 58.5878029, -109.3774109, 109.3374023
6: -74.8286896, 38.0126991, -75.0622864, 38.0502472, -112.8789215, 113.0749817
7: -64.9616852, 54.6877213, -64.9998932, 54.7999535, -119.7616425, 119.6876144
8: -60.1395187, 58.7857399, -60.1718941, 58.8839378, -119.0234528, 118.9576340
9: -46.0431404, 49.9595184, -46.1814995, 50.0121765, -96.0553131, 96.1410217
10: -75.2426224, 63.5466881, -75.3535538, 63.6453476, -138.8879700, 138.9002380
11: -78.0814667, 46.0263023, -78.1504364, 46.2392883, -124.3207550, 124.1767426
12: -79.5330276, 54.1378860, -79.8076706, 54.1768074, -133.7098236, 133.9455566
13: -62.0730362, 77.8260193, -62.4064980, 77.8699875, -139.9430237, 140.2325134
14: -119.2419968, 37.9693756, -119.3311081, 38.2077103, -157.4497070, 157.3004761
15: -58.1407928, 61.8214302, -58.1888313, 61.8824844, -120.0232773, 120.0102539
16: -85.4912109, 58.0256195, -85.6001358, 58.1679535, -143.6591644, 143.6257629
17: -119.7279968, 53.9581909, -119.8194122, 54.3062592, -174.0342560, 173.7776031
18: -75.0183563, 43.4029236, -75.0939484, 43.5904083, -118.6087418, 118.4968719
19: -59.4764519, 27.6233234, -59.5317459, 27.8002243, -87.2766724, 87.1550674
20: -51.0318642, 35.2647781, -51.0742188, 35.4113274, -86.4431839, 86.3389969
21: -71.6180573, 35.8358383, -71.6864777, 36.0448837, -107.6629410, 107.5223160
22: -68.7255554, 41.7242050, -68.7964172, 41.9259605, -110.6515045, 110.5206223
23: -55.1624374, 38.4673615, -55.2117577, 38.7060242, -93.8684616, 93.6791153
24: -58.6959686, 32.9042549, -58.7689667, 33.1717720, -91.8677216, 91.6732178
25: -50.9816322, 44.8768768, -51.0410271, 45.1110153, -96.0926437, 95.9179001
26: -86.4051285, 55.4259758, -86.4910507, 55.5940475, -141.9991608, 141.9170227
27: -69.2762604, 32.3242264, -69.3383408, 32.5603180, -101.8365707, 101.6625671
28: -55.3961449, 44.6982803, -55.4454155, 44.9610367, -100.3571777, 100.1436920
29: -70.0041733, 39.1088600, -70.0823975, 39.3690262, -109.3731995, 109.1912537
30: -69.1576080, 47.8919983, -69.2170258, 48.1727448, -117.3303528, 117.1090088
31: -70.8030472, 33.7351303, -70.8690643, 33.9476700, -104.7507172, 104.6041946
32: -65.2638855, 42.5806084, -65.5722961, 42.6381149, -107.9020004, 108.1529083
33: -84.8336182, 63.9839897, -85.1115875, 64.0170670, -148.8506622, 149.0955811
34: -76.2583160, 53.9190750, -76.4154434, 53.9505997, -130.2089081, 130.3345032
35: -67.4678650, 58.4749985, -67.6366653, 58.4871941, -125.9550400, 126.1116638
36: -70.7268448, 58.9333687, -70.9709778, 58.9451523, -129.6719971, 129.9043427
37: -105.2827682, 49.0075989, -105.6460419, 49.0337143, -154.3164825, 154.6536407
38: -97.7442551, 69.6440125, -98.0884323, 69.6888199, -167.4330750, 167.7324524
39: -104.7718811, 58.6486511, -105.1740417, 58.6632996, -163.4351807, 163.8226929
40: -93.5361938, 39.9317207, -93.8903046, 39.9639244, -133.5001068, 133.8220215
41: -67.4720001, 40.3904800, -67.7556076, 40.4353409, -107.9073410, 108.1460876
42: -53.1878548, 38.0158691, -53.3775597, 38.0649910, -91.2528458, 91.3934174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=504, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 1752
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1751
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 1721
type: A, layer: 1, pos: 1645
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 1636
type: A, layer: 1, pos: 1756
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1694
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 1738
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 632
type: A, layer: 1, pos: 1767
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 1755
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 1768
type: A, layer: 1, pos: 845
type: A, layer: 1, pos: 1754
type: A, layer: 1, pos: 1661
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 599
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 1724
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 646
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 693
type: A, layer: 1, pos: 725
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1769
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 1771
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1740
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 947
type: A, layer: 1, pos: 1651
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 585
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1735
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1781
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 1676
type: A, layer: 1, pos: 1772
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 828
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 923
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1787
type: A, layer: 1, pos: 1723
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 1788
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1715
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 1789
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 867
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 1390
type: A, layer: 1, pos: 963
type: A, layer: 1, pos: 1469
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1736
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 1739
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1741
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 946
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1416
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 1305
type: A, layer: 1, pos: 758
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 962
type: A, layer: 1, pos: 1643
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 939
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 1406
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 617
type: A, layer: 1, pos: 941
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 957
type: A, layer: 1, pos: 1749
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 1501
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 662
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 603
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 650
type: A, layer: 1, pos: 1304
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 961
type: A, layer: 1, pos: 1782
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1417
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 1733
type: A, layer: 1, pos: 728
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 1362
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1368
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 954
type: A, layer: 1, pos: 1371
type: A, layer: 1, pos: 1319
type: A, layer: 1, pos: 742
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 1492
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 1717
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 1367
type: A, layer: 1, pos: 1468
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 1753
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 1329
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1365
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1434
type: A, layer: 1, pos: 1750
type: A, layer: 1, pos: 615
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 995
type: A, layer: 1, pos: 1299
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1525
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 1524
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1395
type: A, layer: 1, pos: 586
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1399
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 629
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 1609
type: A, layer: 1, pos: 1391
type: A, layer: 1, pos: 1011
type: A, layer: 1, pos: 1322
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1608
type: A, layer: 1, pos: 1480
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 1690
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 1674
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 1453
type: A, layer: 1, pos: 1437
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 1343
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1378
type: A, layer: 1, pos: 980
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 1372
type: A, layer: 1, pos: 635
type: A, layer: 1, pos: 1502
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1439
type: A, layer: 1, pos: 1508
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 1345
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1500
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 630
type: A, layer: 1, pos: 882
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 993
type: A, layer: 1, pos: 1428
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 1507
type: A, layer: 1, pos: 1360
type: A, layer: 1, pos: 1484
type: A, layer: 1, pos: 1432
type: A, layer: 1, pos: 1667
type: A, layer: 1, pos: 1485
type: A, layer: 1, pos: 1361
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 1523
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 780
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 1610
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 1607
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1516
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 1483
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 678
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 812
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1431
type: A, layer: 1, pos: 1770
type: A, layer: 1, pos: 938
type: A, layer: 1, pos: 1008
type: A, layer: 1, pos: 1532
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1499
type: A, layer: 1, pos: 712
type: A, layer: 1, pos: 1517
type: A, layer: 1, pos: 1476
type: A, layer: 1, pos: 1308
type: A, layer: 1, pos: 651
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 945
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 762
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 1692
type: A, layer: 1, pos: 1505
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 1382
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 1398
type: A, layer: 1, pos: 1533
type: A, layer: 1, pos: 612
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 1449
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 1460
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 955
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 1387
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 1411
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 1448
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 1407
type: A, layer: 1, pos: 1298
type: A, layer: 1, pos: 1394
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 1370
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1766
type: A, layer: 1, pos: 1290
type: A, layer: 1, pos: 1734
type: A, layer: 1, pos: 1313
type: A, layer: 1, pos: 1506
type: A, layer: 1, pos: 1351
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 1531
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 798
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 1442
type: A, layer: 1, pos: 1282
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 1288
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 1423
type: A, layer: 1, pos: 1534
type: A, layer: 1, pos: 924
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1488
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 990
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 994
type: A, layer: 1, pos: 1377
type: A, layer: 1, pos: 1297
type: A, layer: 1, pos: 811
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 1514
type: A, layer: 1, pos: 1457
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 619
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 1022
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1504
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 1515
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1369
type: A, layer: 1, pos: 992
type: A, layer: 1, pos: 1383
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 614
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1366
type: A, layer: 1, pos: 960
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 1349
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 1006
type: A, layer: 1, pos: 1447
type: A, layer: 1, pos: 1627
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1691
type: A, layer: 1, pos: 1310
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 1023
type: A, layer: 1, pos: 1386
type: A, layer: 1, pos: 1415
type: A, layer: 1, pos: 1287
type: A, layer: 1, pos: 1443
type: A, layer: 1, pos: 1452
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 1653
type: A, layer: 1, pos: 1783
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 1444
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 631
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1518
type: A, layer: 1, pos: 1350
type: A, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 601

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 13, lower bound: -94.2220769, upper bound: 94.2023413
time: 91.09 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 13, lower bound: -94.2220769, upper bound: 94.2220770
time: 75.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 169.13 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 169.13
Output dim: 13, lower bound: -94.1473596, upper bound: 94.2008854
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 169.13
Output dim: 13, lower bound: -94.1473596, upper bound: 94.2205507
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 169.13
Output dim: 13, lower bound: -94.1473596, upper bound: 94.2008854
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 169.13
Output dim: 13, lower bound: -94.1473596, upper bound: 94.2205507
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 169.13
Output dim: 13, lower bound: -94.1473596, upper bound: 94.2023413
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 169.13
Output dim: 13, lower bound: -94.1473596, upper bound: 94.2220770
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 169.13
Output dim: 13, lower bound: -94.2220769, upper bound: 94.2023413
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 169.13
Output dim: 13, lower bound: -94.2220769, upper bound: 94.2220770

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -79.6117401, 61.0148354, -79.7207870, 61.1068573, -140.7185974, 140.7356262
1: -49.8428688, 50.9838791, -49.9131470, 51.0426483, -100.8854980, 100.8970184
2: -42.2930679, 46.4516907, -42.3484306, 46.5395889, -88.8326569, 88.8001175
3: -47.1054535, 57.6372833, -47.1867065, 57.7716866, -104.8771362, 104.8239899
4: -51.5472374, 52.0237732, -51.6533661, 52.1498222, -103.6970596, 103.6771393
5: -50.5879288, 58.2531891, -50.6686935, 58.4107857, -108.9987030, 108.9218750
6: -74.6493454, 37.6989746, -74.7244568, 37.8093033, -112.4586487, 112.4234238
7: -64.7601929, 54.5200272, -64.8468857, 54.6306686, -119.3908539, 119.3669128
8: -59.9385643, 58.5475121, -60.0421219, 58.6922379, -118.6307983, 118.5896301
9: -45.8642883, 49.8480110, -45.9505615, 49.8955383, -95.7598267, 95.7985687
10: -75.0551605, 63.3941269, -75.1540298, 63.4617538, -138.5169067, 138.5481567
11: -77.7321625, 45.8089676, -77.8787308, 45.9109917, -123.6431580, 123.6876984
12: -79.3255157, 53.9162521, -79.4376678, 54.0069771, -133.3324890, 133.3539124
13: -61.7594681, 77.4282379, -61.8964577, 77.6833191, -139.4427643, 139.3246765
14: -118.8615341, 37.8383484, -119.0068970, 37.9049530, -156.7664795, 156.8452454
15: -57.9693413, 61.6508904, -58.0550919, 61.7512169, -119.7205582, 119.7059784
16: -85.2427216, 57.8554688, -85.3379822, 57.9279480, -143.1706390, 143.1934509
17: -119.3048859, 53.8130608, -119.4168396, 53.9002457, -173.2051086, 173.2298889
18: -74.7609406, 43.1604805, -74.9458389, 43.2661133, -118.0270538, 118.1063232
19: -59.1475182, 27.4832764, -59.2796249, 27.5449829, -86.6925049, 86.7629013
20: -50.8745956, 35.1313705, -50.9631004, 35.1875000, -86.0620880, 86.0944672
21: -71.2494965, 35.6499710, -71.4167480, 35.7318268, -106.9813232, 107.0667191
22: -68.3847504, 41.5879669, -68.5192032, 41.6474571, -110.0322113, 110.1071625
23: -54.8274460, 38.2363892, -54.9816628, 38.3424530, -93.1698914, 93.2180481
24: -58.3118820, 32.7294998, -58.4640350, 32.8061905, -91.1180573, 91.1935349
25: -50.6671867, 44.6776276, -50.8033295, 44.7664223, -95.4336090, 95.4809570
26: -86.0914383, 55.1966629, -86.3002090, 55.2906570, -141.3820953, 141.4968719
27: -68.9675903, 32.1336555, -69.1485138, 32.2188301, -101.1864166, 101.2821655
28: -55.0249290, 44.4879074, -55.1730461, 44.5798569, -99.6047821, 99.6609497
29: -69.5757294, 38.9088478, -69.6997986, 39.0005035, -108.5762329, 108.6086426
30: -68.7899475, 47.7097626, -68.9310532, 47.7942734, -116.5841904, 116.6408157
31: -70.4137421, 33.5771828, -70.5835724, 33.6527061, -104.0664444, 104.1607513
32: -65.0625000, 42.2416382, -65.1567688, 42.3786469, -107.4411469, 107.3983917
33: -84.5043564, 63.7759590, -84.6556091, 63.9038773, -148.4082336, 148.4315643
34: -76.0336227, 53.7538033, -76.1407166, 53.8572388, -129.8908691, 129.8945007
35: -67.2266541, 58.3070145, -67.3338852, 58.4232178, -125.6498642, 125.6408997
36: -70.5100708, 58.6352615, -70.6193542, 58.7763748, -129.2864380, 129.2546082
37: -104.9869995, 48.7974777, -105.1232605, 48.8552284, -153.8422241, 153.9207306
38: -97.4636765, 69.1543579, -97.6117020, 69.3316040, -166.7952881, 166.7660522
39: -104.3490601, 58.3256989, -104.5458603, 58.4863434, -162.8354034, 162.8715515
40: -93.2464905, 39.5924911, -93.3906555, 39.6837616, -132.9302521, 132.9831543
41: -67.2872620, 40.1580582, -67.3687973, 40.2292557, -107.5165100, 107.5268555
42: -53.0435524, 37.8449783, -53.1178436, 37.9036827, -90.9472351, 90.9628220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=501, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1647307
time: 87.53 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1645517
time: 89.54 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -79.7555542, 61.1212234, -79.7352676, 61.1175041, -140.8730469, 140.8564758
1: -49.9173470, 51.0931664, -49.9212265, 51.0483093, -100.9656448, 101.0143890
2: -42.3383255, 46.5456238, -42.3430939, 46.5466766, -88.8850021, 88.8887177
3: -47.2209854, 57.7598534, -47.2053680, 57.7824860, -105.0034714, 104.9652100
4: -51.6132507, 52.1411476, -51.6610909, 52.1666985, -103.7799530, 103.8022385
5: -50.6503487, 58.3655853, -50.6670418, 58.4242325, -109.0745773, 109.0326233
6: -74.8917465, 37.8571014, -74.7365646, 37.8669395, -112.7586823, 112.5936661
7: -64.8326416, 54.6492348, -64.8471985, 54.6376190, -119.4702606, 119.4964294
8: -59.9875336, 58.7065506, -60.0334396, 58.7073669, -118.6949005, 118.7399826
9: -45.9876251, 49.9096870, -45.9624672, 49.9034538, -95.8910828, 95.8721542
10: -75.2525864, 63.4993286, -75.1672974, 63.4682426, -138.7208252, 138.6666260
11: -77.9205780, 46.0418816, -77.9169388, 45.9146194, -123.8351974, 123.9588165
12: -79.6347046, 54.0027924, -79.4476929, 54.0101318, -133.6448364, 133.4504852
13: -62.1193428, 77.5536880, -61.9021454, 77.7046814, -139.8240204, 139.4558411
14: -119.0654297, 38.0330620, -119.0456619, 37.9106064, -156.9760437, 157.0787048
15: -58.0570526, 61.7729225, -58.0701904, 61.7649422, -119.8219910, 119.8431091
16: -85.3920059, 58.0316620, -85.3539581, 57.9410133, -143.3330231, 143.3856201
17: -119.5476303, 54.1143494, -119.4814682, 53.9010620, -173.4487000, 173.5958252
18: -74.8808212, 43.3572922, -74.9566956, 43.2765884, -118.1574097, 118.3139877
19: -59.2937851, 27.6639252, -59.3237114, 27.5467796, -86.8405609, 86.9876404
20: -50.9574471, 35.2590981, -50.9673424, 35.1956024, -86.1530457, 86.2264404
21: -71.4689026, 35.9029999, -71.4745483, 35.7325134, -107.2014160, 107.3775406
22: -68.5856934, 41.7849312, -68.5725555, 41.6508255, -110.2365189, 110.3574829
23: -54.9678154, 38.4488678, -55.0208626, 38.3465958, -93.3144073, 93.4697266
24: -58.5179214, 32.9895325, -58.5274582, 32.8096008, -91.3275223, 91.5169830
25: -50.8412094, 44.8953857, -50.8535919, 44.7722473, -95.6134567, 95.7489777
26: -86.2316132, 55.3712006, -86.3159103, 55.3015366, -141.5331421, 141.6871033
27: -69.0822144, 32.3943176, -69.1640320, 32.2260666, -101.3082809, 101.5583496
28: -55.1671600, 44.7642593, -55.2176170, 44.5863228, -99.7534790, 99.9818726
29: -69.8285599, 39.1673317, -69.7815170, 39.0034752, -108.8320312, 108.9488297
30: -69.0276489, 48.0485077, -69.0038910, 47.7981949, -116.8258057, 117.0523987
31: -70.5930176, 33.7749252, -70.6299515, 33.6553192, -104.2483368, 104.4048691
32: -65.4476471, 42.4513512, -65.1667786, 42.4537354, -107.9013672, 107.6181335
33: -84.7556610, 63.8666954, -84.6726227, 63.9155731, -148.6712341, 148.5393219
34: -76.2500153, 53.8309898, -76.1477661, 53.8672409, -130.1172485, 129.9787598
35: -67.4122314, 58.3539276, -67.3454285, 58.4263840, -125.8386154, 125.6993561
36: -70.8166046, 58.7757072, -70.6214752, 58.8178139, -129.6344147, 129.3971863
37: -105.2579041, 48.8622437, -105.1390839, 48.8633842, -154.1212769, 154.0013275
38: -97.9549179, 69.4570160, -97.6177673, 69.4463654, -167.4012756, 167.0747833
39: -104.7929001, 58.4474258, -104.5552216, 58.5258331, -163.3187256, 163.0026550
40: -93.5095673, 39.7622833, -93.3994980, 39.7495689, -133.2591248, 133.1617737
41: -67.5906067, 40.3251610, -67.3783722, 40.2834244, -107.8740234, 107.7035217
42: -53.2888260, 37.9572868, -53.1274643, 37.9298859, -91.2187119, 91.0847473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=500, inp2_unstable=501, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1850823
time: 94.99 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1849985
time: 74.10 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -79.6257553, 61.0214119, -79.8706665, 61.2034378, -140.8291931, 140.8920746
1: -49.8506393, 50.9883575, -49.9862366, 51.1316910, -100.9823303, 100.9745865
2: -42.2928391, 46.4595490, -42.4083595, 46.6550560, -88.9478912, 88.8679047
3: -47.1090889, 57.6497345, -47.2756310, 57.9005356, -105.0096283, 104.9253693
4: -51.5514030, 52.0436172, -51.7266846, 52.2746506, -103.8260498, 103.7703018
5: -50.5955887, 58.2671967, -50.7396851, 58.5354500, -109.1310425, 109.0068817
6: -74.6657333, 37.7434807, -74.9952240, 37.9441452, -112.6098785, 112.7387085
7: -64.7586212, 54.5227280, -64.9040680, 54.7565155, -119.5151367, 119.4267960
8: -59.9255524, 58.5632820, -60.0675926, 58.8296318, -118.7551575, 118.6308746
9: -45.8722954, 49.8577423, -46.1069641, 49.9722824, -95.8445740, 95.9646988
10: -75.0635452, 63.4032707, -75.2859955, 63.5890007, -138.6525421, 138.6892700
11: -77.7637482, 45.8061523, -78.0446625, 46.1291885, -123.8929367, 123.8508148
12: -79.3349152, 53.9099388, -79.7384033, 54.0744743, -133.4093933, 133.6483459
13: -61.7632065, 77.4429855, -62.2417908, 77.7829590, -139.5461426, 139.6847839
14: -118.9135513, 37.8409996, -119.2179947, 38.1509666, -157.0645142, 157.0589905
15: -57.9721146, 61.6592102, -58.1183167, 61.8388252, -119.8109436, 119.7775269
16: -85.2645264, 57.8598404, -85.5080109, 58.0880356, -143.3525696, 143.3678589
17: -119.3863754, 53.8084183, -119.6854706, 54.2580261, -173.6444092, 173.4938965
18: -74.7651367, 43.1683731, -75.0452957, 43.4693565, -118.2344818, 118.2136688
19: -59.2032204, 27.4819870, -59.4509583, 27.7262993, -86.9295044, 86.9329453
20: -50.8805809, 35.1374130, -51.0319099, 35.3481216, -86.2286987, 86.1693192
21: -71.2842789, 35.6489716, -71.5791016, 35.9465141, -107.2307892, 107.2280731
22: -68.4290771, 41.5880775, -68.6922607, 41.8561554, -110.2852325, 110.2803345
23: -54.8721085, 38.2356682, -55.1342125, 38.5836945, -93.4558029, 93.3698730
24: -58.3692474, 32.7325096, -58.6622581, 33.0809326, -91.4501801, 91.3947525
25: -50.7074394, 44.6816483, -50.9517555, 45.0122643, -95.7197037, 95.6333923
26: -86.1045151, 55.2062378, -86.4247589, 55.4841537, -141.5886536, 141.6309967
27: -68.9972534, 32.1372910, -69.2803802, 32.4617538, -101.4589996, 101.4176636
28: -55.0931778, 44.4903030, -55.3650208, 44.8518677, -99.9450378, 99.8553238
29: -69.6526718, 38.9035034, -69.9458389, 39.2608376, -108.9134979, 108.8493347
30: -68.8291626, 47.7104340, -69.0945740, 48.0842743, -116.9134369, 116.8050079
31: -70.4749069, 33.5764008, -70.7786255, 33.8701401, -104.3450317, 104.3550262
32: -65.0735016, 42.2781143, -65.4935532, 42.5153961, -107.5888824, 107.7716675
33: -84.5170135, 63.7801781, -84.9627151, 63.9636383, -148.4806366, 148.7428894
34: -76.0418701, 53.7479897, -76.3200531, 53.8983040, -129.9401703, 130.0680237
35: -67.2390594, 58.3038368, -67.5303802, 58.4427223, -125.6817780, 125.8342133
36: -70.5132141, 58.6467628, -70.8763428, 58.8386383, -129.3518524, 129.5231018
37: -105.0016785, 48.8308487, -105.5244522, 48.9701004, -153.9717712, 154.3553009
38: -97.4622421, 69.2015076, -97.9678879, 69.5111160, -166.9733582, 167.1693878
39: -104.3565598, 58.3502846, -104.9709549, 58.5831451, -162.9396973, 163.3212280
40: -93.2496643, 39.6604843, -93.7631989, 39.8612213, -133.1108856, 133.4236755
41: -67.3020325, 40.1896210, -67.6866608, 40.3444023, -107.6464386, 107.8762817
42: -53.0546570, 37.8642311, -53.3325157, 37.9996719, -91.0543289, 91.1967468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1329319, upper bound: 94.1647307
time: 76.13 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1645517
time: 82.85 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -79.7697220, 61.1277466, -79.8850021, 61.2139778, -140.9836731, 141.0127563
1: -49.9253349, 51.0976181, -49.9941444, 51.1372604, -101.0625916, 101.0917664
2: -42.3381500, 46.5532379, -42.4029770, 46.6620407, -89.0001907, 88.9562073
3: -47.2248001, 57.7720108, -47.2936249, 57.9112968, -105.1360931, 105.0656357
4: -51.6173477, 52.1609879, -51.7345238, 52.2918243, -103.9091644, 103.8955002
5: -50.6580200, 58.3789024, -50.7380142, 58.5486984, -109.2067108, 109.1168976
6: -74.9080505, 37.9017372, -75.0073090, 38.0018616, -112.9099045, 112.9090271
7: -64.8313751, 54.6518745, -64.9045258, 54.7634163, -119.5947876, 119.5563965
8: -59.9748192, 58.7218170, -60.0586891, 58.8445854, -118.8194046, 118.7805023
9: -45.9953423, 49.9194412, -46.1186981, 49.9803200, -95.9756470, 96.0381393
10: -75.2608643, 63.5085487, -75.2993011, 63.5959167, -138.8567505, 138.8078461
11: -77.9522247, 46.0390472, -78.0826263, 46.1325264, -124.0847473, 124.1216736
12: -79.6441193, 53.9962654, -79.7484283, 54.0775871, -133.7216949, 133.7446899
13: -62.1229210, 77.5680466, -62.2473373, 77.8043518, -139.9272614, 139.8153687
14: -119.1173401, 38.0355644, -119.2568283, 38.1565704, -157.2739105, 157.2923889
15: -58.0597267, 61.7811775, -58.1333427, 61.8525658, -119.9122925, 119.9145203
16: -85.4135590, 58.0361023, -85.5237274, 58.1011620, -143.5147095, 143.5598297
17: -119.6291504, 54.1097031, -119.7501526, 54.2587509, -173.8878937, 173.8598633
18: -74.8854599, 43.3651199, -75.0562820, 43.4796333, -118.3650970, 118.4213943
19: -59.3495522, 27.6626549, -59.4951134, 27.7280407, -87.0775909, 87.1577682
20: -50.9635315, 35.2650909, -51.0362282, 35.3561935, -86.3197250, 86.3013153
21: -71.5039368, 35.9020157, -71.6371918, 35.9471664, -107.4510956, 107.5391998
22: -68.6300812, 41.7850113, -68.7458191, 41.8595123, -110.4895859, 110.5308228
23: -55.0126457, 38.4481049, -55.1736450, 38.5877609, -93.6004028, 93.6217499
24: -58.5754051, 32.9924431, -58.7258072, 33.0842743, -91.6596832, 91.7182465
25: -50.8817673, 44.8993111, -51.0024567, 45.0180397, -95.8998032, 95.9017639
26: -86.2444458, 55.3806305, -86.4404526, 55.4949493, -141.7393799, 141.8210754
27: -69.1118240, 32.3978348, -69.2957611, 32.4688797, -101.5807037, 101.6935959
28: -55.2355804, 44.7666321, -55.4100151, 44.8582649, -100.0938416, 100.1766434
29: -69.9057617, 39.1619339, -70.0280304, 39.2636719, -109.1694336, 109.1899567
30: -69.0672302, 48.0490913, -69.1678772, 48.0881004, -117.1553345, 117.2169647
31: -70.6541824, 33.7741470, -70.8250275, 33.8727150, -104.5268936, 104.5991592
32: -65.4585876, 42.4878464, -65.5034332, 42.5905380, -108.0491180, 107.9912643
33: -84.7681580, 63.8709564, -84.9794235, 63.9753609, -148.7435150, 148.8503571
34: -76.2582703, 53.8251915, -76.3271561, 53.9082756, -130.1665497, 130.1523438
35: -67.4244690, 58.3508453, -67.5419922, 58.4459229, -125.8703918, 125.8928299
36: -70.8196945, 58.7873459, -70.8784027, 58.8803482, -129.7000427, 129.6657410
37: -105.2724152, 48.8952484, -105.5401459, 48.9779968, -154.2503967, 154.4353943
38: -97.9534836, 69.5048218, -97.9738083, 69.6261368, -167.5796204, 167.4786224
39: -104.8001785, 58.4721756, -104.9798813, 58.6227684, -163.4229279, 163.4520569
40: -93.5127716, 39.8305359, -93.7720184, 39.9273033, -133.4400787, 133.6025391
41: -67.6053085, 40.3567123, -67.6962204, 40.3985901, -108.0038834, 108.0529327
42: -53.2998314, 37.9761734, -53.3421173, 38.0257950, -91.3256226, 91.3182907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=500, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1850823
time: 83.24 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1849985
time: 80.39 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -79.7889175, 61.1234703, -79.7959137, 61.1280060, -140.9169312, 140.9193726
1: -49.9461670, 51.0591202, -49.9560318, 51.0622101, -101.0083618, 101.0151520
2: -42.3967896, 46.5580750, -42.3995895, 46.5586052, -88.9553833, 88.9576645
3: -47.2562943, 57.7950706, -47.2629318, 57.7935905, -105.0498810, 105.0579987
4: -51.7413406, 52.1703987, -51.7517776, 52.1671181, -103.9084625, 103.9221725
5: -50.7458153, 58.4410591, -50.7510681, 58.4409523, -109.1867676, 109.1921234
6: -74.7744827, 37.8294868, -74.7713928, 37.8413544, -112.6158295, 112.6008759
7: -64.8955078, 54.6541176, -64.9070282, 54.6577072, -119.5532150, 119.5611420
8: -60.1089249, 58.7193565, -60.1231918, 58.7193413, -118.8282623, 118.8425446
9: -45.9929276, 49.9161110, -46.0029869, 49.9175873, -95.9105148, 95.9190979
10: -75.1918564, 63.4955101, -75.1991043, 63.4958725, -138.6877136, 138.6946106
11: -77.9155426, 45.9986191, -77.9129715, 46.0050583, -123.9206009, 123.9115906
12: -79.4813232, 54.0347900, -79.4844818, 54.0504723, -133.5317993, 133.5192719
13: -62.0300560, 77.6971207, -62.0406151, 77.7088013, -139.7388611, 139.7377319
14: -119.0706863, 37.9420624, -119.0569687, 37.9487190, -157.0193787, 156.9990234
15: -58.0838814, 61.7739220, -58.0969925, 61.7737846, -119.8576660, 119.8709106
16: -85.3867798, 57.9718513, -85.3862457, 57.9815483, -143.3683319, 143.3580933
17: -119.4779358, 53.9291306, -119.4617157, 53.9306717, -173.4085999, 173.3908386
18: -74.9730988, 43.3552170, -74.9725800, 43.3663063, -118.3394012, 118.3277893
19: -59.3077278, 27.6085606, -59.2993774, 27.6104183, -86.9181442, 86.9079361
20: -50.9827499, 35.2295952, -50.9823570, 35.2353363, -86.2180862, 86.2119446
21: -71.4260559, 35.8189926, -71.4403076, 35.8207092, -107.2467651, 107.2593002
22: -68.5424805, 41.7042389, -68.5497589, 41.7066879, -110.2491684, 110.2539902
23: -55.0106392, 38.4492874, -55.0014153, 38.4548340, -93.4654694, 93.4506989
24: -58.4844360, 32.8849335, -58.4888687, 32.8883896, -91.3728180, 91.3737869
25: -50.8170624, 44.8509598, -50.8256531, 44.8535080, -95.6705627, 95.6766052
26: -86.3400269, 55.3735352, -86.3389969, 55.3779907, -141.7180023, 141.7125244
27: -69.1844330, 32.2948227, -69.1734314, 32.3039169, -101.4883423, 101.4682541
28: -55.2113571, 44.6698761, -55.1909981, 44.6752472, -99.8866043, 99.8608627
29: -69.7244644, 39.0858841, -69.7285919, 39.0937042, -108.8181686, 108.8144760
30: -68.9313278, 47.8680000, -68.9535828, 47.8704185, -116.8017426, 116.8215790
31: -70.6155777, 33.7159882, -70.6072083, 33.7196159, -104.3351898, 104.3231964
32: -65.2121582, 42.3664780, -65.2140045, 42.4060249, -107.6181641, 107.5804749
33: -84.7613678, 63.9409523, -84.7729950, 63.9367752, -148.6981506, 148.7139435
34: -76.2115707, 53.8536644, -76.2154465, 53.8720627, -130.0836334, 130.0691071
35: -67.4113312, 58.4386749, -67.4165421, 58.4466705, -125.8579941, 125.8552094
36: -70.6979599, 58.7536621, -70.7003326, 58.7942657, -129.4922180, 129.4539948
37: -105.2110901, 48.9034805, -105.2145309, 48.8817596, -154.0928497, 154.1180115
38: -97.7057419, 69.3056641, -97.7109756, 69.3531570, -167.0588989, 167.0166321
39: -104.7178345, 58.5004807, -104.7244568, 58.5011597, -163.2189789, 163.2249451
40: -93.4919739, 39.7065506, -93.4959641, 39.7024002, -133.1943665, 133.2025146
41: -67.4252930, 40.2294579, -67.4207916, 40.2520866, -107.6773834, 107.6502533
42: -53.1472130, 37.9227448, -53.1471558, 37.9297180, -91.0769196, 91.0698853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1647307
time: 651.97 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1659513
time: 90.58 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -79.9327545, 61.2298279, -79.8103561, 61.1386757, -141.0714264, 141.0401917
1: -50.0206528, 51.1683464, -49.9640427, 51.0678558, -101.0885010, 101.1323853
2: -42.4420433, 46.6519775, -42.3942413, 46.5657234, -89.0077667, 89.0462189
3: -47.3717880, 57.9175949, -47.2815742, 57.8044395, -105.1762238, 105.1991577
4: -51.8073921, 52.2877655, -51.7594833, 52.1839600, -103.9913483, 104.0472488
5: -50.8082733, 58.5533981, -50.7494011, 58.4544182, -109.2626953, 109.3027954
6: -75.0168915, 37.9875565, -74.7835083, 37.8989487, -112.9158401, 112.7710648
7: -64.9680099, 54.7832756, -64.9073105, 54.6645889, -119.6325989, 119.6905823
8: -60.1579132, 58.8783112, -60.1144562, 58.7344933, -118.8924103, 118.9927673
9: -46.1162949, 49.9777794, -46.0149002, 49.9255219, -96.0418091, 95.9926758
10: -75.3892517, 63.6007690, -75.2123489, 63.5023499, -138.8916016, 138.8131104
11: -78.1038742, 46.2315254, -77.9511871, 46.0086555, -124.1125183, 124.1827087
12: -79.7905121, 54.1213417, -79.4945679, 54.0536118, -133.8441162, 133.6159058
13: -62.3899841, 77.8225555, -62.0463104, 77.7301559, -140.1201477, 139.8688660
14: -119.2744980, 38.1368294, -119.0957413, 37.9543457, -157.2288513, 157.2325745
15: -58.1715813, 61.8959122, -58.1120834, 61.7875404, -119.9591217, 120.0079880
16: -85.5359421, 58.1481133, -85.4022217, 57.9946938, -143.5306396, 143.5503235
17: -119.7206650, 54.2304459, -119.5262909, 53.9314575, -173.6520996, 173.7567444
18: -75.0930023, 43.5520630, -74.9834290, 43.3767853, -118.4697723, 118.5354919
19: -59.4539337, 27.7892075, -59.3434563, 27.6122074, -87.0661392, 87.1326599
20: -51.0655785, 35.3573456, -50.9866066, 35.2434311, -86.3090057, 86.3439484
21: -71.6454239, 36.0720215, -71.4980850, 35.8214035, -107.4668274, 107.5701065
22: -68.7433929, 41.9012032, -68.6030731, 41.7100906, -110.4534836, 110.5042725
23: -55.1509819, 38.6617813, -55.0406075, 38.4589767, -93.6099548, 93.7023849
24: -58.6904411, 33.1449394, -58.5523071, 32.8917923, -91.5822296, 91.6972504
25: -50.9910698, 45.0687218, -50.8758850, 44.8593521, -95.8504028, 95.9446106
26: -86.4801254, 55.5480995, -86.3546753, 55.3888664, -141.8689880, 141.9027710
27: -69.2990265, 32.5555038, -69.1889496, 32.3111572, -101.6101837, 101.7444458
28: -55.3535118, 44.9462852, -55.2355728, 44.6817589, -100.0352478, 100.1818542
29: -69.9772186, 39.3443336, -69.8102951, 39.0966911, -109.0738983, 109.1546326
30: -69.1689911, 48.2067680, -69.0264130, 47.8743706, -117.0433502, 117.2331772
31: -70.7948227, 33.9137802, -70.6535797, 33.7222252, -104.5170441, 104.5673599
32: -65.5973206, 42.5760918, -65.2239532, 42.4811058, -108.0784302, 107.8000488
33: -85.0128479, 64.0316620, -84.7899857, 63.9484940, -148.9613190, 148.8216400
34: -76.4281006, 53.9307976, -76.2224655, 53.8820496, -130.3101501, 130.1532593
35: -67.5969696, 58.4855347, -67.4281082, 58.4498062, -126.0467758, 125.9136429
36: -71.0045471, 58.8940048, -70.7024765, 58.8356934, -129.8402405, 129.5964813
37: -105.4821014, 48.9682465, -105.2303696, 48.8899002, -154.3719788, 154.1986084
38: -98.1970215, 69.6082001, -97.7170944, 69.4678955, -167.6649170, 167.3252869
39: -105.1617737, 58.6222267, -104.7338028, 58.5406532, -163.7024231, 163.3560333
40: -93.7551270, 39.8763046, -93.5048065, 39.7681732, -133.5233002, 133.3811035
41: -67.7286377, 40.3965340, -67.4303360, 40.3062668, -108.0349045, 107.8268661
42: -53.3924561, 38.0350037, -53.1567764, 37.9559250, -91.3483810, 91.1917801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=502, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1864686
time: 87.31 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1863951
time: 81.99 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -79.8029709, 61.1300659, -79.9458008, 61.2245750, -141.0275421, 141.0758667
1: -49.9539452, 51.0636024, -50.0290947, 51.1512375, -101.1051788, 101.0926971
2: -42.3965645, 46.5659447, -42.4595261, 46.6740685, -89.0706253, 89.0254669
3: -47.2598915, 57.8075371, -47.3518524, 57.9224625, -105.1823578, 105.1593933
4: -51.7454681, 52.1902504, -51.8250847, 52.2918968, -104.0373535, 104.0153351
5: -50.7534676, 58.4551010, -50.8220749, 58.5656395, -109.3191071, 109.2771759
6: -74.7908478, 37.8740044, -75.0421600, 37.9761124, -112.7669525, 112.9161453
7: -64.8939590, 54.6567993, -64.9642105, 54.7835197, -119.6774750, 119.6210022
8: -60.0959091, 58.7351570, -60.1486359, 58.8567200, -118.9526215, 118.8837814
9: -46.0009689, 49.9258461, -46.1594086, 49.9943504, -95.9953156, 96.0852509
10: -75.2002335, 63.5046539, -75.3310852, 63.6231194, -138.8233337, 138.8357391
11: -77.9471359, 45.9957619, -78.0788727, 46.2232361, -124.1703720, 124.0746307
12: -79.4907150, 54.0284996, -79.7853622, 54.1179657, -133.6086731, 133.8138580
13: -62.0338173, 77.7118912, -62.3860092, 77.8083649, -139.8421783, 140.0978851
14: -119.1226578, 37.9447174, -119.2680054, 38.1947899, -157.3174438, 157.2127228
15: -58.0866814, 61.7822227, -58.1602402, 61.8613701, -119.9480515, 119.9424591
16: -85.4086151, 57.9762650, -85.5562057, 58.1416969, -143.5503082, 143.5324707
17: -119.5594559, 53.9245453, -119.7302551, 54.2884712, -173.8479309, 173.6548004
18: -74.9772949, 43.3631134, -75.0720215, 43.5696030, -118.5468826, 118.4351349
19: -59.3634338, 27.6072617, -59.4706841, 27.7917309, -87.1551666, 87.0779419
20: -50.9887581, 35.2356415, -51.0511589, 35.3959885, -86.3847275, 86.2867966
21: -71.4608307, 35.8180046, -71.6026154, 36.0354080, -107.4962387, 107.4206238
22: -68.5867767, 41.7043533, -68.7227478, 41.9154167, -110.5021896, 110.4271011
23: -55.0552826, 38.4485626, -55.1539497, 38.6960831, -93.7513580, 93.6025009
24: -58.5418243, 32.8879547, -58.6870651, 33.1631393, -91.7049561, 91.5750122
25: -50.8573265, 44.8549538, -50.9740295, 45.0994034, -95.9567261, 95.8289795
26: -86.3530960, 55.3831177, -86.4634552, 55.5715027, -141.9245911, 141.8465576
27: -69.2141190, 32.2984390, -69.3052063, 32.5468369, -101.7609558, 101.6036377
28: -55.2795982, 44.6722717, -55.3829231, 44.9472961, -100.2268982, 100.0551910
29: -69.8014221, 39.0805435, -69.9745865, 39.3540192, -109.1554413, 109.0551300
30: -68.9705658, 47.8687134, -69.1170578, 48.1604958, -117.1310577, 116.9857712
31: -70.6767731, 33.7151985, -70.8022461, 33.9370537, -104.6138306, 104.5174408
32: -65.2231293, 42.4029465, -65.5508041, 42.5427055, -107.7658157, 107.9537430
33: -84.7740021, 63.9452019, -85.0803680, 63.9964905, -148.7704620, 149.0255737
34: -76.2198181, 53.8478394, -76.3948059, 53.9131546, -130.1329651, 130.2426300
35: -67.4237518, 58.4354630, -67.6131363, 58.4661522, -125.8898926, 126.0485992
36: -70.7010651, 58.7651215, -70.9573669, 58.8564835, -129.5575409, 129.7224884
37: -105.2257614, 48.9368401, -105.6158981, 48.9965973, -154.2223511, 154.5527344
38: -97.7043304, 69.3528442, -98.0672760, 69.5326233, -167.2369537, 167.4201050
39: -104.7252960, 58.5251236, -105.1497116, 58.5980148, -163.3233032, 163.6748047
40: -93.4951859, 39.7745590, -93.8685608, 39.8797951, -133.3749847, 133.6431274
41: -67.4400864, 40.2610359, -67.7386551, 40.3672409, -107.8073273, 107.9996948
42: -53.1583023, 37.9420013, -53.3618507, 38.0256844, -91.1839752, 91.3038483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=502, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1863950, upper bound: 94.1661122
time: 94.82 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1863950, upper bound: 94.1659513
time: 82.53 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -79.9469147, 61.2363434, -79.9600601, 61.2351265, -141.1820374, 141.1964111
1: -50.0286789, 51.1728134, -50.0370140, 51.1567993, -101.1854782, 101.2098236
2: -42.4418716, 46.6595955, -42.4541168, 46.6810303, -89.1229019, 89.1137085
3: -47.3756027, 57.9297867, -47.3698425, 57.9332504, -105.3088531, 105.2996292
4: -51.8114738, 52.3075638, -51.8329201, 52.3090744, -104.1205444, 104.1404877
5: -50.8159103, 58.5666809, -50.8203812, 58.5788727, -109.3947830, 109.3870621
6: -75.0331879, 38.0321503, -75.0542068, 38.0338745, -113.0670624, 113.0863495
7: -64.9666977, 54.7858963, -64.9646759, 54.7904015, -119.7570801, 119.7505646
8: -60.1452103, 58.8935432, -60.1397057, 58.8716354, -119.0168457, 119.0332489
9: -46.1240082, 49.9875679, -46.1711693, 50.0023956, -96.1264038, 96.1587372
10: -75.3975372, 63.6099243, -75.3443604, 63.6300240, -139.0275574, 138.9542847
11: -78.1355438, 46.2287102, -78.1168518, 46.2265778, -124.3621063, 124.3455505
12: -79.7998810, 54.1147995, -79.7954025, 54.1210518, -133.9209290, 133.9101868
13: -62.3936272, 77.8369141, -62.3915100, 77.8297882, -140.2234192, 140.2284241
14: -119.3263855, 38.1393204, -119.3068466, 38.2003860, -157.5267639, 157.4461670
15: -58.1742630, 61.9041862, -58.1752586, 61.8751602, -120.0494232, 120.0794449
16: -85.5575943, 58.1525421, -85.5719147, 58.1548462, -143.7124329, 143.7244568
17: -119.8022079, 54.2258759, -119.7949600, 54.2892075, -174.0914154, 174.0208282
18: -75.0976105, 43.5598984, -75.0830154, 43.5798950, -118.6774902, 118.6429138
19: -59.5096970, 27.7879238, -59.5148201, 27.7934608, -87.3031616, 87.3027420
20: -51.0716858, 35.3633728, -51.0554771, 35.4040756, -86.4757614, 86.4188385
21: -71.6804581, 36.0710373, -71.6606674, 36.0361023, -107.7165451, 107.7317047
22: -68.7877655, 41.9012833, -68.7763214, 41.9187813, -110.7065353, 110.6776047
23: -55.1958389, 38.6610031, -55.1933975, 38.7001419, -93.8959808, 93.8543930
24: -58.7479477, 33.1478577, -58.7505951, 33.1665077, -91.9144440, 91.8984528
25: -51.0316544, 45.0726280, -51.0247345, 45.1051941, -96.1368484, 96.0973663
26: -86.4929886, 55.5575066, -86.4791336, 55.5822830, -142.0752716, 142.0366364
27: -69.3286591, 32.5590134, -69.3206329, 32.5539703, -101.8826141, 101.8796463
28: -55.4219513, 44.9486313, -55.4279060, 44.9536743, -100.3756256, 100.3765411
29: -70.0544281, 39.3389587, -70.0567627, 39.3568840, -109.4113159, 109.3957214
30: -69.2085876, 48.2073746, -69.1903839, 48.1642914, -117.3728790, 117.3977509
31: -70.8560181, 33.9129410, -70.8486633, 33.9396744, -104.7956924, 104.7616043
32: -65.6082458, 42.6125793, -65.5606537, 42.6178513, -108.2260742, 108.1732254
33: -85.0253448, 64.0359497, -85.0970917, 64.0082474, -149.0335846, 149.1330414
34: -76.4363098, 53.9250374, -76.4019318, 53.9231262, -130.3594360, 130.3269653
35: -67.6092224, 58.4824791, -67.6247406, 58.4693680, -126.0785675, 126.1072235
36: -71.0075989, 58.9056396, -70.9594269, 58.8981705, -129.9057617, 129.8650513
37: -105.4966049, 49.0011673, -105.6316223, 49.0045166, -154.5011292, 154.6327820
38: -98.1956024, 69.6560059, -98.0731583, 69.6476669, -167.8432617, 167.7291565
39: -105.1690216, 58.6469345, -105.1586380, 58.6375580, -163.8065796, 163.8055725
40: -93.7583008, 39.9445534, -93.8773727, 39.9458618, -133.7041626, 133.8219299
41: -67.7433624, 40.4280777, -67.7482147, 40.4213943, -108.1647568, 108.1762924
42: -53.4034958, 38.0538712, -53.3714600, 38.0517960, -91.4552917, 91.4253235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=503, inp2_unstable=504, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=501, inp2_unstable=503, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=44
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 1752
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1751
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 1721
type: B, layer: 1, pos: 1645
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 1636
type: B, layer: 1, pos: 1756
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1694
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 1738
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 632
type: B, layer: 1, pos: 1767
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 1755
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1725
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 845
type: B, layer: 1, pos: 1768
type: B, layer: 1, pos: 1754
type: B, layer: 1, pos: 1661
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 1724
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 646
type: B, layer: 1, pos: 693
type: B, layer: 1, pos: 725
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1769
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1771
type: B, layer: 1, pos: 1740
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 947
type: B, layer: 1, pos: 1651
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 585
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 1735
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 1781
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 1676
type: B, layer: 1, pos: 1772
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 923
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 1723
type: B, layer: 1, pos: 1787
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 1788
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1715
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 1789
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 867
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1390
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 963
type: B, layer: 1, pos: 1469
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1736
type: B, layer: 1, pos: 1739
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1741
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 946
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1416
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 1305
type: B, layer: 1, pos: 758
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 962
type: B, layer: 1, pos: 1643
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 939
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1406
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 941
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 617
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 957
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 1749
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1501
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 662
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 603
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 650
type: B, layer: 1, pos: 1304
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 961
type: B, layer: 1, pos: 1782
type: B, layer: 1, pos: 1417
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 1733
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1362
type: B, layer: 1, pos: 728
type: B, layer: 1, pos: 1368
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 954
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1371
type: B, layer: 1, pos: 1319
type: B, layer: 1, pos: 742
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 1492
type: B, layer: 1, pos: 1717
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 1468
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1367
type: B, layer: 1, pos: 1753
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 1329
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 1365
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1434
type: B, layer: 1, pos: 1750
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 1299
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 995
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 615
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 1525
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1524
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 586
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 1395
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 1608
type: B, layer: 1, pos: 1399
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 1609
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 1391
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 629
type: B, layer: 1, pos: 1322
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1011
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 1480
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 1690
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 1674
type: B, layer: 1, pos: 1453
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 1437
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 1343
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1378
type: B, layer: 1, pos: 980
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 1372
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 1502
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 635
type: B, layer: 1, pos: 1439
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 1508
type: B, layer: 1, pos: 1345
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1500
type: B, layer: 1, pos: 630
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 882
type: B, layer: 1, pos: 993
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 1428
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 1507
type: B, layer: 1, pos: 1360
type: B, layer: 1, pos: 1484
type: B, layer: 1, pos: 1432
type: B, layer: 1, pos: 1485
type: B, layer: 1, pos: 1361
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 1667
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 1523
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 1607
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1610
type: B, layer: 1, pos: 780
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1516
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1483
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 678
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 812
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1431
type: B, layer: 1, pos: 1770
type: B, layer: 1, pos: 938
type: B, layer: 1, pos: 1008
type: B, layer: 1, pos: 1532
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 712
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1499
type: B, layer: 1, pos: 1517
type: B, layer: 1, pos: 1476
type: B, layer: 1, pos: 1308
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 651
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 945
type: B, layer: 1, pos: 1505
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 762
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1692
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 1382
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 1398
type: B, layer: 1, pos: 1533
type: B, layer: 1, pos: 612
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 1449
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 1460
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 955
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1387
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 1411
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 1448
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 1298
type: B, layer: 1, pos: 1407
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 1394
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 1370
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 1766
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1290
type: B, layer: 1, pos: 1313
type: B, layer: 1, pos: 1734
type: B, layer: 1, pos: 1506
type: B, layer: 1, pos: 1351
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 1531
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1442
type: B, layer: 1, pos: 1282
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 798
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 1288
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1369
type: B, layer: 1, pos: 1534
type: B, layer: 1, pos: 1423
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 924
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 1488
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 990
type: B, layer: 1, pos: 1377
type: B, layer: 1, pos: 1297
type: B, layer: 1, pos: 614
type: B, layer: 1, pos: 1457
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 811
type: B, layer: 1, pos: 1514
type: B, layer: 1, pos: 994
type: B, layer: 1, pos: 631
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 1022
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 619
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 1504
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1383
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 1515
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 992
type: B, layer: 1, pos: 1366
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 960
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 1349
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 1006
type: B, layer: 1, pos: 1447
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1627
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1691
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 1310
type: B, layer: 1, pos: 1023
type: B, layer: 1, pos: 1386
type: B, layer: 1, pos: 1783
type: B, layer: 1, pos: 1287
type: B, layer: 1, pos: 1415
type: B, layer: 1, pos: 1443
type: B, layer: 1, pos: 1452
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 1653
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 1444
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 1518
type: B, layer: 1, pos: 1350
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 966

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 664

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1864686
time: 93.96 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1863951
time: 93.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 189.86 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1647307
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1645517
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1850823
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1849985
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1329319, upper bound: 94.1647307
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1645517
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1850823
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1849985
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1647307
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1659513
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1864686
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1863951
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1863950, upper bound: 94.1661122
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1863950, upper bound: 94.1659513
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1864686
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 189.86
Output dim: 13, lower bound: -94.1120421, upper bound: 94.1863951
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=139.97683715820312
rel_dist={13: [-94.2982147551906, 94.2982147548251]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 11943.64 seconds
