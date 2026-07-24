## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 6)
Time budget: 3600 seconds
Split limit: 100
Threshold: 63.9952334073


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=412, inp2_unstable=412, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-74.4441681, 51.2501221, -74.4441681, 51.2501221, -125.6942902, 125.6942825)
1: (-35.7569427, 43.2586746, -35.7569427, 43.2586746, -79.0156097, 79.0156097)
2: (-30.5907345, 44.9403458, -30.5907345, 44.9403458, -75.5310822, 75.5310822)
3: (-35.0810623, 49.6928406, -35.0810623, 49.6928406, -84.7739029, 84.7739029)
4: (-40.9552193, 51.8817978, -40.9552193, 51.8817978, -92.8370209, 92.8370132)
5: (-34.2261429, 47.9517632, -34.2261429, 47.9517632, -82.1779022, 82.1779022)
6: (-70.8315430, 41.0467148, -70.8315430, 41.0467148, -111.8782501, 111.8782578)
7: (-43.1463509, 45.9498024, -43.1463509, 45.9498024, -89.0961456, 89.0961533)
8: (-48.8081589, 63.1175346, -48.8081589, 63.1175346, -111.9256897, 111.9256897)
9: (-41.9876938, 45.6585083, -41.9876938, 45.6585083, -87.6461945, 87.6462021)
10: (-63.0807076, 57.5979233, -63.0807076, 57.5979233, -120.6786346, 120.6786270)
11: (-60.7652092, 33.4686699, -60.7652092, 33.4686699, -94.2338791, 94.2338791)
12: (-67.9794083, 38.1531181, -67.9794083, 38.1531181, -106.1325226, 106.1325226)
13: (-65.2255249, 61.3555145, -65.2255249, 61.3555145, -126.5810394, 126.5810394)
14: (-101.7791443, 45.9462204, -101.7791443, 45.9462204, -147.7253571, 147.7253723)
15: (-49.4616508, 45.3876305, -49.4616508, 45.3876305, -94.8492737, 94.8492813)
16: (-63.1231842, 43.5457993, -63.1231842, 43.5457993, -106.6689758, 106.6689758)
17: (-97.0565567, 40.5237808, -97.0565567, 40.5237808, -137.5803375, 137.5803223)
18: (-64.3055725, 40.1181068, -64.3055725, 40.1181068, -104.4236755, 104.4236755)
19: (-46.9785995, 28.0002918, -46.9785995, 28.0002918, -74.9788895, 74.9788895)
20: (-45.3601379, 29.8137321, -45.3601379, 29.8137321, -75.1738663, 75.1738663)
21: (-59.0123825, 31.9141216, -59.0123825, 31.9141216, -90.9265060, 90.9265060)
22: (-58.7619171, 34.4748764, -58.7619171, 34.4748764, -93.2367859, 93.2367935)
23: (-46.6954193, 36.9088135, -46.6954193, 36.9088135, -83.6042328, 83.6042328)
24: (-57.0340157, 37.7480583, -57.0340157, 37.7480583, -94.7820740, 94.7820740)
25: (-50.1265450, 38.0982132, -50.1265450, 38.0982132, -88.2247543, 88.2247620)
26: (-70.8370209, 45.6955872, -70.8370209, 45.6955872, -116.5326080, 116.5326080)
27: (-57.3656578, 39.1326370, -57.3656578, 39.1326370, -96.4982910, 96.4982910)
28: (-46.4588852, 38.8290672, -46.4588852, 38.8290672, -85.2879486, 85.2879486)
29: (-60.1362267, 32.4817734, -60.1362267, 32.4817734, -92.6179962, 92.6179962)
30: (-58.1264915, 41.4087524, -58.1264915, 41.4087524, -99.5352478, 99.5352402)
31: (-62.8394241, 38.8694992, -62.8394241, 38.8694992, -101.7089233, 101.7089157)
32: (-68.7925339, 35.4185715, -68.7925339, 35.4185715, -104.2111053, 104.2111053)
33: (-87.5928345, 47.7316704, -87.5928345, 47.7316704, -135.3244934, 135.3244934)
34: (-74.3134232, 31.5479107, -74.3134232, 31.5479107, -105.8613281, 105.8613281)
35: (-72.0159607, 37.8115234, -72.0159607, 37.8115234, -109.8274689, 109.8274841)
36: (-75.3385239, 38.7858200, -75.3385239, 38.7858200, -114.1243439, 114.1243362)
37: (-106.4644547, 30.5911770, -106.4644547, 30.5911770, -137.0556335, 137.0556335)
38: (-88.1569672, 41.6150589, -88.1569672, 41.6150589, -129.7720337, 129.7720184)
39: (-98.3843613, 45.1909943, -98.3843613, 45.1909943, -143.5753326, 143.5753479)
40: (-81.9499817, 30.8376102, -81.9499817, 30.8376102, -112.7875900, 112.7875900)
41: (-69.2990265, 43.2153130, -69.2990265, 43.2153130, -112.5143280, 112.5143356)
42: (-50.8413391, 35.0825958, -50.8413391, 35.0825958, -85.9239349, 85.9239349)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.34 + 143.60 = 145.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -64.0592927, upper bound: 64.0592927

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 941

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 723

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0575321, upper bound: 63.9960450
time: 88.00 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0580918, upper bound: 64.0580916
time: 83.04 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 171.16 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 171.16
Output dim: 8, lower bound: -64.0575321, upper bound: 63.9960450
IS_A2, status: Status.UNKNOWN, split count: 1, time: 171.16
Output dim: 8, lower bound: -64.0580918, upper bound: 64.0580916

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -74.3905945, 51.0240097, -74.4120026, 51.1142921, -125.5048828, 125.4360123
1: -35.7303162, 43.1126099, -35.7409515, 43.1715965, -78.9019165, 78.8535614
2: -30.5691147, 44.8014603, -30.5777950, 44.8575134, -75.4266281, 75.3792572
3: -35.0576019, 49.5772018, -35.0670242, 49.6236267, -84.6812286, 84.6442184
4: -40.9235992, 51.7546005, -40.9363403, 51.8059235, -92.7295227, 92.6909409
5: -34.2031326, 47.8147087, -34.2123451, 47.8698502, -82.0729828, 82.0270538
6: -70.7342224, 40.9991531, -70.7733765, 41.0182610, -111.7524796, 111.7725296
7: -43.1157913, 45.8261299, -43.1280136, 45.8760452, -88.9918365, 88.9541473
8: -48.7745361, 62.9206161, -48.7880669, 63.0001221, -111.7746582, 111.7086792
9: -41.9534225, 45.5163498, -41.9671707, 45.5736389, -87.5270538, 87.4835205
10: -63.0326309, 57.4816399, -63.0518074, 57.5279770, -120.5605927, 120.5334396
11: -60.6135292, 33.4330177, -60.6743279, 33.4473305, -94.0608597, 94.1073456
12: -67.9186401, 38.0894279, -67.9430008, 38.1150284, -106.0336685, 106.0324249
13: -65.1754150, 61.2341614, -65.1954803, 61.2826118, -126.4580231, 126.4296341
14: -101.7190323, 45.7728195, -101.7430267, 45.8429184, -147.5619507, 147.5158386
15: -49.4201965, 45.2793655, -49.4368362, 45.3227463, -94.7429428, 94.7162018
16: -63.0542221, 43.4355927, -63.0817604, 43.4783783, -106.5326004, 106.5173492
17: -96.9860077, 40.3814354, -97.0141144, 40.4381943, -137.4241791, 137.3955383
18: -64.2043762, 40.0881042, -64.2448120, 40.1001434, -104.3045197, 104.3329163
19: -46.8372307, 27.9819145, -46.8939705, 27.9892654, -74.8264923, 74.8758850
20: -45.2045250, 29.7889805, -45.2671967, 29.7988567, -75.0033798, 75.0561752
21: -58.8205795, 31.8892574, -58.8977814, 31.8992310, -90.7198105, 90.7870331
22: -58.5629501, 34.4470367, -58.6431999, 34.4581757, -93.0211182, 93.0902405
23: -46.5705605, 36.8779449, -46.6207390, 36.8903351, -83.4608917, 83.4986725
24: -56.8472252, 37.7267532, -56.9221764, 37.7352753, -94.5824890, 94.6489258
25: -49.9304390, 38.0675011, -50.0093307, 38.0798492, -88.0102844, 88.0768280
26: -70.7267303, 45.6649780, -70.7709427, 45.6772079, -116.4039383, 116.4359207
27: -57.2176247, 39.1083908, -57.2771950, 39.1180954, -96.3357239, 96.3855743
28: -46.3113251, 38.8055878, -46.3707199, 38.8149796, -85.1262970, 85.1763077
29: -59.9621239, 32.4537582, -60.0322495, 32.4650116, -92.4271393, 92.4860001
30: -57.9149399, 41.3714943, -58.0000458, 41.3864098, -99.3013458, 99.3715363
31: -62.6390343, 38.8423386, -62.7195168, 38.8532448, -101.4922791, 101.5618439
32: -68.7052917, 35.3669739, -68.7402039, 35.3875160, -104.0928040, 104.1071777
33: -87.3954163, 47.6895370, -87.4749146, 47.7064056, -135.1018219, 135.1644440
34: -74.1956635, 31.5172806, -74.2420578, 31.5295296, -105.7251892, 105.7593384
35: -71.8767395, 37.7787704, -71.9322357, 37.7918472, -109.6685791, 109.7110062
36: -75.2197723, 38.7557755, -75.2676926, 38.7678070, -113.9875793, 114.0234604
37: -106.3247147, 30.5547466, -106.3808899, 30.5693207, -136.8940277, 136.9356384
38: -88.0258713, 41.5614548, -88.0786209, 41.5829086, -129.6087646, 129.6400604
39: -98.2192535, 45.1470337, -98.2844925, 45.1646500, -143.3838959, 143.4315186
40: -81.8588943, 30.7500610, -81.8954620, 30.7850685, -112.6439667, 112.6455231
41: -69.2440948, 43.1754646, -69.2662354, 43.1914749, -112.4355698, 112.4416962
42: -50.7952423, 35.0366516, -50.8137779, 35.0548630, -85.8501053, 85.8504181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=411, inp2_unstable=412, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 941

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1785

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0084104, upper bound: 63.9907623
time: 85.97 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0084104, upper bound: 63.9907623
time: 361.19 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -74.5740585, 51.2540779, -74.4371490, 51.2388725, -125.8129272, 125.6912231
1: -35.8476944, 43.2620239, -35.7532959, 43.2518463, -79.0995331, 79.0153198
2: -30.6987381, 44.9436493, -30.5877762, 44.9340897, -75.6328278, 75.5314178
3: -35.1449966, 49.6941833, -35.0778084, 49.6861343, -84.8311234, 84.7719879
4: -41.0581245, 51.8857346, -40.9518013, 51.8752289, -92.9333496, 92.8375244
5: -34.2855797, 47.9586678, -34.2226639, 47.9450073, -82.2305756, 82.1813354
6: -70.8358841, 41.0902176, -70.8246231, 41.0396461, -111.8755264, 111.9148407
7: -43.2369804, 45.9519806, -43.1424103, 45.9441795, -89.1811600, 89.0943909
8: -48.9674454, 63.1196976, -48.8044930, 63.1087761, -112.0762024, 111.9241867
9: -42.0727005, 45.6620216, -41.9840431, 45.6517563, -87.7244568, 87.6460648
10: -63.1739044, 57.6085968, -63.0750542, 57.5891571, -120.7630463, 120.6836472
11: -60.7779655, 33.5136795, -60.7559052, 33.4647827, -94.2427444, 94.2695847
12: -67.9861832, 38.2062454, -67.9739075, 38.1481895, -106.1343689, 106.1801376
13: -65.2874146, 61.3678589, -65.2192993, 61.3465500, -126.6339645, 126.5871582
14: -101.8784714, 45.9411354, -101.7697372, 45.9335251, -147.8119812, 147.7108765
15: -49.5192947, 45.3981247, -49.4569778, 45.3810043, -94.9002838, 94.8551025
16: -63.1994514, 43.5507050, -63.1161842, 43.5330162, -106.7324677, 106.6668777
17: -97.1099777, 40.5310364, -97.0475922, 40.5155869, -137.6255493, 137.5786285
18: -64.3094711, 40.1777878, -64.2978516, 40.1143761, -104.4238358, 104.4756393
19: -46.9876060, 28.0663452, -46.9705811, 27.9976196, -74.9852295, 75.0369263
20: -45.3715057, 29.8957958, -45.3521805, 29.8101215, -75.1816254, 75.2479706
21: -59.0231934, 32.0203934, -59.0021782, 31.9107285, -90.9339218, 91.0225677
22: -58.7717285, 34.5868454, -58.7526207, 34.4714127, -93.2431412, 93.3394623
23: -46.7071304, 36.9654579, -46.6882591, 36.9055138, -83.6126404, 83.6537094
24: -57.0412827, 37.8248444, -57.0251236, 37.7449837, -94.7862701, 94.8499680
25: -50.1368523, 38.2333221, -50.1179771, 38.0943413, -88.2311935, 88.3513031
26: -70.8497620, 45.7769775, -70.8265991, 45.6912804, -116.5410461, 116.6035767
27: -57.3731728, 39.1983871, -57.3574104, 39.1295357, -96.5027008, 96.5557938
28: -46.4695282, 38.9156723, -46.4513855, 38.8254089, -85.2949371, 85.3670578
29: -60.1427422, 32.5522842, -60.1271248, 32.4784355, -92.6211624, 92.6794052
30: -58.1380310, 41.5365028, -58.1166077, 41.4043846, -99.5424118, 99.6531067
31: -62.8494644, 38.9728699, -62.8295860, 38.8658104, -101.7152710, 101.8024597
32: -68.8025818, 35.4465408, -68.7862396, 35.4116859, -104.2142639, 104.2327728
33: -87.6208344, 47.8276291, -87.5807037, 47.7269859, -135.3478241, 135.4083252
34: -74.3281784, 31.6419621, -74.3071442, 31.5441437, -105.8723145, 105.9491043
35: -72.0335999, 37.9132309, -72.0089417, 37.8073425, -109.8409424, 109.9221649
36: -75.3423615, 38.8796272, -75.3318176, 38.7825966, -114.1249542, 114.2114410
37: -106.4905548, 30.6090736, -106.4554062, 30.5852680, -137.0758209, 137.0644836
38: -88.1630402, 41.6903877, -88.1483002, 41.6098785, -129.7729187, 129.8386841
39: -98.4156418, 45.2173500, -98.3716812, 45.1865234, -143.6021576, 143.5890198
40: -81.9721680, 30.8572464, -81.9430847, 30.8312531, -112.8033981, 112.8003311
41: -69.3071594, 43.2188187, -69.2940063, 43.1913338, -112.4984894, 112.5128250
42: -50.8469505, 35.1005630, -50.8366852, 35.0704880, -85.9174347, 85.9372406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=411, inp2_unstable=412, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1785
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 941

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1785

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0089626, upper bound: 64.0528098
time: 126.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0089626, upper bound: 64.0528098
time: 96.20 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 224.50 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 224.50
Output dim: 8, lower bound: -64.0084104, upper bound: 63.9907623
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 224.50
Output dim: 8, lower bound: -64.0084104, upper bound: 63.9907623
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 224.50
Output dim: 8, lower bound: -64.0089626, upper bound: 64.0528098
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 224.50
Output dim: 8, lower bound: -64.0089626, upper bound: 64.0528098

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -74.3600769, 51.0190544, -74.3594513, 51.1056976, -125.4657745, 125.3784943
1: -35.7209854, 43.0935593, -35.7248459, 43.1385803, -78.8595657, 78.8184052
2: -30.5634327, 44.7769699, -30.5680656, 44.8151894, -75.3786087, 75.3450317
3: -35.0527687, 49.5450630, -35.0586624, 49.5682144, -84.6209869, 84.6037216
4: -40.9175606, 51.7101364, -40.9258118, 51.7291679, -92.6467285, 92.6359482
5: -34.1966629, 47.7886734, -34.2012062, 47.8246155, -82.0212708, 81.9898834
6: -70.7143784, 40.9924622, -70.7390747, 41.0066223, -111.7210007, 111.7315369
7: -43.1061707, 45.7863541, -43.1114883, 45.8073044, -88.9134674, 88.8978424
8: -48.7691879, 62.8474770, -48.7787933, 62.8727341, -111.6419067, 111.6262665
9: -41.9483147, 45.4963531, -41.9583588, 45.5387192, -87.4870300, 87.4547119
10: -63.0240784, 57.4720345, -63.0370331, 57.5113220, -120.5353928, 120.5090561
11: -60.5675812, 33.4280396, -60.5950432, 33.4386826, -94.0062637, 94.0230789
12: -67.8831558, 38.0776062, -67.8818054, 38.0944977, -105.9776459, 105.9594116
13: -65.1673584, 61.2019691, -65.1815643, 61.2271004, -126.3944550, 126.3835297
14: -101.6972580, 45.7670898, -101.7053070, 45.8329239, -147.5301819, 147.4723969
15: -49.4113998, 45.2656784, -49.4215508, 45.2991562, -94.7105560, 94.6872253
16: -63.0188065, 43.4322052, -63.0206490, 43.4724731, -106.4912796, 106.4528503
17: -96.9198303, 40.3742218, -96.8984985, 40.4258575, -137.3456879, 137.2727203
18: -64.1454315, 40.0835037, -64.1419983, 40.0921860, -104.2376175, 104.2255020
19: -46.8100204, 27.9781590, -46.8468094, 27.9827499, -74.7927704, 74.8249664
20: -45.1972542, 29.7767220, -45.2546616, 29.7775764, -74.9748230, 75.0313797
21: -58.7941704, 31.8842220, -58.8519058, 31.8904266, -90.6845932, 90.7361298
22: -58.5310555, 34.4441223, -58.5875893, 34.4532013, -92.9842529, 93.0317078
23: -46.5359917, 36.8724747, -46.5608788, 36.8808670, -83.4168549, 83.4333496
24: -56.8097343, 37.7237930, -56.8571434, 37.7301483, -94.5398712, 94.5809326
25: -49.9064293, 38.0645027, -49.9677849, 38.0746613, -87.9810944, 88.0322876
26: -70.7054901, 45.6562233, -70.7339706, 45.6621933, -116.3676758, 116.3901825
27: -57.1952019, 39.1046791, -57.2382698, 39.1116829, -96.3068848, 96.3429413
28: -46.2896652, 38.8017197, -46.3333588, 38.8082657, -85.0979309, 85.1350784
29: -59.9168816, 32.4516830, -59.9537964, 32.4614487, -92.3783264, 92.4054794
30: -57.8853264, 41.3654022, -57.9489021, 41.3758011, -99.2611237, 99.3142929
31: -62.5953751, 38.8385849, -62.6439247, 38.8467407, -101.4421158, 101.4824982
32: -68.6946411, 35.3563728, -68.7218933, 35.3690643, -104.0636902, 104.0782623
33: -87.3841171, 47.6800308, -87.4552765, 47.6900635, -135.0741882, 135.1352844
34: -74.1550446, 31.5085430, -74.1709976, 31.5143948, -105.6694412, 105.6795425
35: -71.8364944, 37.7704201, -71.8619385, 37.7773895, -109.6138763, 109.6323547
36: -75.1947937, 38.7514877, -75.2242126, 38.7603569, -113.9551544, 113.9756927
37: -106.2794876, 30.5494461, -106.3023376, 30.5601864, -136.8396759, 136.8517761
38: -88.0121460, 41.5481186, -88.0547562, 41.5598221, -129.5719604, 129.6028748
39: -98.2059937, 45.1153259, -98.2616119, 45.1098175, -143.3157959, 143.3769379
40: -81.8457565, 30.7391682, -81.8726883, 30.7661343, -112.6118927, 112.6118546
41: -69.2205658, 43.1691933, -69.2254486, 43.1806068, -112.4011688, 112.3946304
42: -50.7885208, 35.0208511, -50.8021545, 35.0272598, -85.8157806, 85.8230057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=411, inp2_unstable=411, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 941

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 721

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0067215, upper bound: 63.9178893
time: 89.16 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0067447, upper bound: 63.9891060
time: 89.06 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -74.3760223, 51.0227280, -74.4756851, 51.1453705, -125.5213928, 125.4984131
1: -35.7287865, 43.1084747, -35.7979431, 43.1789703, -78.9077606, 78.9064178
2: -30.5680542, 44.7965469, -30.6490459, 44.8650131, -75.4330673, 75.4455948
3: -35.0565910, 49.5722160, -35.1389999, 49.6367683, -84.6933594, 84.7112122
4: -40.9225426, 51.7481003, -41.0430069, 51.8144569, -92.7369995, 92.7911072
5: -34.2019157, 47.8105927, -34.2536278, 47.8806915, -82.0825958, 82.0642242
6: -70.7301788, 40.9981880, -70.7957840, 41.0562286, -111.7864075, 111.7939758
7: -43.1136360, 45.8213730, -43.2072639, 45.8875122, -89.0011444, 89.0286407
8: -48.7733612, 62.9116631, -48.9533195, 63.0131989, -111.7865448, 111.8649750
9: -41.9526596, 45.5134773, -42.0221291, 45.5870628, -87.5397186, 87.5356064
10: -63.0312195, 57.4800797, -63.0965462, 57.5564651, -120.5876770, 120.5766296
11: -60.6033783, 33.4322243, -60.6852112, 33.5087128, -94.1120911, 94.1174316
12: -67.9140625, 38.0876923, -67.9573212, 38.1951294, -106.1091843, 106.0450134
13: -65.1741486, 61.2276230, -65.2673340, 61.3097916, -126.4839325, 126.4949570
14: -101.7061081, 45.7718697, -101.7909698, 45.8758926, -147.5820007, 147.5628357
15: -49.4189491, 45.2762337, -49.4873428, 45.3318405, -94.7507935, 94.7635651
16: -63.0487099, 43.4350586, -63.1121559, 43.5313339, -106.5800476, 106.5472107
17: -96.9772186, 40.3801193, -97.0517807, 40.5413284, -137.5185547, 137.4319000
18: -64.1970367, 40.0872955, -64.2703552, 40.2382431, -104.4352722, 104.3576431
19: -46.8325310, 27.9813175, -46.9108047, 28.0305920, -74.8631210, 74.8921127
20: -45.2031174, 29.7781410, -45.3032722, 29.8139820, -75.0170975, 75.0814133
21: -58.8148842, 31.8885612, -58.9179077, 31.9432068, -90.7580719, 90.8064575
22: -58.5591545, 34.4460182, -58.6790848, 34.5188026, -93.0779572, 93.1251068
23: -46.5652618, 36.8770828, -46.6340561, 36.9745483, -83.5398102, 83.5111389
24: -56.8426704, 37.7261848, -56.9529533, 37.8173218, -94.6599884, 94.6791382
25: -49.9272614, 38.0663872, -50.0315208, 38.1238556, -88.0511169, 88.0979080
26: -70.7228088, 45.6637764, -70.7945786, 45.7541161, -116.4769287, 116.4583511
27: -57.2127151, 39.1078300, -57.3111687, 39.1648445, -96.3775558, 96.4189987
28: -46.3072052, 38.8049011, -46.3850632, 38.8693924, -85.1765976, 85.1899643
29: -59.9575691, 32.4530487, -60.0765991, 32.5609894, -92.5185547, 92.5296478
30: -57.9097290, 41.3705139, -58.0264091, 41.4595184, -99.3692398, 99.3969193
31: -62.6318932, 38.8416252, -62.7438698, 38.9400520, -101.5719452, 101.5854950
32: -68.7032471, 35.3656235, -68.7677307, 35.4232292, -104.1264648, 104.1333542
33: -87.3923340, 47.6864624, -87.5041656, 47.7260208, -135.1183472, 135.1906281
34: -74.1905594, 31.5158501, -74.2676926, 31.6167297, -105.8072891, 105.7835388
35: -71.8717804, 37.7772102, -71.9558868, 37.8708878, -109.7426605, 109.7330856
36: -75.2163544, 38.7550354, -75.2849197, 38.8191414, -114.0354919, 114.0399551
37: -106.3173141, 30.5534573, -106.4125214, 30.6490955, -136.9664154, 136.9659729
38: -88.0229950, 41.5558090, -88.1032104, 41.6053314, -129.6283264, 129.6590118
39: -98.2172165, 45.1425552, -98.3432693, 45.1729317, -143.3901367, 143.4858246
40: -81.8566589, 30.7451630, -81.9383698, 30.7976742, -112.6543274, 112.6835327
41: -69.2388382, 43.1745415, -69.2847977, 43.2464180, -112.4852600, 112.4593353
42: -50.7938042, 35.0295334, -50.8387299, 35.0694046, -85.8632050, 85.8682632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=411, inp2_unstable=411, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=566, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 941

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 721

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0067215, upper bound: 63.9178893
time: 100.96 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0067215, upper bound: 63.9178893
time: 411.65 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -74.5435562, 51.2490845, -74.3845444, 51.2302551, -125.7738113, 125.6336288
1: -35.8383713, 43.2429581, -35.7371826, 43.2188263, -79.0571976, 78.9801407
2: -30.6930370, 44.9191589, -30.5780106, 44.8917542, -75.5847931, 75.4971695
3: -35.1401405, 49.6620522, -35.0694313, 49.6307373, -84.7708740, 84.7314835
4: -41.0520973, 51.8412743, -40.9412651, 51.7984886, -92.8505859, 92.7825394
5: -34.2791023, 47.9326172, -34.2115250, 47.8997498, -82.1788330, 82.1441345
6: -70.8160248, 41.0835419, -70.7903290, 41.0280075, -111.8440323, 111.8738708
7: -43.2273941, 45.9122009, -43.1258621, 45.8754501, -89.1028366, 89.0380554
8: -48.9621048, 63.0465546, -48.7952576, 62.9813576, -111.9434662, 111.8418121
9: -42.0676270, 45.6420250, -41.9752502, 45.6168060, -87.6844330, 87.6172714
10: -63.1653748, 57.5989876, -63.0602722, 57.5725365, -120.7379150, 120.6592560
11: -60.7320061, 33.5087433, -60.6765976, 33.4561348, -94.1881409, 94.1853409
12: -67.9506531, 38.1944237, -67.9126968, 38.1276245, -106.0782776, 106.1071167
13: -65.2793884, 61.3356819, -65.2054214, 61.2910233, -126.5704041, 126.5410995
14: -101.8567047, 45.9354095, -101.7320099, 45.9235115, -147.7802124, 147.6674194
15: -49.5105019, 45.3843918, -49.4417114, 45.3573761, -94.8678741, 94.8261032
16: -63.1640663, 43.5473175, -63.0550499, 43.5271454, -106.6912079, 106.6023712
17: -97.0438385, 40.5238419, -96.9319839, 40.5032578, -137.5470886, 137.4558258
18: -64.2504959, 40.1732025, -64.1950302, 40.1064148, -104.3569107, 104.3682327
19: -46.9603958, 28.0626030, -46.9234428, 27.9911118, -74.9515076, 74.9860458
20: -45.3642502, 29.8835182, -45.3396683, 29.7888470, -75.1530991, 75.2231903
21: -58.9967995, 32.0153809, -58.9563293, 31.9019260, -90.8987274, 90.9716949
22: -58.7398071, 34.5839767, -58.6970215, 34.4664536, -93.2062607, 93.2809906
23: -46.6725845, 36.9600105, -46.6284103, 36.8960571, -83.5686340, 83.5884247
24: -57.0038033, 37.8218880, -56.9601021, 37.7398376, -94.7436371, 94.7819824
25: -50.1128311, 38.2303009, -50.0764084, 38.0891495, -88.2019806, 88.3067093
26: -70.8285294, 45.7681961, -70.7896347, 45.6762619, -116.5047913, 116.5578308
27: -57.3507652, 39.1947212, -57.3185158, 39.1231422, -96.4739075, 96.5132370
28: -46.4478989, 38.9117737, -46.4140472, 38.8186951, -85.2665939, 85.3258209
29: -60.0974884, 32.5502090, -60.0486984, 32.4748802, -92.5723648, 92.5989075
30: -58.1083984, 41.5304260, -58.0654564, 41.3937531, -99.5021362, 99.5958786
31: -62.8057899, 38.9691200, -62.7540054, 38.8592987, -101.6650848, 101.7231140
32: -68.7919540, 35.4359322, -68.7679443, 35.3932419, -104.1851883, 104.2038727
33: -87.6094971, 47.8181190, -87.5610352, 47.7106285, -135.3201294, 135.3791351
34: -74.2875366, 31.6331882, -74.2361298, 31.5290089, -105.8165436, 105.8693161
35: -71.9933243, 37.9048691, -71.9386520, 37.7929001, -109.7862244, 109.8435211
36: -75.3173370, 38.8753395, -75.2883453, 38.7751312, -114.0924683, 114.1636810
37: -106.4453201, 30.6037464, -106.3768158, 30.5760670, -137.0213928, 136.9805603
38: -88.1493073, 41.6770630, -88.1244125, 41.5867691, -129.7360840, 129.8014832
39: -98.4024048, 45.1856537, -98.3487854, 45.1316872, -143.5340881, 143.5344238
40: -81.9590454, 30.8463383, -81.9202881, 30.8122940, -112.7713318, 112.7666245
41: -69.2836151, 43.2125969, -69.2531815, 43.1805077, -112.4641113, 112.4657745
42: -50.8402328, 35.0847397, -50.8250275, 35.0428658, -85.8831024, 85.9097672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=411, inp2_unstable=411, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 941

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 721

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0072418, upper bound: 63.9799332
time: 102.14 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0072418, upper bound: 64.0510899
time: 94.97 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -74.5594940, 51.2527695, -74.5008774, 51.2699394, -125.8294373, 125.7536316
1: -35.8461571, 43.2578812, -35.8102837, 43.2591896, -79.1053467, 79.0681534
2: -30.6976700, 44.9387589, -30.6590137, 44.9415932, -75.6392670, 75.5977707
3: -35.1439819, 49.6891632, -35.1497612, 49.6992874, -84.8432693, 84.8389130
4: -41.0570564, 51.8792496, -41.0584335, 51.8837852, -92.9408417, 92.9376831
5: -34.2843742, 47.9545364, -34.2639580, 47.9558182, -82.2401886, 82.2184906
6: -70.8318405, 41.0892906, -70.8470459, 41.0776443, -111.9094696, 111.9363403
7: -43.2348289, 45.9472046, -43.2216339, 45.9556389, -89.1904602, 89.1688385
8: -48.9663048, 63.1107559, -48.9697533, 63.1218719, -112.0881805, 112.0805054
9: -42.0719452, 45.6591301, -42.0389938, 45.6651459, -87.7370911, 87.6981201
10: -63.1724663, 57.6070137, -63.1197815, 57.6176682, -120.7901306, 120.7267914
11: -60.7677994, 33.5129433, -60.7667732, 33.5261154, -94.2939148, 94.2797165
12: -67.9815826, 38.2044907, -67.9882126, 38.2282944, -106.2098694, 106.1927032
13: -65.2861786, 61.3612976, -65.2911606, 61.3737297, -126.6598969, 126.6524582
14: -101.8655548, 45.9401894, -101.8177338, 45.9664917, -147.8320465, 147.7579193
15: -49.5180664, 45.3949776, -49.5075188, 45.3901138, -94.9081802, 94.9024963
16: -63.1939392, 43.5501938, -63.1465378, 43.5860176, -106.7799530, 106.6967316
17: -97.1012573, 40.5297012, -97.0852966, 40.6187592, -137.7200165, 137.6149902
18: -64.3021240, 40.1769562, -64.3233490, 40.2525024, -104.5546265, 104.5002975
19: -46.9829025, 28.0657425, -46.9874229, 28.0389404, -75.0218430, 75.0531616
20: -45.3701096, 29.8849697, -45.3882408, 29.8252602, -75.1953735, 75.2732086
21: -59.0175285, 32.0197144, -59.0223045, 31.9546890, -90.9722137, 91.0420074
22: -58.7679253, 34.5858727, -58.7884789, 34.5320435, -93.2999573, 93.3743515
23: -46.7018394, 36.9645767, -46.7015724, 36.9897308, -83.6915588, 83.6661453
24: -57.0367432, 37.8242722, -57.0558853, 37.8270035, -94.8637390, 94.8801498
25: -50.1336479, 38.2322083, -50.1401787, 38.1383324, -88.2719803, 88.3723755
26: -70.8458481, 45.7757874, -70.8502197, 45.7681847, -116.6140289, 116.6260071
27: -57.3682823, 39.1978607, -57.3913879, 39.1762810, -96.5445633, 96.5892487
28: -46.4654083, 38.9149704, -46.4657364, 38.8798370, -85.3452301, 85.3807068
29: -60.1381683, 32.5515823, -60.1714935, 32.5744019, -92.7125549, 92.7230759
30: -58.1328087, 41.5355301, -58.1429558, 41.4774628, -99.6102448, 99.6784821
31: -62.8422966, 38.9721336, -62.8539658, 38.9526100, -101.7949066, 101.8260956
32: -68.8005524, 35.4452248, -68.8137360, 35.4474144, -104.2479630, 104.2589493
33: -87.6177216, 47.8245773, -87.6099930, 47.7465820, -135.3643036, 135.4345703
34: -74.3230820, 31.6405525, -74.3327789, 31.6313438, -105.9544220, 105.9733276
35: -72.0286713, 37.9116707, -72.0325775, 37.8864021, -109.9150696, 109.9442444
36: -75.3389053, 38.8788757, -75.3490143, 38.8339462, -114.1728363, 114.2278900
37: -106.4831924, 30.6078320, -106.4870605, 30.6649990, -137.1481934, 137.0948944
38: -88.1601868, 41.6847229, -88.1728363, 41.6323051, -129.7924957, 129.8575592
39: -98.4136124, 45.2128143, -98.4304199, 45.1948090, -143.6084290, 143.6432343
40: -81.9699707, 30.8523293, -81.9859772, 30.8438568, -112.8138275, 112.8383026
41: -69.3018951, 43.2179413, -69.3125610, 43.2462845, -112.5481796, 112.5305023
42: -50.8454971, 35.0934219, -50.8616180, 35.0850372, -85.9305344, 85.9550400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=411, inp2_unstable=411, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=566, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 1784
type: A, layer: 1, pos: 733
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 665
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 722
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 753
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 702
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 705
type: A, layer: 1, pos: 647
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 1657
type: A, layer: 1, pos: 737
type: A, layer: 1, pos: 1786
type: A, layer: 1, pos: 1718
type: A, layer: 1, pos: 739
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 703
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 689
type: A, layer: 1, pos: 1785
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 984
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 1656
type: A, layer: 1, pos: 1669
type: A, layer: 1, pos: 999
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 1722
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 765
type: A, layer: 1, pos: 654
type: A, layer: 1, pos: 1001
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 593
type: A, layer: 1, pos: 1744
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 1465
type: A, layer: 1, pos: 724
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 1758
type: A, layer: 1, pos: 1000
type: A, layer: 1, pos: 1016
type: A, layer: 1, pos: 1655
type: A, layer: 1, pos: 1712
type: A, layer: 1, pos: 1757
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 670
type: A, layer: 1, pos: 588
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 655
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 1002
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 1589
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 1598
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 983
type: A, layer: 1, pos: 1672
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 1713
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 1728
type: A, layer: 1, pos: 998
type: A, layer: 1, pos: 1745
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 590
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 1702
type: A, layer: 1, pos: 1670
type: A, layer: 1, pos: 1616
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 704
type: A, layer: 1, pos: 1017
type: A, layer: 1, pos: 681
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 591
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 559
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 1729
type: A, layer: 1, pos: 1599
type: A, layer: 1, pos: 658
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 985
type: A, layer: 1, pos: 592
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 1605
type: A, layer: 1, pos: 562
type: A, layer: 1, pos: 563
type: A, layer: 1, pos: 1018
type: A, layer: 1, pos: 606
type: A, layer: 1, pos: 607
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 605
type: A, layer: 1, pos: 1659
type: A, layer: 1, pos: 1777
type: A, layer: 1, pos: 1763
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 1760
type: A, layer: 1, pos: 1742
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 1597
type: A, layer: 1, pos: 622
type: A, layer: 1, pos: 1665
type: A, layer: 1, pos: 1714
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 687
type: A, layer: 1, pos: 1650
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 1600
type: A, layer: 1, pos: 1566
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 1776
type: A, layer: 1, pos: 1620
type: A, layer: 1, pos: 967
type: A, layer: 1, pos: 1654
type: A, layer: 1, pos: 1582
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 1759
type: A, layer: 1, pos: 1663
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 1765
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 1779
type: A, layer: 1, pos: 1590
type: A, layer: 1, pos: 609
type: A, layer: 1, pos: 1695
type: A, layer: 1, pos: 1726
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 1761
type: A, layer: 1, pos: 1747
type: A, layer: 1, pos: 948
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 1584
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 876
type: A, layer: 1, pos: 1730
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 1696
type: A, layer: 1, pos: 1680
type: A, layer: 1, pos: 1617
type: A, layer: 1, pos: 1588
type: A, layer: 1, pos: 949
type: A, layer: 1, pos: 1464
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 623
type: A, layer: 1, pos: 549
type: A, layer: 1, pos: 1435
type: A, layer: 1, pos: 1614
type: A, layer: 1, pos: 1648
type: A, layer: 1, pos: 1632
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 1565
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 1727
type: A, layer: 1, pos: 854
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 1774
type: A, layer: 1, pos: 1778
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 1613
type: A, layer: 1, pos: 1619
type: A, layer: 1, pos: 1573
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 1701
type: A, layer: 1, pos: 966
type: A, layer: 1, pos: 1649
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 1401
type: A, layer: 1, pos: 608
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 1586
type: A, layer: 1, pos: 1666
type: A, layer: 1, pos: 1773
type: A, layer: 1, pos: 1581
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 1601
type: A, layer: 1, pos: 1705
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 1615
type: A, layer: 1, pos: 673
type: A, layer: 1, pos: 1635
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 1567
type: A, layer: 1, pos: 1604
type: A, layer: 1, pos: 1602
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 1697
type: A, layer: 1, pos: 1585
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 595
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 532
type: A, layer: 1, pos: 570
type: A, layer: 1, pos: 1743
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 1541
type: A, layer: 1, pos: 624
type: A, layer: 1, pos: 1679
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 1003
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 1583
type: A, layer: 1, pos: 1418
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 1790
type: A, layer: 1, pos: 1574
type: A, layer: 1, pos: 1014
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 1629
type: A, layer: 1, pos: 1466
type: A, layer: 1, pos: 1638
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 1707
type: A, layer: 1, pos: 1618
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 1791
type: A, layer: 1, pos: 539
type: A, layer: 1, pos: 1641
type: A, layer: 1, pos: 956
type: A, layer: 1, pos: 1537
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 1591
type: A, layer: 1, pos: 1686
type: A, layer: 1, pos: 1612
type: A, layer: 1, pos: 1403
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 1545
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 1419
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 1603
type: A, layer: 1, pos: 1775
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 1647
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 970
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 965
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 986
type: A, layer: 1, pos: 1564
type: A, layer: 1, pos: 857
type: A, layer: 1, pos: 1642
type: A, layer: 1, pos: 1552
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 1012
type: A, layer: 1, pos: 1558
type: A, layer: 1, pos: 1572
type: A, layer: 1, pos: 674
type: A, layer: 1, pos: 1020
type: A, layer: 1, pos: 1546
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 1467
type: A, layer: 1, pos: 672
type: A, layer: 1, pos: 1536
type: A, layer: 1, pos: 1538
type: A, layer: 1, pos: 1664
type: A, layer: 1, pos: 546
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 1385
type: A, layer: 1, pos: 1557
type: A, layer: 1, pos: 1569
type: A, layer: 1, pos: 1570
type: A, layer: 1, pos: 1542
type: A, layer: 1, pos: 565
type: A, layer: 1, pos: 1553
type: A, layer: 1, pos: 1547
type: A, layer: 1, pos: 1563
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 971
type: A, layer: 1, pos: 872
type: A, layer: 1, pos: 564
type: A, layer: 1, pos: 1548
type: A, layer: 1, pos: 1764
type: A, layer: 1, pos: 545
type: A, layer: 1, pos: 1402
type: A, layer: 1, pos: 1631
type: A, layer: 1, pos: 1606
type: A, layer: 1, pos: 997
type: A, layer: 1, pos: 1549
type: A, layer: 1, pos: 1780
type: A, layer: 1, pos: 1595
type: A, layer: 1, pos: 1746
type: A, layer: 1, pos: 1646
type: A, layer: 1, pos: 1013
type: A, layer: 1, pos: 518
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 1596
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 1561
type: A, layer: 1, pos: 1543
type: A, layer: 1, pos: 1633
type: A, layer: 1, pos: 982
type: A, layer: 1, pos: 1575
type: A, layer: 1, pos: 1592
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 1559
type: A, layer: 1, pos: 1634
type: A, layer: 1, pos: 554
type: A, layer: 1, pos: 940
type: A, layer: 1, pos: 582
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 1594
type: A, layer: 1, pos: 516
type: A, layer: 1, pos: 1451
type: A, layer: 1, pos: 1562
type: A, layer: 1, pos: 871
type: A, layer: 1, pos: 1555
type: A, layer: 1, pos: 1433
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 1550
type: A, layer: 1, pos: 1544
type: A, layer: 1, pos: 950
type: A, layer: 1, pos: 1540
type: A, layer: 1, pos: 1556
type: A, layer: 1, pos: 1682
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 1630
type: A, layer: 1, pos: 1587
type: A, layer: 1, pos: 1577
type: A, layer: 1, pos: 1571
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 1578
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 1436
type: A, layer: 1, pos: 515
type: A, layer: 1, pos: 1551
type: A, layer: 1, pos: 1568
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 611
type: A, layer: 1, pos: 1640
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 610
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 1580
type: A, layer: 1, pos: 1711
type: A, layer: 1, pos: 1539
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 1560
type: A, layer: 1, pos: 1685
type: A, layer: 1, pos: 1611
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 1698
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 1384
type: A, layer: 1, pos: 1678
type: A, layer: 1, pos: 1400
type: A, layer: 1, pos: 1554
type: A, layer: 1, pos: 530
type: A, layer: 1, pos: 640
type: A, layer: 1, pos: 1710
type: A, layer: 1, pos: 964
type: A, layer: 1, pos: 544
type: A, layer: 1, pos: 1015
type: A, layer: 1, pos: 1681
type: A, layer: 1, pos: 1593
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 1762
type: A, layer: 1, pos: 1579
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 1639
type: A, layer: 1, pos: 1576
type: A, layer: 1, pos: 1662
type: A, layer: 1, pos: 664
type: A, layer: 1, pos: 1628
type: A, layer: 1, pos: 941

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 721

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0072418, upper bound: 63.9799332
time: 94.33 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0072418, upper bound: 64.0510899
time: 84.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 180.81 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 180.81
Output dim: 8, lower bound: -64.0067215, upper bound: 63.9178893
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 180.81
Output dim: 8, lower bound: -64.0067447, upper bound: 63.9891060
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 180.81
Output dim: 8, lower bound: -64.0067215, upper bound: 63.9178893
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 180.81
Output dim: 8, lower bound: -64.0067215, upper bound: 63.9178893
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 180.81
Output dim: 8, lower bound: -64.0072418, upper bound: 63.9799332
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 180.81
Output dim: 8, lower bound: -64.0072418, upper bound: 64.0510899
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 180.81
Output dim: 8, lower bound: -64.0072418, upper bound: 63.9799332
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 180.81
Output dim: 8, lower bound: -64.0072418, upper bound: 64.0510899

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -74.1982498, 50.8448143, -74.3251038, 51.0051346, -125.2033844, 125.1699219
1: -35.6259880, 42.9625893, -35.7120514, 43.0626831, -78.6886597, 78.6746368
2: -30.4781494, 44.6638184, -30.5548401, 44.7527046, -75.2308502, 75.2186584
3: -34.9926682, 49.4517288, -35.0473404, 49.5161476, -84.5088196, 84.4990692
4: -40.8106079, 51.5841446, -40.9053879, 51.6565781, -92.4671860, 92.4895325
5: -34.1307106, 47.6878281, -34.1875763, 47.7680244, -81.8987274, 81.8754044
6: -70.4844742, 40.8430481, -70.6063843, 40.9816856, -111.4661560, 111.4494247
7: -43.0057907, 45.6772232, -43.0959778, 45.7456970, -88.7514801, 88.7731934
8: -48.5994568, 62.5627289, -48.7623291, 62.7119675, -111.3114243, 111.3250580
9: -41.8355865, 45.3408546, -41.9401932, 45.4494591, -87.2850342, 87.2810516
10: -62.8619881, 57.3157234, -63.0065651, 57.4259109, -120.2879028, 120.3222885
11: -60.4608612, 33.3582458, -60.5459595, 33.4172134, -93.8780670, 93.9042053
12: -67.7693176, 37.9721909, -67.8212662, 38.0720482, -105.8413696, 105.7934570
13: -65.0576630, 61.1149139, -65.1399994, 61.1902390, -126.2479019, 126.2549133
14: -101.4956436, 45.4982071, -101.6699448, 45.6784439, -147.1740875, 147.1681519
15: -49.2728195, 45.0768204, -49.3940163, 45.1911201, -94.4639282, 94.4708328
16: -62.8932343, 43.3384666, -62.9918900, 43.4256744, -106.3189087, 106.3303528
17: -96.7548141, 40.2290764, -96.8666229, 40.3452911, -137.1000977, 137.0956879
18: -64.0343781, 40.0038872, -64.0896378, 40.0747299, -104.1091080, 104.0935211
19: -46.7087517, 27.9281006, -46.7958145, 27.9719219, -74.6806717, 74.7239075
20: -45.0929260, 29.7211170, -45.2002640, 29.7619362, -74.8548584, 74.9213791
21: -58.6424980, 31.8128929, -58.7754364, 31.8725815, -90.5150757, 90.5883331
22: -58.4027252, 34.3839569, -58.5241776, 34.4381027, -92.8408203, 92.9081268
23: -46.4705353, 36.7975464, -46.5294991, 36.8506241, -83.3211517, 83.3270416
24: -56.6874084, 37.6728058, -56.7885818, 37.7193985, -94.4068069, 94.4613800
25: -49.8130951, 37.9900703, -49.9214516, 38.0513344, -87.8644257, 87.9115219
26: -70.6306915, 45.5730820, -70.7048645, 45.6329231, -116.2636108, 116.2779388
27: -57.0960159, 39.0391006, -57.1886787, 39.0918999, -96.1879120, 96.2277756
28: -46.2225990, 38.7343102, -46.3000717, 38.7891693, -85.0117645, 85.0343781
29: -59.8050804, 32.3927727, -59.8988876, 32.4417419, -92.2468262, 92.2916565
30: -57.7550240, 41.2764244, -57.8804741, 41.3508301, -99.1058502, 99.1568985
31: -62.4083099, 38.7529259, -62.5425110, 38.8316422, -101.2399445, 101.2954407
32: -68.5031738, 35.2576675, -68.6138000, 35.3529282, -103.8560944, 103.8714676
33: -87.1164246, 47.5338631, -87.3093262, 47.6742477, -134.7906799, 134.8431854
34: -73.9513550, 31.3780518, -74.0599976, 31.4987373, -105.4500885, 105.4380493
35: -71.6161346, 37.6157761, -71.7388611, 37.7626534, -109.3787842, 109.3546295
36: -75.0027008, 38.6169701, -75.1140747, 38.7473412, -113.7500458, 113.7310333
37: -106.1073227, 30.4848900, -106.2141571, 30.5468636, -136.6541748, 136.6990509
38: -87.7734985, 41.3822403, -87.9215927, 41.5408630, -129.3143616, 129.3038330
39: -97.9582138, 44.9900589, -98.1269836, 45.0972137, -143.0554199, 143.1170349
40: -81.6931610, 30.6629944, -81.7946625, 30.7510147, -112.4441681, 112.4576569
41: -69.0646667, 43.0783157, -69.1407852, 43.1642075, -112.2288742, 112.2191010
42: -50.7327499, 34.9488754, -50.7771530, 35.0027390, -85.7354813, 85.7260284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=410, inp2_unstable=411, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 941

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 1784

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -63.9643699, upper bound: 63.9143999
time: 86.19 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -63.9643699, upper bound: 63.9143999
time: 456.32 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -74.3541718, 51.0040283, -74.3558884, 51.0948792, -125.4490509, 125.3599014
1: -35.7176514, 43.0865059, -35.7228661, 43.1342392, -78.8518906, 78.8093719
2: -30.5605698, 44.7710571, -30.5663471, 44.8115921, -75.3721619, 75.3374023
3: -35.0499039, 49.5388565, -35.0569382, 49.5644302, -84.6143341, 84.5957947
4: -40.9133301, 51.7035484, -40.9232597, 51.7251663, -92.6384888, 92.6268082
5: -34.1934967, 47.7823944, -34.1992989, 47.8208466, -82.0143433, 81.9816895
6: -70.7031784, 40.9858780, -70.7322083, 41.0027504, -111.7059326, 111.7180862
7: -43.1021118, 45.7804375, -43.1090393, 45.8036880, -88.9057846, 88.8894806
8: -48.7643433, 62.8351517, -48.7758980, 62.8652191, -111.6295624, 111.6110458
9: -41.9427567, 45.4892883, -41.9550323, 45.5344734, -87.4772186, 87.4443130
10: -63.0163155, 57.4645424, -63.0323639, 57.5068283, -120.5231476, 120.4968948
11: -60.5594635, 33.4226456, -60.5901718, 33.4354439, -93.9949036, 94.0128174
12: -67.8757706, 38.0739059, -67.8773727, 38.0922699, -105.9680328, 105.9512634
13: -65.1531525, 61.1963081, -65.1730499, 61.2236710, -126.3768234, 126.3693542
14: -101.6877441, 45.7556877, -101.6995850, 45.8259964, -147.5137329, 147.4552612
15: -49.4041481, 45.2564545, -49.4171143, 45.2935638, -94.6977081, 94.6735687
16: -63.0112038, 43.4191208, -63.0160980, 43.4642639, -106.4754639, 106.4352188
17: -96.9108124, 40.3666077, -96.8931351, 40.4211960, -137.3320007, 137.2597351
18: -64.1396332, 40.0785217, -64.1384735, 40.0891647, -104.2287979, 104.2169876
19: -46.8014374, 27.9755135, -46.8416023, 27.9811707, -74.7826080, 74.8171158
20: -45.1875229, 29.7733765, -45.2486191, 29.7756004, -74.9631195, 75.0219955
21: -58.7841682, 31.8801937, -58.8457909, 31.8879929, -90.6721649, 90.7259827
22: -58.5235634, 34.4405975, -58.5830307, 34.4511070, -92.9746704, 93.0236206
23: -46.5312004, 36.8602982, -46.5579567, 36.8734665, -83.4046631, 83.4182587
24: -56.8027229, 37.7191238, -56.8529015, 37.7273521, -94.5300751, 94.5720215
25: -49.8963547, 38.0596924, -49.9615402, 38.0717697, -87.9681244, 88.0212326
26: -70.6983032, 45.6334152, -70.7295837, 45.6485977, -116.3468933, 116.3629761
27: -57.1886292, 39.0956306, -57.2342720, 39.1062164, -96.2948456, 96.3299026
28: -46.2815018, 38.7977753, -46.3284225, 38.8059044, -85.0874023, 85.1261902
29: -59.9094200, 32.4477692, -59.9491501, 32.4590988, -92.3685150, 92.3969193
30: -57.8755531, 41.3602753, -57.9429550, 41.3726959, -99.2482452, 99.3032227
31: -62.5857964, 38.8333435, -62.6380920, 38.8435898, -101.4293594, 101.4714279
32: -68.6849518, 35.3520546, -68.7159576, 35.3664932, -104.0514145, 104.0680084
33: -87.3718567, 47.6736832, -87.4478455, 47.6862793, -135.0581360, 135.1215210
34: -74.1458893, 31.5023956, -74.1654434, 31.5106926, -105.6565857, 105.6678391
35: -71.8264084, 37.7644577, -71.8558502, 37.7738266, -109.6002350, 109.6203079
36: -75.1854935, 38.7463799, -75.2185516, 38.7573166, -113.9428101, 113.9649200
37: -106.2696152, 30.5443420, -106.2963104, 30.5570850, -136.8266907, 136.8406372
38: -88.0015411, 41.5411911, -88.0482941, 41.5557251, -129.5572510, 129.5894775
39: -98.1934662, 45.1100159, -98.2539597, 45.1066742, -143.3001404, 143.3639679
40: -81.8369598, 30.7343025, -81.8673019, 30.7632389, -112.6001968, 112.6016006
41: -69.2127380, 43.1648216, -69.2206573, 43.1780243, -112.3907471, 112.3854752
42: -50.7841492, 35.0148430, -50.7994919, 35.0235672, -85.8077164, 85.8143311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=410, inp2_unstable=411, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 941

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1784

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -63.9643890, upper bound: 63.9855952
time: 92.81 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -63.9643890, upper bound: 63.9855952
time: 128.08 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -74.2141724, 50.8485069, -74.4412613, 51.0448265, -125.2589722, 125.2897568
1: -35.6337891, 42.9775085, -35.7851524, 43.1030617, -78.7368469, 78.7626572
2: -30.4827709, 44.6834145, -30.6358185, 44.8025475, -75.2853088, 75.3192215
3: -34.9965019, 49.4788399, -35.1276932, 49.5847206, -84.5812225, 84.6065369
4: -40.8155899, 51.6221313, -41.0225601, 51.7418823, -92.5574722, 92.6446838
5: -34.1359673, 47.7097397, -34.2400017, 47.8241043, -81.9600677, 81.9497375
6: -70.5002594, 40.8487701, -70.6631165, 41.0312843, -111.5315399, 111.5118866
7: -43.0132141, 45.7122650, -43.1917648, 45.8259125, -88.8391190, 88.9040298
8: -48.6036415, 62.6269302, -48.9368896, 62.8524857, -111.4561234, 111.5638123
9: -41.8399239, 45.3579750, -42.0039673, 45.4977951, -87.3377228, 87.3619385
10: -62.8691139, 57.3237381, -63.0660744, 57.4710846, -120.3401947, 120.3898163
11: -60.4966812, 33.3624229, -60.6361389, 33.4872169, -93.9839020, 93.9985657
12: -67.8001785, 37.9822807, -67.8967667, 38.1726990, -105.9728775, 105.8790436
13: -65.0644531, 61.1405373, -65.2257538, 61.2729530, -126.3374023, 126.3662872
14: -101.5045013, 45.5029755, -101.7555771, 45.7214050, -147.2259064, 147.2585449
15: -49.2803802, 45.0874062, -49.4598160, 45.2238693, -94.5042496, 94.5472183
16: -62.9231300, 43.3413315, -63.0833511, 43.4845123, -106.4076385, 106.4246826
17: -96.8122330, 40.2349472, -97.0199127, 40.4607544, -137.2729797, 137.2548523
18: -64.0860291, 40.0076675, -64.2179947, 40.2207947, -104.3068237, 104.2256622
19: -46.7312393, 27.9312305, -46.8597946, 28.0197449, -74.7509766, 74.7910156
20: -45.0987892, 29.7225533, -45.2488556, 29.7983475, -74.8971405, 74.9714050
21: -58.6632080, 31.8171940, -58.8414307, 31.9253788, -90.5885696, 90.6586151
22: -58.4308662, 34.3858414, -58.6156921, 34.5036774, -92.9345398, 93.0015335
23: -46.4997787, 36.8021011, -46.6026878, 36.9443169, -83.4440918, 83.4047852
24: -56.7203407, 37.6752052, -56.8843918, 37.8065643, -94.5269012, 94.5595932
25: -49.8339386, 37.9919853, -49.9852219, 38.1005287, -87.9344635, 87.9772034
26: -70.6480560, 45.5806389, -70.7654572, 45.7248306, -116.3728867, 116.3460922
27: -57.1135521, 39.0422440, -57.2615776, 39.1450577, -96.2586060, 96.3038177
28: -46.2401276, 38.7374725, -46.3518066, 38.8503265, -85.0904541, 85.0892792
29: -59.8457947, 32.3941612, -60.0217476, 32.5412445, -92.3870239, 92.4159012
30: -57.7794228, 41.2815552, -57.9579887, 41.4345322, -99.2139587, 99.2395325
31: -62.4448509, 38.7559776, -62.6424751, 38.9249496, -101.3697968, 101.3984528
32: -68.5117722, 35.2669182, -68.6596298, 35.4071198, -103.9188919, 103.9265442
33: -87.1246185, 47.5403252, -87.3582764, 47.7102013, -134.8348236, 134.8985901
34: -73.9869156, 31.3853760, -74.1566849, 31.6010990, -105.5879822, 105.5420532
35: -71.6514740, 37.6225586, -71.8327942, 37.8561363, -109.5076141, 109.4553528
36: -75.0242920, 38.6205673, -75.1747742, 38.8061142, -113.8303986, 113.7953339
37: -106.1451950, 30.4889507, -106.3243866, 30.6357479, -136.7809296, 136.8133392
38: -87.7843857, 41.3899307, -87.9700165, 41.5862923, -129.3706818, 129.3599548
39: -97.9694138, 45.0172577, -98.2085953, 45.1603394, -143.1297302, 143.2258453
40: -81.7040558, 30.6690083, -81.8603287, 30.7825470, -112.4866028, 112.5293350
41: -69.0829620, 43.0837135, -69.2001801, 43.2299919, -112.3129578, 112.2838898
42: -50.7380066, 34.9575577, -50.8137054, 35.0448837, -85.7828751, 85.7712631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=410, inp2_unstable=411, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=566, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 941

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1784

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -63.9643699, upper bound: 63.9143999
time: 81.87 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -63.9643699, upper bound: 63.9143999
time: 96.65 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -74.3701019, 51.0077324, -74.4721451, 51.1345406, -125.5046387, 125.4798737
1: -35.7254639, 43.1014481, -35.7959671, 43.1746330, -78.9000854, 78.8974075
2: -30.5651951, 44.7906494, -30.6473446, 44.8614502, -75.4266357, 75.4379959
3: -35.0537186, 49.5659828, -35.1372833, 49.6329727, -84.6866837, 84.7032623
4: -40.9182968, 51.7415085, -41.0404587, 51.8104706, -92.7287598, 92.7819672
5: -34.1987381, 47.8043175, -34.2517586, 47.8768768, -82.0756149, 82.0560760
6: -70.7189789, 40.9916306, -70.7889404, 41.0523262, -111.7713013, 111.7805557
7: -43.1095505, 45.8154373, -43.2048454, 45.8838806, -88.9934311, 89.0202789
8: -48.7685089, 62.8993492, -48.9504433, 63.0057106, -111.7742157, 111.8497925
9: -41.9470940, 45.5064316, -42.0187912, 45.5828094, -87.5299072, 87.5252151
10: -63.0234528, 57.4725533, -63.0918388, 57.5519485, -120.5753937, 120.5643921
11: -60.5952301, 33.4268570, -60.6803360, 33.5054550, -94.1006851, 94.1071930
12: -67.9066544, 38.0840340, -67.9528427, 38.1929512, -106.0996094, 106.0368805
13: -65.1599579, 61.2219620, -65.2588043, 61.3063507, -126.4663086, 126.4807510
14: -101.6965561, 45.7605286, -101.7852478, 45.8689651, -147.5655212, 147.5457764
15: -49.4117241, 45.2670441, -49.4829445, 45.3262634, -94.7379761, 94.7499771
16: -63.0410538, 43.4219894, -63.1075668, 43.5231247, -106.5641785, 106.5295563
17: -96.9682083, 40.3724747, -97.0464478, 40.5366745, -137.5048828, 137.4189148
18: -64.1912537, 40.0822906, -64.2668152, 40.2352219, -104.4264755, 104.3490982
19: -46.8239441, 27.9786739, -46.9055710, 28.0289955, -74.8529282, 74.8842468
20: -45.1933823, 29.7748222, -45.2972221, 29.8119888, -75.0053711, 75.0720444
21: -58.8048973, 31.8845043, -58.9117966, 31.9407425, -90.7456360, 90.7963028
22: -58.5516930, 34.4425011, -58.6744995, 34.5166931, -93.0683746, 93.1169968
23: -46.5604744, 36.8648682, -46.6311493, 36.9671555, -83.5276337, 83.4960022
24: -56.8356743, 37.7215385, -56.9486732, 37.8145332, -94.6501923, 94.6702118
25: -49.9172134, 38.0615768, -50.0252838, 38.1209564, -88.0381699, 88.0868607
26: -70.7156525, 45.6409645, -70.7901764, 45.7405243, -116.4561691, 116.4311295
27: -57.2061882, 39.0987701, -57.3071594, 39.1593933, -96.3655853, 96.4059219
28: -46.2990341, 38.8009796, -46.3801346, 38.8670387, -85.1660767, 85.1811142
29: -59.9500923, 32.4491272, -60.0719681, 32.5586243, -92.5087128, 92.5210953
30: -57.8999672, 41.3653870, -58.0204811, 41.4564056, -99.3563690, 99.3858643
31: -62.6223373, 38.8364143, -62.7380524, 38.9368973, -101.5592346, 101.5744629
32: -68.6935425, 35.3613205, -68.7617645, 35.4206543, -104.1141968, 104.1230698
33: -87.3801270, 47.6801453, -87.4967651, 47.7222061, -135.1023254, 135.1769104
34: -74.1814270, 31.5097198, -74.2621155, 31.6130505, -105.7944794, 105.7718353
35: -71.8617477, 37.7712631, -71.9498138, 37.8673439, -109.7290878, 109.7210693
36: -75.2070770, 38.7499123, -75.2792206, 38.8160973, -114.0231781, 114.0291290
37: -106.3075333, 30.5483837, -106.4065323, 30.6460400, -136.9535675, 136.9549103
38: -88.0124283, 41.5488205, -88.0967407, 41.6012268, -129.6136475, 129.6455536
39: -98.2046738, 45.1372299, -98.3356171, 45.1697845, -143.3744507, 143.4728394
40: -81.8478851, 30.7403145, -81.9329834, 30.7947598, -112.6426468, 112.6732941
41: -69.2310028, 43.1701851, -69.2799988, 43.2437897, -112.4747925, 112.4501801
42: -50.7894020, 35.0235062, -50.8360443, 35.0657387, -85.8551331, 85.8595428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=410, inp2_unstable=411, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=566, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 941

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1784

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -63.9643890, upper bound: 63.9855952
time: 97.79 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -63.9643890, upper bound: 63.9855952
time: 324.99 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -74.3818970, 51.0748215, -74.3502045, 51.1295815, -125.5114746, 125.4250259
1: -35.7434120, 43.1120491, -35.7243805, 43.1429024, -78.8863144, 78.8364258
2: -30.6078491, 44.8060875, -30.5648041, 44.8292274, -75.4370728, 75.3708878
3: -35.0800629, 49.5685654, -35.0581436, 49.5786018, -84.6586609, 84.6267014
4: -40.9453278, 51.7152863, -40.9208946, 51.7258606, -92.6711884, 92.6361847
5: -34.2131500, 47.8316116, -34.1979523, 47.8430367, -82.0561829, 82.0295563
6: -70.5859909, 40.9343681, -70.6575851, 41.0034790, -111.5894699, 111.5919495
7: -43.1270142, 45.8030014, -43.1103745, 45.8137474, -88.9407501, 88.9133682
8: -48.7925682, 62.7618980, -48.7788010, 62.8205643, -111.6131287, 111.5406952
9: -41.9550781, 45.4866333, -41.9571152, 45.5274506, -87.4825287, 87.4437485
10: -63.0036469, 57.4426270, -63.0299072, 57.4869690, -120.4906082, 120.4725342
11: -60.6252365, 33.4391556, -60.6274567, 33.4346619, -94.0598907, 94.0666122
12: -67.8367157, 38.0888901, -67.8520889, 38.1053162, -105.9420319, 105.9409790
13: -65.1697311, 61.2486343, -65.1638107, 61.2542267, -126.4239502, 126.4124451
14: -101.6551819, 45.6664848, -101.6966400, 45.7690887, -147.4242706, 147.3631287
15: -49.3721275, 45.1956940, -49.4142456, 45.2492943, -94.6214142, 94.6099396
16: -63.0386734, 43.4535446, -63.0263329, 43.4802475, -106.5189209, 106.4798737
17: -96.8787689, 40.3785706, -96.9000397, 40.4225540, -137.3013306, 137.2785950
18: -64.1394348, 40.0936203, -64.1425476, 40.0889969, -104.2284317, 104.2361679
19: -46.8591614, 28.0125713, -46.8723907, 27.9802647, -74.8394241, 74.8849640
20: -45.2599220, 29.8280125, -45.2852173, 29.7731323, -75.0330505, 75.1132278
21: -58.8452644, 31.9441319, -58.8797989, 31.8839550, -90.7292175, 90.8239288
22: -58.6114807, 34.5239372, -58.6335182, 34.4513359, -93.0628128, 93.1574554
23: -46.6070442, 36.8851280, -46.5970306, 36.8658218, -83.4728546, 83.4821548
24: -56.8814888, 37.7709846, -56.8914413, 37.7291412, -94.6106262, 94.6624146
25: -50.0195427, 38.1559601, -50.0300293, 38.0657883, -88.0853271, 88.1859818
26: -70.7536316, 45.6850967, -70.7605362, 45.6468735, -116.4004974, 116.4456253
27: -57.2514572, 39.1290741, -57.2688751, 39.1033516, -96.3548050, 96.3979492
28: -46.3808250, 38.8445053, -46.3807144, 38.7996750, -85.1804962, 85.2252197
29: -59.9856682, 32.4914627, -59.9937248, 32.4552383, -92.4409027, 92.4851837
30: -57.9781723, 41.4416656, -57.9969864, 41.3688278, -99.3470001, 99.4386520
31: -62.6189613, 38.8835297, -62.6525269, 38.8442612, -101.4632263, 101.5360565
32: -68.6004333, 35.3369865, -68.6597900, 35.3771400, -103.9775696, 103.9967728
33: -87.3419724, 47.6721535, -87.4149933, 47.6948662, -135.0368195, 135.0871277
34: -74.0839920, 31.5028515, -74.1250076, 31.5133591, -105.5973358, 105.6278610
35: -71.7731934, 37.7504692, -71.8154373, 37.7782211, -109.5514145, 109.5659027
36: -75.1252365, 38.7411423, -75.1781158, 38.7621918, -113.8874283, 113.9192505
37: -106.2730331, 30.5392895, -106.2885056, 30.5628948, -136.8359070, 136.8277893
38: -87.9107361, 41.5115280, -87.9911118, 41.5678253, -129.4785614, 129.5026398
39: -98.1548386, 45.0603485, -98.2138977, 45.1191025, -143.2739410, 143.2742462
40: -81.8063507, 30.7699928, -81.8422546, 30.7969933, -112.6033478, 112.6122360
41: -69.1276169, 43.1214752, -69.1685333, 43.1642456, -112.2918625, 112.2900085
42: -50.7844315, 35.0127182, -50.8000717, 35.0183182, -85.8027496, 85.8127899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=410, inp2_unstable=411, delta_unstable=2048
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=564, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1784
type: B, layer: 1, pos: 733
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 665
type: B, layer: 1, pos: 722
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 753
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 702
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 647
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 1657
type: B, layer: 1, pos: 737
type: B, layer: 1, pos: 1786
type: B, layer: 1, pos: 1718
type: B, layer: 1, pos: 739
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 703
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 689
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 984
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 1656
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 1669
type: B, layer: 1, pos: 999
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 1722
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 765
type: B, layer: 1, pos: 654
type: B, layer: 1, pos: 1001
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 593
type: B, layer: 1, pos: 1744
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 1465
type: B, layer: 1, pos: 724
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 1758
type: B, layer: 1, pos: 1000
type: B, layer: 1, pos: 1016
type: B, layer: 1, pos: 1655
type: B, layer: 1, pos: 1712
type: B, layer: 1, pos: 1757
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 670
type: B, layer: 1, pos: 588
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 655
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 1002
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 1589
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 1598
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 983
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 1672
type: B, layer: 1, pos: 1713
type: B, layer: 1, pos: 1728
type: B, layer: 1, pos: 1745
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 998
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 590
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 1702
type: B, layer: 1, pos: 1670
type: B, layer: 1, pos: 1616
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 704
type: B, layer: 1, pos: 1017
type: B, layer: 1, pos: 681
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 591
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 559
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 1729
type: B, layer: 1, pos: 658
type: B, layer: 1, pos: 1599
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 985
type: B, layer: 1, pos: 592
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 1605
type: B, layer: 1, pos: 562
type: B, layer: 1, pos: 563
type: B, layer: 1, pos: 1018
type: B, layer: 1, pos: 606
type: B, layer: 1, pos: 607
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 605
type: B, layer: 1, pos: 1659
type: B, layer: 1, pos: 1777
type: B, layer: 1, pos: 1763
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 1760
type: B, layer: 1, pos: 1742
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 1597
type: B, layer: 1, pos: 622
type: B, layer: 1, pos: 1665
type: B, layer: 1, pos: 1714
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 687
type: B, layer: 1, pos: 1650
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 1600
type: B, layer: 1, pos: 1566
type: B, layer: 1, pos: 1776
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 1620
type: B, layer: 1, pos: 967
type: B, layer: 1, pos: 1654
type: B, layer: 1, pos: 1582
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 1759
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 1663
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 1765
type: B, layer: 1, pos: 1779
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 1590
type: B, layer: 1, pos: 609
type: B, layer: 1, pos: 1695
type: B, layer: 1, pos: 1726
type: B, layer: 1, pos: 948
type: B, layer: 1, pos: 1761
type: B, layer: 1, pos: 1747
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 1584
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 1730
type: B, layer: 1, pos: 876
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 1696
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 1680
type: B, layer: 1, pos: 1617
type: B, layer: 1, pos: 1588
type: B, layer: 1, pos: 1464
type: B, layer: 1, pos: 949
type: B, layer: 1, pos: 623
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 549
type: B, layer: 1, pos: 1435
type: B, layer: 1, pos: 1614
type: B, layer: 1, pos: 1648
type: B, layer: 1, pos: 1632
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 1565
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 1727
type: B, layer: 1, pos: 854
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 1774
type: B, layer: 1, pos: 1778
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 1613
type: B, layer: 1, pos: 1573
type: B, layer: 1, pos: 1619
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 1701
type: B, layer: 1, pos: 966
type: B, layer: 1, pos: 1649
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 1401
type: B, layer: 1, pos: 608
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 1586
type: B, layer: 1, pos: 1666
type: B, layer: 1, pos: 1773
type: B, layer: 1, pos: 1581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 1601
type: B, layer: 1, pos: 1705
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 1615
type: B, layer: 1, pos: 673
type: B, layer: 1, pos: 1635
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 1567
type: B, layer: 1, pos: 1604
type: B, layer: 1, pos: 1602
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 1585
type: B, layer: 1, pos: 1697
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 595
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 570
type: B, layer: 1, pos: 532
type: B, layer: 1, pos: 1743
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 1541
type: B, layer: 1, pos: 624
type: B, layer: 1, pos: 1679
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 1003
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 1583
type: B, layer: 1, pos: 1418
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 1790
type: B, layer: 1, pos: 1574
type: B, layer: 1, pos: 1014
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 1629
type: B, layer: 1, pos: 1466
type: B, layer: 1, pos: 1638
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 1707
type: B, layer: 1, pos: 1618
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 1791
type: B, layer: 1, pos: 539
type: B, layer: 1, pos: 1641
type: B, layer: 1, pos: 1537
type: B, layer: 1, pos: 956
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 1591
type: B, layer: 1, pos: 1612
type: B, layer: 1, pos: 1403
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 1686
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 1545
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 1419
type: B, layer: 1, pos: 1603
type: B, layer: 1, pos: 1775
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 1647
type: B, layer: 1, pos: 970
type: B, layer: 1, pos: 965
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 986
type: B, layer: 1, pos: 1564
type: B, layer: 1, pos: 857
type: B, layer: 1, pos: 1552
type: B, layer: 1, pos: 1642
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 1012
type: B, layer: 1, pos: 1558
type: B, layer: 1, pos: 1572
type: B, layer: 1, pos: 1020
type: B, layer: 1, pos: 674
type: B, layer: 1, pos: 1546
type: B, layer: 1, pos: 1467
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 672
type: B, layer: 1, pos: 1536
type: B, layer: 1, pos: 1538
type: B, layer: 1, pos: 1664
type: B, layer: 1, pos: 1557
type: B, layer: 1, pos: 546
type: B, layer: 1, pos: 1385
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 1569
type: B, layer: 1, pos: 1570
type: B, layer: 1, pos: 1542
type: B, layer: 1, pos: 1553
type: B, layer: 1, pos: 565
type: B, layer: 1, pos: 1547
type: B, layer: 1, pos: 1563
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 971
type: B, layer: 1, pos: 680
type: B, layer: 1, pos: 872
type: B, layer: 1, pos: 564
type: B, layer: 1, pos: 1548
type: B, layer: 1, pos: 1764
type: B, layer: 1, pos: 545
type: B, layer: 1, pos: 1402
type: B, layer: 1, pos: 1631
type: B, layer: 1, pos: 1606
type: B, layer: 1, pos: 997
type: B, layer: 1, pos: 1549
type: B, layer: 1, pos: 1780
type: B, layer: 1, pos: 1595
type: B, layer: 1, pos: 1746
type: B, layer: 1, pos: 1646
type: B, layer: 1, pos: 518
type: B, layer: 1, pos: 1013
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 1596
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 982
type: B, layer: 1, pos: 1633
type: B, layer: 1, pos: 1543
type: B, layer: 1, pos: 1561
type: B, layer: 1, pos: 1575
type: B, layer: 1, pos: 1592
type: B, layer: 1, pos: 1559
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 554
type: B, layer: 1, pos: 1634
type: B, layer: 1, pos: 940
type: B, layer: 1, pos: 582
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 1594
type: B, layer: 1, pos: 516
type: B, layer: 1, pos: 1451
type: B, layer: 1, pos: 871
type: B, layer: 1, pos: 1562
type: B, layer: 1, pos: 1555
type: B, layer: 1, pos: 1433
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 1550
type: B, layer: 1, pos: 1544
type: B, layer: 1, pos: 950
type: B, layer: 1, pos: 1540
type: B, layer: 1, pos: 1556
type: B, layer: 1, pos: 1682
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 1630
type: B, layer: 1, pos: 1587
type: B, layer: 1, pos: 1577
type: B, layer: 1, pos: 1571
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 1578
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 1436
type: B, layer: 1, pos: 515
type: B, layer: 1, pos: 1551
type: B, layer: 1, pos: 1568
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 611
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 1640
type: B, layer: 1, pos: 610
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 1580
type: B, layer: 1, pos: 1711
type: B, layer: 1, pos: 1539
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 1560
type: B, layer: 1, pos: 1685
type: B, layer: 1, pos: 1611
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 1698
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 1384
type: B, layer: 1, pos: 1678
type: B, layer: 1, pos: 1400
type: B, layer: 1, pos: 1554
type: B, layer: 1, pos: 530
type: B, layer: 1, pos: 640
type: B, layer: 1, pos: 1710
type: B, layer: 1, pos: 964
type: B, layer: 1, pos: 544
type: B, layer: 1, pos: 1015
type: B, layer: 1, pos: 1593
type: B, layer: 1, pos: 1681
type: B, layer: 1, pos: 1762
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 1579
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 1639
type: B, layer: 1, pos: 1576
type: B, layer: 1, pos: 1662
type: B, layer: 1, pos: 664
type: B, layer: 1, pos: 1628
type: B, layer: 1, pos: 941

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1784

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -63.9648769, upper bound: 63.9763906
time: 82.49 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -63.9648769, upper bound: 63.9763906
time: 128.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 213.30 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 213.30
Output dim: 8, lower bound: -63.9643699, upper bound: 63.9143999
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 213.30
Output dim: 8, lower bound: -63.9643699, upper bound: 63.9143999
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 213.30
Output dim: 8, lower bound: -63.9643890, upper bound: 63.9855952
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 213.30
Output dim: 8, lower bound: -63.9643890, upper bound: 63.9855952
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 213.30
Output dim: 8, lower bound: -63.9643699, upper bound: 63.9143999
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 213.30
Output dim: 8, lower bound: -63.9643699, upper bound: 63.9143999
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 213.30
Output dim: 8, lower bound: -63.9643890, upper bound: 63.9855952
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 213.30
Output dim: 8, lower bound: -63.9643890, upper bound: 63.9855952
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 213.30
Output dim: 8, lower bound: -63.9648769, upper bound: 63.9763906
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 213.30
Output dim: 8, lower bound: -63.9648769, upper bound: 63.9763906
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 213.30
Output dim: 8, lower bound: -64.0072418, upper bound: 64.0510899
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 213.30
Output dim: 8, lower bound: -64.0072418, upper bound: 63.9799332
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 213.30
Output dim: 8, lower bound: -64.0072418, upper bound: 64.0510899

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 145.94 + 3506.26 = 3652.20 seconds
