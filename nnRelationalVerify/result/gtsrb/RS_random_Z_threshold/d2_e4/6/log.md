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
execution time: IAR + RelationalAnalysis = 2.61 + 145.51 = 148.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -64.0592927, upper bound: 64.0592927

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 526

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 514

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0578952, upper bound: 64.0592540
time: 87.35 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0592540, upper bound: 64.0578952
time: 95.08 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 182.45 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 182.45
Output dim: 8, lower bound: -64.0578952, upper bound: 64.0592540
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 182.45
Output dim: 8, lower bound: -64.0592540, upper bound: 64.0578952

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -74.4441681, 51.2501221, -74.4441681, 51.2501221, -125.6942902, 125.6942825
1: -35.7569427, 43.2586746, -35.7569427, 43.2586746, -79.0156097, 79.0156097
2: -30.5907345, 44.9403458, -30.5907345, 44.9403458, -75.5310822, 75.5310822
3: -35.0810623, 49.6928406, -35.0810623, 49.6928406, -84.7739029, 84.7739029
4: -40.9552193, 51.8817978, -40.9552193, 51.8817978, -92.8370209, 92.8370132
5: -34.2261429, 47.9517632, -34.2261429, 47.9517632, -82.1779022, 82.1779022
6: -70.8315430, 41.0467148, -70.8315430, 41.0467148, -111.8782501, 111.8782578
7: -43.1463509, 45.9498024, -43.1463509, 45.9498024, -89.0961456, 89.0961533
8: -48.8081589, 63.1175346, -48.8081589, 63.1175346, -111.9256897, 111.9256897
9: -41.9876938, 45.6585083, -41.9876938, 45.6585083, -87.6461945, 87.6462021
10: -63.0807076, 57.5979233, -63.0807076, 57.5979233, -120.6786346, 120.6786270
11: -60.7652092, 33.4686699, -60.7652092, 33.4686699, -94.2338791, 94.2338791
12: -67.9794083, 38.1531181, -67.9794083, 38.1531181, -106.1325226, 106.1325226
13: -65.2255249, 61.3555145, -65.2255249, 61.3555145, -126.5810394, 126.5810394
14: -101.7791443, 45.9462204, -101.7791443, 45.9462204, -147.7253571, 147.7253723
15: -49.4616508, 45.3876305, -49.4616508, 45.3876305, -94.8492737, 94.8492813
16: -63.1231842, 43.5457993, -63.1231842, 43.5457993, -106.6689758, 106.6689758
17: -97.0565567, 40.5237808, -97.0565567, 40.5237808, -137.5803375, 137.5803223
18: -64.3055725, 40.1181068, -64.3055725, 40.1181068, -104.4236755, 104.4236755
19: -46.9785995, 28.0002918, -46.9785995, 28.0002918, -74.9788895, 74.9788895
20: -45.3601379, 29.8137321, -45.3601379, 29.8137321, -75.1738663, 75.1738663
21: -59.0123825, 31.9141216, -59.0123825, 31.9141216, -90.9265060, 90.9265060
22: -58.7619171, 34.4748764, -58.7619171, 34.4748764, -93.2367859, 93.2367935
23: -46.6954193, 36.9088135, -46.6954193, 36.9088135, -83.6042328, 83.6042328
24: -57.0340157, 37.7480583, -57.0340157, 37.7480583, -94.7820740, 94.7820740
25: -50.1265450, 38.0982132, -50.1265450, 38.0982132, -88.2247543, 88.2247620
26: -70.8370209, 45.6955872, -70.8370209, 45.6955872, -116.5326080, 116.5326080
27: -57.3656578, 39.1326370, -57.3656578, 39.1326370, -96.4982910, 96.4982910
28: -46.4588852, 38.8290672, -46.4588852, 38.8290672, -85.2879486, 85.2879486
29: -60.1362267, 32.4817734, -60.1362267, 32.4817734, -92.6179962, 92.6179962
30: -58.1264915, 41.4087524, -58.1264915, 41.4087524, -99.5352478, 99.5352402
31: -62.8394241, 38.8694992, -62.8394241, 38.8694992, -101.7089233, 101.7089157
32: -68.7925339, 35.4185715, -68.7925339, 35.4185715, -104.2111053, 104.2111053
33: -87.5928345, 47.7316704, -87.5928345, 47.7316704, -135.3244934, 135.3244934
34: -74.3134232, 31.5479107, -74.3134232, 31.5479107, -105.8613281, 105.8613281
35: -72.0159607, 37.8115234, -72.0159607, 37.8115234, -109.8274689, 109.8274841
36: -75.3385239, 38.7858200, -75.3385239, 38.7858200, -114.1243439, 114.1243362
37: -106.4644547, 30.5911770, -106.4644547, 30.5911770, -137.0556335, 137.0556335
38: -88.1569672, 41.6150589, -88.1569672, 41.6150589, -129.7720337, 129.7720184
39: -98.3843613, 45.1909943, -98.3843613, 45.1909943, -143.5753326, 143.5753479
40: -81.9499817, 30.8376102, -81.9499817, 30.8376102, -112.7875900, 112.7875900
41: -69.2990265, 43.2153130, -69.2990265, 43.2153130, -112.5143280, 112.5143356
42: -50.8413391, 35.0825958, -50.8413391, 35.0825958, -85.9239349, 85.9239349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=412, inp2_unstable=412, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1640

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 569

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0565201, upper bound: 64.0563080
time: 99.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0549452, upper bound: 64.0578760
time: 1018.03 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -74.4441681, 51.2501221, -74.4441681, 51.2501221, -125.6942902, 125.6942825
1: -35.7569427, 43.2586746, -35.7569427, 43.2586746, -79.0156097, 79.0156097
2: -30.5907345, 44.9403458, -30.5907345, 44.9403458, -75.5310822, 75.5310822
3: -35.0810623, 49.6928406, -35.0810623, 49.6928406, -84.7739029, 84.7739029
4: -40.9552193, 51.8817978, -40.9552193, 51.8817978, -92.8370209, 92.8370132
5: -34.2261429, 47.9517632, -34.2261429, 47.9517632, -82.1779022, 82.1779022
6: -70.8315430, 41.0467148, -70.8315430, 41.0467148, -111.8782501, 111.8782578
7: -43.1463509, 45.9498024, -43.1463509, 45.9498024, -89.0961456, 89.0961533
8: -48.8081589, 63.1175346, -48.8081589, 63.1175346, -111.9256897, 111.9256897
9: -41.9876938, 45.6585083, -41.9876938, 45.6585083, -87.6461945, 87.6462021
10: -63.0807076, 57.5979233, -63.0807076, 57.5979233, -120.6786346, 120.6786270
11: -60.7652092, 33.4686699, -60.7652092, 33.4686699, -94.2338791, 94.2338791
12: -67.9794083, 38.1531181, -67.9794083, 38.1531181, -106.1325226, 106.1325226
13: -65.2255249, 61.3555145, -65.2255249, 61.3555145, -126.5810394, 126.5810394
14: -101.7791443, 45.9462204, -101.7791443, 45.9462204, -147.7253571, 147.7253723
15: -49.4616508, 45.3876305, -49.4616508, 45.3876305, -94.8492737, 94.8492813
16: -63.1231842, 43.5457993, -63.1231842, 43.5457993, -106.6689758, 106.6689758
17: -97.0565567, 40.5237808, -97.0565567, 40.5237808, -137.5803375, 137.5803223
18: -64.3055725, 40.1181068, -64.3055725, 40.1181068, -104.4236755, 104.4236755
19: -46.9785995, 28.0002918, -46.9785995, 28.0002918, -74.9788895, 74.9788895
20: -45.3601379, 29.8137321, -45.3601379, 29.8137321, -75.1738663, 75.1738663
21: -59.0123825, 31.9141216, -59.0123825, 31.9141216, -90.9265060, 90.9265060
22: -58.7619171, 34.4748764, -58.7619171, 34.4748764, -93.2367859, 93.2367935
23: -46.6954193, 36.9088135, -46.6954193, 36.9088135, -83.6042328, 83.6042328
24: -57.0340157, 37.7480583, -57.0340157, 37.7480583, -94.7820740, 94.7820740
25: -50.1265450, 38.0982132, -50.1265450, 38.0982132, -88.2247543, 88.2247620
26: -70.8370209, 45.6955872, -70.8370209, 45.6955872, -116.5326080, 116.5326080
27: -57.3656578, 39.1326370, -57.3656578, 39.1326370, -96.4982910, 96.4982910
28: -46.4588852, 38.8290672, -46.4588852, 38.8290672, -85.2879486, 85.2879486
29: -60.1362267, 32.4817734, -60.1362267, 32.4817734, -92.6179962, 92.6179962
30: -58.1264915, 41.4087524, -58.1264915, 41.4087524, -99.5352478, 99.5352402
31: -62.8394241, 38.8694992, -62.8394241, 38.8694992, -101.7089233, 101.7089157
32: -68.7925339, 35.4185715, -68.7925339, 35.4185715, -104.2111053, 104.2111053
33: -87.5928345, 47.7316704, -87.5928345, 47.7316704, -135.3244934, 135.3244934
34: -74.3134232, 31.5479107, -74.3134232, 31.5479107, -105.8613281, 105.8613281
35: -72.0159607, 37.8115234, -72.0159607, 37.8115234, -109.8274689, 109.8274841
36: -75.3385239, 38.7858200, -75.3385239, 38.7858200, -114.1243439, 114.1243362
37: -106.4644547, 30.5911770, -106.4644547, 30.5911770, -137.0556335, 137.0556335
38: -88.1569672, 41.6150589, -88.1569672, 41.6150589, -129.7720337, 129.7720184
39: -98.3843613, 45.1909943, -98.3843613, 45.1909943, -143.5753326, 143.5753479
40: -81.9499817, 30.8376102, -81.9499817, 30.8376102, -112.7875900, 112.7875900
41: -69.2990265, 43.2153130, -69.2990265, 43.2153130, -112.5143280, 112.5143356
42: -50.8413391, 35.0825958, -50.8413391, 35.0825958, -85.9239349, 85.9239349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=412, inp2_unstable=412, delta_unstable=2047
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1619

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1548

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0545901, upper bound: 64.0576886
time: 90.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0545901, upper bound: 64.0545901
time: 109.98 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 201.99 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 201.99
Output dim: 8, lower bound: -64.0565201, upper bound: 64.0563080
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 201.99
Output dim: 8, lower bound: -64.0549452, upper bound: 64.0578760
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 201.99
Output dim: 8, lower bound: -64.0545901, upper bound: 64.0576886
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 201.99
Output dim: 8, lower bound: -64.0545901, upper bound: 64.0545901

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -74.4441681, 51.2501221, -74.4441681, 51.2501221, -125.6942902, 125.6942825
1: -35.7569427, 43.2586746, -35.7569427, 43.2586746, -79.0156097, 79.0156097
2: -30.5907345, 44.9403458, -30.5907345, 44.9403458, -75.5310822, 75.5310822
3: -35.0810623, 49.6928406, -35.0810623, 49.6928406, -84.7739029, 84.7739029
4: -40.9552193, 51.8817978, -40.9552193, 51.8817978, -92.8370209, 92.8370132
5: -34.2261429, 47.9517632, -34.2261429, 47.9517632, -82.1779022, 82.1779022
6: -70.8315430, 41.0467148, -70.8315430, 41.0467148, -111.8782501, 111.8782578
7: -43.1463509, 45.9498024, -43.1463509, 45.9498024, -89.0961456, 89.0961533
8: -48.8081589, 63.1175346, -48.8081589, 63.1175346, -111.9256897, 111.9256897
9: -41.9876938, 45.6585083, -41.9876938, 45.6585083, -87.6461945, 87.6462021
10: -63.0807076, 57.5979233, -63.0807076, 57.5979233, -120.6786346, 120.6786270
11: -60.7652092, 33.4686699, -60.7652092, 33.4686699, -94.2338791, 94.2338791
12: -67.9794083, 38.1531181, -67.9794083, 38.1531181, -106.1325226, 106.1325226
13: -65.2255249, 61.3555145, -65.2255249, 61.3555145, -126.5810394, 126.5810394
14: -101.7791443, 45.9462204, -101.7791443, 45.9462204, -147.7253571, 147.7253723
15: -49.4616508, 45.3876305, -49.4616508, 45.3876305, -94.8492737, 94.8492813
16: -63.1231842, 43.5457993, -63.1231842, 43.5457993, -106.6689758, 106.6689758
17: -97.0565567, 40.5237808, -97.0565567, 40.5237808, -137.5803375, 137.5803223
18: -64.3055725, 40.1181068, -64.3055725, 40.1181068, -104.4236755, 104.4236755
19: -46.9785995, 28.0002918, -46.9785995, 28.0002918, -74.9788895, 74.9788895
20: -45.3601379, 29.8137321, -45.3601379, 29.8137321, -75.1738663, 75.1738663
21: -59.0123825, 31.9141216, -59.0123825, 31.9141216, -90.9265060, 90.9265060
22: -58.7619171, 34.4748764, -58.7619171, 34.4748764, -93.2367859, 93.2367935
23: -46.6954193, 36.9088135, -46.6954193, 36.9088135, -83.6042328, 83.6042328
24: -57.0340157, 37.7480583, -57.0340157, 37.7480583, -94.7820740, 94.7820740
25: -50.1265450, 38.0982132, -50.1265450, 38.0982132, -88.2247543, 88.2247620
26: -70.8370209, 45.6955872, -70.8370209, 45.6955872, -116.5326080, 116.5326080
27: -57.3656578, 39.1326370, -57.3656578, 39.1326370, -96.4982910, 96.4982910
28: -46.4588852, 38.8290672, -46.4588852, 38.8290672, -85.2879486, 85.2879486
29: -60.1362267, 32.4817734, -60.1362267, 32.4817734, -92.6179962, 92.6179962
30: -58.1264915, 41.4087524, -58.1264915, 41.4087524, -99.5352478, 99.5352402
31: -62.8394241, 38.8694992, -62.8394241, 38.8694992, -101.7089233, 101.7089157
32: -68.7925339, 35.4185715, -68.7925339, 35.4185715, -104.2111053, 104.2111053
33: -87.5928345, 47.7316704, -87.5928345, 47.7316704, -135.3244934, 135.3244934
34: -74.3134232, 31.5479107, -74.3134232, 31.5479107, -105.8613281, 105.8613281
35: -72.0159607, 37.8115234, -72.0159607, 37.8115234, -109.8274689, 109.8274841
36: -75.3385239, 38.7858200, -75.3385239, 38.7858200, -114.1243439, 114.1243362
37: -106.4644547, 30.5911770, -106.4644547, 30.5911770, -137.0556335, 137.0556335
38: -88.1569672, 41.6150589, -88.1569672, 41.6150589, -129.7720337, 129.7720184
39: -98.3843613, 45.1909943, -98.3843613, 45.1909943, -143.5753326, 143.5753479
40: -81.9499817, 30.8376102, -81.9499817, 30.8376102, -112.7875900, 112.7875900
41: -69.2990265, 43.2153130, -69.2990265, 43.2153130, -112.5143280, 112.5143356
42: -50.8413391, 35.0825958, -50.8413391, 35.0825958, -85.9239349, 85.9239349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=412, inp2_unstable=412, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1744

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 654

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0158428, upper bound: 64.0546715
time: 105.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0548797, upper bound: 64.0156313
time: 107.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -74.4441681, 51.2501221, -74.4441681, 51.2501221, -125.6942902, 125.6942825
1: -35.7569427, 43.2586746, -35.7569427, 43.2586746, -79.0156097, 79.0156097
2: -30.5907345, 44.9403458, -30.5907345, 44.9403458, -75.5310822, 75.5310822
3: -35.0810623, 49.6928406, -35.0810623, 49.6928406, -84.7739029, 84.7739029
4: -40.9552193, 51.8817978, -40.9552193, 51.8817978, -92.8370209, 92.8370132
5: -34.2261429, 47.9517632, -34.2261429, 47.9517632, -82.1779022, 82.1779022
6: -70.8315430, 41.0467148, -70.8315430, 41.0467148, -111.8782501, 111.8782578
7: -43.1463509, 45.9498024, -43.1463509, 45.9498024, -89.0961456, 89.0961533
8: -48.8081589, 63.1175346, -48.8081589, 63.1175346, -111.9256897, 111.9256897
9: -41.9876938, 45.6585083, -41.9876938, 45.6585083, -87.6461945, 87.6462021
10: -63.0807076, 57.5979233, -63.0807076, 57.5979233, -120.6786346, 120.6786270
11: -60.7652092, 33.4686699, -60.7652092, 33.4686699, -94.2338791, 94.2338791
12: -67.9794083, 38.1531181, -67.9794083, 38.1531181, -106.1325226, 106.1325226
13: -65.2255249, 61.3555145, -65.2255249, 61.3555145, -126.5810394, 126.5810394
14: -101.7791443, 45.9462204, -101.7791443, 45.9462204, -147.7253571, 147.7253723
15: -49.4616508, 45.3876305, -49.4616508, 45.3876305, -94.8492737, 94.8492813
16: -63.1231842, 43.5457993, -63.1231842, 43.5457993, -106.6689758, 106.6689758
17: -97.0565567, 40.5237808, -97.0565567, 40.5237808, -137.5803375, 137.5803223
18: -64.3055725, 40.1181068, -64.3055725, 40.1181068, -104.4236755, 104.4236755
19: -46.9785995, 28.0002918, -46.9785995, 28.0002918, -74.9788895, 74.9788895
20: -45.3601379, 29.8137321, -45.3601379, 29.8137321, -75.1738663, 75.1738663
21: -59.0123825, 31.9141216, -59.0123825, 31.9141216, -90.9265060, 90.9265060
22: -58.7619171, 34.4748764, -58.7619171, 34.4748764, -93.2367859, 93.2367935
23: -46.6954193, 36.9088135, -46.6954193, 36.9088135, -83.6042328, 83.6042328
24: -57.0340157, 37.7480583, -57.0340157, 37.7480583, -94.7820740, 94.7820740
25: -50.1265450, 38.0982132, -50.1265450, 38.0982132, -88.2247543, 88.2247620
26: -70.8370209, 45.6955872, -70.8370209, 45.6955872, -116.5326080, 116.5326080
27: -57.3656578, 39.1326370, -57.3656578, 39.1326370, -96.4982910, 96.4982910
28: -46.4588852, 38.8290672, -46.4588852, 38.8290672, -85.2879486, 85.2879486
29: -60.1362267, 32.4817734, -60.1362267, 32.4817734, -92.6179962, 92.6179962
30: -58.1264915, 41.4087524, -58.1264915, 41.4087524, -99.5352478, 99.5352402
31: -62.8394241, 38.8694992, -62.8394241, 38.8694992, -101.7089233, 101.7089157
32: -68.7925339, 35.4185715, -68.7925339, 35.4185715, -104.2111053, 104.2111053
33: -87.5928345, 47.7316704, -87.5928345, 47.7316704, -135.3244934, 135.3244934
34: -74.3134232, 31.5479107, -74.3134232, 31.5479107, -105.8613281, 105.8613281
35: -72.0159607, 37.8115234, -72.0159607, 37.8115234, -109.8274689, 109.8274841
36: -75.3385239, 38.7858200, -75.3385239, 38.7858200, -114.1243439, 114.1243362
37: -106.4644547, 30.5911770, -106.4644547, 30.5911770, -137.0556335, 137.0556335
38: -88.1569672, 41.6150589, -88.1569672, 41.6150589, -129.7720337, 129.7720184
39: -98.3843613, 45.1909943, -98.3843613, 45.1909943, -143.5753326, 143.5753479
40: -81.9499817, 30.8376102, -81.9499817, 30.8376102, -112.7875900, 112.7875900
41: -69.2990265, 43.2153130, -69.2990265, 43.2153130, -112.5143280, 112.5143356
42: -50.8413391, 35.0825958, -50.8413391, 35.0825958, -85.9239349, 85.9239349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=412, inp2_unstable=412, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 670

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1573

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0540604, upper bound: 64.0518486
time: 103.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0489132, upper bound: 64.0570001
time: 161.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -74.4441681, 51.2501221, -74.4441681, 51.2501221, -125.6942902, 125.6942825
1: -35.7569427, 43.2586746, -35.7569427, 43.2586746, -79.0156097, 79.0156097
2: -30.5907345, 44.9403458, -30.5907345, 44.9403458, -75.5310822, 75.5310822
3: -35.0810623, 49.6928406, -35.0810623, 49.6928406, -84.7739029, 84.7739029
4: -40.9552193, 51.8817978, -40.9552193, 51.8817978, -92.8370209, 92.8370132
5: -34.2261429, 47.9517632, -34.2261429, 47.9517632, -82.1779022, 82.1779022
6: -70.8315430, 41.0467148, -70.8315430, 41.0467148, -111.8782501, 111.8782578
7: -43.1463509, 45.9498024, -43.1463509, 45.9498024, -89.0961456, 89.0961533
8: -48.8081589, 63.1175346, -48.8081589, 63.1175346, -111.9256897, 111.9256897
9: -41.9876938, 45.6585083, -41.9876938, 45.6585083, -87.6461945, 87.6462021
10: -63.0807076, 57.5979233, -63.0807076, 57.5979233, -120.6786346, 120.6786270
11: -60.7652092, 33.4686699, -60.7652092, 33.4686699, -94.2338791, 94.2338791
12: -67.9794083, 38.1531181, -67.9794083, 38.1531181, -106.1325226, 106.1325226
13: -65.2255249, 61.3555145, -65.2255249, 61.3555145, -126.5810394, 126.5810394
14: -101.7791443, 45.9462204, -101.7791443, 45.9462204, -147.7253571, 147.7253723
15: -49.4616508, 45.3876305, -49.4616508, 45.3876305, -94.8492737, 94.8492813
16: -63.1231842, 43.5457993, -63.1231842, 43.5457993, -106.6689758, 106.6689758
17: -97.0565567, 40.5237808, -97.0565567, 40.5237808, -137.5803375, 137.5803223
18: -64.3055725, 40.1181068, -64.3055725, 40.1181068, -104.4236755, 104.4236755
19: -46.9785995, 28.0002918, -46.9785995, 28.0002918, -74.9788895, 74.9788895
20: -45.3601379, 29.8137321, -45.3601379, 29.8137321, -75.1738663, 75.1738663
21: -59.0123825, 31.9141216, -59.0123825, 31.9141216, -90.9265060, 90.9265060
22: -58.7619171, 34.4748764, -58.7619171, 34.4748764, -93.2367859, 93.2367935
23: -46.6954193, 36.9088135, -46.6954193, 36.9088135, -83.6042328, 83.6042328
24: -57.0340157, 37.7480583, -57.0340157, 37.7480583, -94.7820740, 94.7820740
25: -50.1265450, 38.0982132, -50.1265450, 38.0982132, -88.2247543, 88.2247620
26: -70.8370209, 45.6955872, -70.8370209, 45.6955872, -116.5326080, 116.5326080
27: -57.3656578, 39.1326370, -57.3656578, 39.1326370, -96.4982910, 96.4982910
28: -46.4588852, 38.8290672, -46.4588852, 38.8290672, -85.2879486, 85.2879486
29: -60.1362267, 32.4817734, -60.1362267, 32.4817734, -92.6179962, 92.6179962
30: -58.1264915, 41.4087524, -58.1264915, 41.4087524, -99.5352478, 99.5352402
31: -62.8394241, 38.8694992, -62.8394241, 38.8694992, -101.7089233, 101.7089157
32: -68.7925339, 35.4185715, -68.7925339, 35.4185715, -104.2111053, 104.2111053
33: -87.5928345, 47.7316704, -87.5928345, 47.7316704, -135.3244934, 135.3244934
34: -74.3134232, 31.5479107, -74.3134232, 31.5479107, -105.8613281, 105.8613281
35: -72.0159607, 37.8115234, -72.0159607, 37.8115234, -109.8274689, 109.8274841
36: -75.3385239, 38.7858200, -75.3385239, 38.7858200, -114.1243439, 114.1243362
37: -106.4644547, 30.5911770, -106.4644547, 30.5911770, -137.0556335, 137.0556335
38: -88.1569672, 41.6150589, -88.1569672, 41.6150589, -129.7720337, 129.7720184
39: -98.3843613, 45.1909943, -98.3843613, 45.1909943, -143.5753326, 143.5753479
40: -81.9499817, 30.8376102, -81.9499817, 30.8376102, -112.7875900, 112.7875900
41: -69.2990265, 43.2153130, -69.2990265, 43.2153130, -112.5143280, 112.5143356
42: -50.8413391, 35.0825958, -50.8413391, 35.0825958, -85.9239349, 85.9239349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=412, inp2_unstable=412, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1593

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 872

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0541299, upper bound: 64.0569747
time: 80.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0538760, upper bound: 64.0572269
time: 99.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -74.4441681, 51.2501221, -74.4441681, 51.2501221, -125.6942902, 125.6942825
1: -35.7569427, 43.2586746, -35.7569427, 43.2586746, -79.0156097, 79.0156097
2: -30.5907345, 44.9403458, -30.5907345, 44.9403458, -75.5310822, 75.5310822
3: -35.0810623, 49.6928406, -35.0810623, 49.6928406, -84.7739029, 84.7739029
4: -40.9552193, 51.8817978, -40.9552193, 51.8817978, -92.8370209, 92.8370132
5: -34.2261429, 47.9517632, -34.2261429, 47.9517632, -82.1779022, 82.1779022
6: -70.8315430, 41.0467148, -70.8315430, 41.0467148, -111.8782501, 111.8782578
7: -43.1463509, 45.9498024, -43.1463509, 45.9498024, -89.0961456, 89.0961533
8: -48.8081589, 63.1175346, -48.8081589, 63.1175346, -111.9256897, 111.9256897
9: -41.9876938, 45.6585083, -41.9876938, 45.6585083, -87.6461945, 87.6462021
10: -63.0807076, 57.5979233, -63.0807076, 57.5979233, -120.6786346, 120.6786270
11: -60.7652092, 33.4686699, -60.7652092, 33.4686699, -94.2338791, 94.2338791
12: -67.9794083, 38.1531181, -67.9794083, 38.1531181, -106.1325226, 106.1325226
13: -65.2255249, 61.3555145, -65.2255249, 61.3555145, -126.5810394, 126.5810394
14: -101.7791443, 45.9462204, -101.7791443, 45.9462204, -147.7253571, 147.7253723
15: -49.4616508, 45.3876305, -49.4616508, 45.3876305, -94.8492737, 94.8492813
16: -63.1231842, 43.5457993, -63.1231842, 43.5457993, -106.6689758, 106.6689758
17: -97.0565567, 40.5237808, -97.0565567, 40.5237808, -137.5803375, 137.5803223
18: -64.3055725, 40.1181068, -64.3055725, 40.1181068, -104.4236755, 104.4236755
19: -46.9785995, 28.0002918, -46.9785995, 28.0002918, -74.9788895, 74.9788895
20: -45.3601379, 29.8137321, -45.3601379, 29.8137321, -75.1738663, 75.1738663
21: -59.0123825, 31.9141216, -59.0123825, 31.9141216, -90.9265060, 90.9265060
22: -58.7619171, 34.4748764, -58.7619171, 34.4748764, -93.2367859, 93.2367935
23: -46.6954193, 36.9088135, -46.6954193, 36.9088135, -83.6042328, 83.6042328
24: -57.0340157, 37.7480583, -57.0340157, 37.7480583, -94.7820740, 94.7820740
25: -50.1265450, 38.0982132, -50.1265450, 38.0982132, -88.2247543, 88.2247620
26: -70.8370209, 45.6955872, -70.8370209, 45.6955872, -116.5326080, 116.5326080
27: -57.3656578, 39.1326370, -57.3656578, 39.1326370, -96.4982910, 96.4982910
28: -46.4588852, 38.8290672, -46.4588852, 38.8290672, -85.2879486, 85.2879486
29: -60.1362267, 32.4817734, -60.1362267, 32.4817734, -92.6179962, 92.6179962
30: -58.1264915, 41.4087524, -58.1264915, 41.4087524, -99.5352478, 99.5352402
31: -62.8394241, 38.8694992, -62.8394241, 38.8694992, -101.7089233, 101.7089157
32: -68.7925339, 35.4185715, -68.7925339, 35.4185715, -104.2111053, 104.2111053
33: -87.5928345, 47.7316704, -87.5928345, 47.7316704, -135.3244934, 135.3244934
34: -74.3134232, 31.5479107, -74.3134232, 31.5479107, -105.8613281, 105.8613281
35: -72.0159607, 37.8115234, -72.0159607, 37.8115234, -109.8274689, 109.8274841
36: -75.3385239, 38.7858200, -75.3385239, 38.7858200, -114.1243439, 114.1243362
37: -106.4644547, 30.5911770, -106.4644547, 30.5911770, -137.0556335, 137.0556335
38: -88.1569672, 41.6150589, -88.1569672, 41.6150589, -129.7720337, 129.7720184
39: -98.3843613, 45.1909943, -98.3843613, 45.1909943, -143.5753326, 143.5753479
40: -81.9499817, 30.8376102, -81.9499817, 30.8376102, -112.7875900, 112.7875900
41: -69.2990265, 43.2153130, -69.2990265, 43.2153130, -112.5143280, 112.5143356
42: -50.8413391, 35.0825958, -50.8413391, 35.0825958, -85.9239349, 85.9239349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=412, inp2_unstable=412, delta_unstable=2046
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1763

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1541

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0514104, upper bound: 64.0540362
time: 95.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0514104, upper bound: 64.0482994
time: 88.95 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 186.66 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 186.66
Output dim: 8, lower bound: -64.0158428, upper bound: 64.0546715
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 186.66
Output dim: 8, lower bound: -64.0548797, upper bound: 64.0156313
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 186.66
Output dim: 8, lower bound: -64.0540604, upper bound: 64.0518486
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 186.66
Output dim: 8, lower bound: -64.0489132, upper bound: 64.0570001
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 186.66
Output dim: 8, lower bound: -64.0541299, upper bound: 64.0569747
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 186.66
Output dim: 8, lower bound: -64.0538760, upper bound: 64.0572269
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 186.66
Output dim: 8, lower bound: -64.0514104, upper bound: 64.0540362
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 186.66
Output dim: 8, lower bound: -64.0514104, upper bound: 64.0482994

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -74.4441681, 51.2501221, -74.4441681, 51.2501221, -125.6942902, 125.6942825
1: -35.7569427, 43.2586746, -35.7569427, 43.2586746, -79.0156097, 79.0156097
2: -30.5907345, 44.9403458, -30.5907345, 44.9403458, -75.5310822, 75.5310822
3: -35.0810623, 49.6928406, -35.0810623, 49.6928406, -84.7739029, 84.7739029
4: -40.9552193, 51.8817978, -40.9552193, 51.8817978, -92.8370209, 92.8370132
5: -34.2261429, 47.9517632, -34.2261429, 47.9517632, -82.1779022, 82.1779022
6: -70.8315430, 41.0467148, -70.8315430, 41.0467148, -111.8782501, 111.8782578
7: -43.1463509, 45.9498024, -43.1463509, 45.9498024, -89.0961456, 89.0961533
8: -48.8081589, 63.1175346, -48.8081589, 63.1175346, -111.9256897, 111.9256897
9: -41.9876938, 45.6585083, -41.9876938, 45.6585083, -87.6461945, 87.6462021
10: -63.0807076, 57.5979233, -63.0807076, 57.5979233, -120.6786346, 120.6786270
11: -60.7652092, 33.4686699, -60.7652092, 33.4686699, -94.2338791, 94.2338791
12: -67.9794083, 38.1531181, -67.9794083, 38.1531181, -106.1325226, 106.1325226
13: -65.2255249, 61.3555145, -65.2255249, 61.3555145, -126.5810394, 126.5810394
14: -101.7791443, 45.9462204, -101.7791443, 45.9462204, -147.7253571, 147.7253723
15: -49.4616508, 45.3876305, -49.4616508, 45.3876305, -94.8492737, 94.8492813
16: -63.1231842, 43.5457993, -63.1231842, 43.5457993, -106.6689758, 106.6689758
17: -97.0565567, 40.5237808, -97.0565567, 40.5237808, -137.5803375, 137.5803223
18: -64.3055725, 40.1181068, -64.3055725, 40.1181068, -104.4236755, 104.4236755
19: -46.9785995, 28.0002918, -46.9785995, 28.0002918, -74.9788895, 74.9788895
20: -45.3601379, 29.8137321, -45.3601379, 29.8137321, -75.1738663, 75.1738663
21: -59.0123825, 31.9141216, -59.0123825, 31.9141216, -90.9265060, 90.9265060
22: -58.7619171, 34.4748764, -58.7619171, 34.4748764, -93.2367859, 93.2367935
23: -46.6954193, 36.9088135, -46.6954193, 36.9088135, -83.6042328, 83.6042328
24: -57.0340157, 37.7480583, -57.0340157, 37.7480583, -94.7820740, 94.7820740
25: -50.1265450, 38.0982132, -50.1265450, 38.0982132, -88.2247543, 88.2247620
26: -70.8370209, 45.6955872, -70.8370209, 45.6955872, -116.5326080, 116.5326080
27: -57.3656578, 39.1326370, -57.3656578, 39.1326370, -96.4982910, 96.4982910
28: -46.4588852, 38.8290672, -46.4588852, 38.8290672, -85.2879486, 85.2879486
29: -60.1362267, 32.4817734, -60.1362267, 32.4817734, -92.6179962, 92.6179962
30: -58.1264915, 41.4087524, -58.1264915, 41.4087524, -99.5352478, 99.5352402
31: -62.8394241, 38.8694992, -62.8394241, 38.8694992, -101.7089233, 101.7089157
32: -68.7925339, 35.4185715, -68.7925339, 35.4185715, -104.2111053, 104.2111053
33: -87.5928345, 47.7316704, -87.5928345, 47.7316704, -135.3244934, 135.3244934
34: -74.3134232, 31.5479107, -74.3134232, 31.5479107, -105.8613281, 105.8613281
35: -72.0159607, 37.8115234, -72.0159607, 37.8115234, -109.8274689, 109.8274841
36: -75.3385239, 38.7858200, -75.3385239, 38.7858200, -114.1243439, 114.1243362
37: -106.4644547, 30.5911770, -106.4644547, 30.5911770, -137.0556335, 137.0556335
38: -88.1569672, 41.6150589, -88.1569672, 41.6150589, -129.7720337, 129.7720184
39: -98.3843613, 45.1909943, -98.3843613, 45.1909943, -143.5753326, 143.5753479
40: -81.9499817, 30.8376102, -81.9499817, 30.8376102, -112.7875900, 112.7875900
41: -69.2990265, 43.2153130, -69.2990265, 43.2153130, -112.5143280, 112.5143356
42: -50.8413391, 35.0825958, -50.8413391, 35.0825958, -85.9239349, 85.9239349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=412, inp2_unstable=412, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1757

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1018

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -63.9911935, upper bound: 64.0536370
time: 95.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0147409, upper bound: 64.0316461
time: 108.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -74.4441681, 51.2501221, -74.4441681, 51.2501221, -125.6942902, 125.6942825
1: -35.7569427, 43.2586746, -35.7569427, 43.2586746, -79.0156097, 79.0156097
2: -30.5907345, 44.9403458, -30.5907345, 44.9403458, -75.5310822, 75.5310822
3: -35.0810623, 49.6928406, -35.0810623, 49.6928406, -84.7739029, 84.7739029
4: -40.9552193, 51.8817978, -40.9552193, 51.8817978, -92.8370209, 92.8370132
5: -34.2261429, 47.9517632, -34.2261429, 47.9517632, -82.1779022, 82.1779022
6: -70.8315430, 41.0467148, -70.8315430, 41.0467148, -111.8782501, 111.8782578
7: -43.1463509, 45.9498024, -43.1463509, 45.9498024, -89.0961456, 89.0961533
8: -48.8081589, 63.1175346, -48.8081589, 63.1175346, -111.9256897, 111.9256897
9: -41.9876938, 45.6585083, -41.9876938, 45.6585083, -87.6461945, 87.6462021
10: -63.0807076, 57.5979233, -63.0807076, 57.5979233, -120.6786346, 120.6786270
11: -60.7652092, 33.4686699, -60.7652092, 33.4686699, -94.2338791, 94.2338791
12: -67.9794083, 38.1531181, -67.9794083, 38.1531181, -106.1325226, 106.1325226
13: -65.2255249, 61.3555145, -65.2255249, 61.3555145, -126.5810394, 126.5810394
14: -101.7791443, 45.9462204, -101.7791443, 45.9462204, -147.7253571, 147.7253723
15: -49.4616508, 45.3876305, -49.4616508, 45.3876305, -94.8492737, 94.8492813
16: -63.1231842, 43.5457993, -63.1231842, 43.5457993, -106.6689758, 106.6689758
17: -97.0565567, 40.5237808, -97.0565567, 40.5237808, -137.5803375, 137.5803223
18: -64.3055725, 40.1181068, -64.3055725, 40.1181068, -104.4236755, 104.4236755
19: -46.9785995, 28.0002918, -46.9785995, 28.0002918, -74.9788895, 74.9788895
20: -45.3601379, 29.8137321, -45.3601379, 29.8137321, -75.1738663, 75.1738663
21: -59.0123825, 31.9141216, -59.0123825, 31.9141216, -90.9265060, 90.9265060
22: -58.7619171, 34.4748764, -58.7619171, 34.4748764, -93.2367859, 93.2367935
23: -46.6954193, 36.9088135, -46.6954193, 36.9088135, -83.6042328, 83.6042328
24: -57.0340157, 37.7480583, -57.0340157, 37.7480583, -94.7820740, 94.7820740
25: -50.1265450, 38.0982132, -50.1265450, 38.0982132, -88.2247543, 88.2247620
26: -70.8370209, 45.6955872, -70.8370209, 45.6955872, -116.5326080, 116.5326080
27: -57.3656578, 39.1326370, -57.3656578, 39.1326370, -96.4982910, 96.4982910
28: -46.4588852, 38.8290672, -46.4588852, 38.8290672, -85.2879486, 85.2879486
29: -60.1362267, 32.4817734, -60.1362267, 32.4817734, -92.6179962, 92.6179962
30: -58.1264915, 41.4087524, -58.1264915, 41.4087524, -99.5352478, 99.5352402
31: -62.8394241, 38.8694992, -62.8394241, 38.8694992, -101.7089233, 101.7089157
32: -68.7925339, 35.4185715, -68.7925339, 35.4185715, -104.2111053, 104.2111053
33: -87.5928345, 47.7316704, -87.5928345, 47.7316704, -135.3244934, 135.3244934
34: -74.3134232, 31.5479107, -74.3134232, 31.5479107, -105.8613281, 105.8613281
35: -72.0159607, 37.8115234, -72.0159607, 37.8115234, -109.8274689, 109.8274841
36: -75.3385239, 38.7858200, -75.3385239, 38.7858200, -114.1243439, 114.1243362
37: -106.4644547, 30.5911770, -106.4644547, 30.5911770, -137.0556335, 137.0556335
38: -88.1569672, 41.6150589, -88.1569672, 41.6150589, -129.7720337, 129.7720184
39: -98.3843613, 45.1909943, -98.3843613, 45.1909943, -143.5753326, 143.5753479
40: -81.9499817, 30.8376102, -81.9499817, 30.8376102, -112.7875900, 112.7875900
41: -69.2990265, 43.2153130, -69.2990265, 43.2153130, -112.5143280, 112.5143356
42: -50.8413391, 35.0825958, -50.8413391, 35.0825958, -85.9239349, 85.9239349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=412, inp2_unstable=412, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 740

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1589

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0534827, upper bound: 63.9959682
time: 100.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0352234, upper bound: 64.0141974
time: 122.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -74.4441681, 51.2501221, -74.4441681, 51.2501221, -125.6942902, 125.6942825
1: -35.7569427, 43.2586746, -35.7569427, 43.2586746, -79.0156097, 79.0156097
2: -30.5907345, 44.9403458, -30.5907345, 44.9403458, -75.5310822, 75.5310822
3: -35.0810623, 49.6928406, -35.0810623, 49.6928406, -84.7739029, 84.7739029
4: -40.9552193, 51.8817978, -40.9552193, 51.8817978, -92.8370209, 92.8370132
5: -34.2261429, 47.9517632, -34.2261429, 47.9517632, -82.1779022, 82.1779022
6: -70.8315430, 41.0467148, -70.8315430, 41.0467148, -111.8782501, 111.8782578
7: -43.1463509, 45.9498024, -43.1463509, 45.9498024, -89.0961456, 89.0961533
8: -48.8081589, 63.1175346, -48.8081589, 63.1175346, -111.9256897, 111.9256897
9: -41.9876938, 45.6585083, -41.9876938, 45.6585083, -87.6461945, 87.6462021
10: -63.0807076, 57.5979233, -63.0807076, 57.5979233, -120.6786346, 120.6786270
11: -60.7652092, 33.4686699, -60.7652092, 33.4686699, -94.2338791, 94.2338791
12: -67.9794083, 38.1531181, -67.9794083, 38.1531181, -106.1325226, 106.1325226
13: -65.2255249, 61.3555145, -65.2255249, 61.3555145, -126.5810394, 126.5810394
14: -101.7791443, 45.9462204, -101.7791443, 45.9462204, -147.7253571, 147.7253723
15: -49.4616508, 45.3876305, -49.4616508, 45.3876305, -94.8492737, 94.8492813
16: -63.1231842, 43.5457993, -63.1231842, 43.5457993, -106.6689758, 106.6689758
17: -97.0565567, 40.5237808, -97.0565567, 40.5237808, -137.5803375, 137.5803223
18: -64.3055725, 40.1181068, -64.3055725, 40.1181068, -104.4236755, 104.4236755
19: -46.9785995, 28.0002918, -46.9785995, 28.0002918, -74.9788895, 74.9788895
20: -45.3601379, 29.8137321, -45.3601379, 29.8137321, -75.1738663, 75.1738663
21: -59.0123825, 31.9141216, -59.0123825, 31.9141216, -90.9265060, 90.9265060
22: -58.7619171, 34.4748764, -58.7619171, 34.4748764, -93.2367859, 93.2367935
23: -46.6954193, 36.9088135, -46.6954193, 36.9088135, -83.6042328, 83.6042328
24: -57.0340157, 37.7480583, -57.0340157, 37.7480583, -94.7820740, 94.7820740
25: -50.1265450, 38.0982132, -50.1265450, 38.0982132, -88.2247543, 88.2247620
26: -70.8370209, 45.6955872, -70.8370209, 45.6955872, -116.5326080, 116.5326080
27: -57.3656578, 39.1326370, -57.3656578, 39.1326370, -96.4982910, 96.4982910
28: -46.4588852, 38.8290672, -46.4588852, 38.8290672, -85.2879486, 85.2879486
29: -60.1362267, 32.4817734, -60.1362267, 32.4817734, -92.6179962, 92.6179962
30: -58.1264915, 41.4087524, -58.1264915, 41.4087524, -99.5352478, 99.5352402
31: -62.8394241, 38.8694992, -62.8394241, 38.8694992, -101.7089233, 101.7089157
32: -68.7925339, 35.4185715, -68.7925339, 35.4185715, -104.2111053, 104.2111053
33: -87.5928345, 47.7316704, -87.5928345, 47.7316704, -135.3244934, 135.3244934
34: -74.3134232, 31.5479107, -74.3134232, 31.5479107, -105.8613281, 105.8613281
35: -72.0159607, 37.8115234, -72.0159607, 37.8115234, -109.8274689, 109.8274841
36: -75.3385239, 38.7858200, -75.3385239, 38.7858200, -114.1243439, 114.1243362
37: -106.4644547, 30.5911770, -106.4644547, 30.5911770, -137.0556335, 137.0556335
38: -88.1569672, 41.6150589, -88.1569672, 41.6150589, -129.7720337, 129.7720184
39: -98.3843613, 45.1909943, -98.3843613, 45.1909943, -143.5753326, 143.5753479
40: -81.9499817, 30.8376102, -81.9499817, 30.8376102, -112.7875900, 112.7875900
41: -69.2990265, 43.2153130, -69.2990265, 43.2153130, -112.5143280, 112.5143356
42: -50.8413391, 35.0825958, -50.8413391, 35.0825958, -85.9239349, 85.9239349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=412, inp2_unstable=412, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1537

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 941

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0488579, upper bound: 64.0518486
time: 98.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0540604, upper bound: 64.0517925
time: 78.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -74.4441681, 51.2501221, -74.4441681, 51.2501221, -125.6942902, 125.6942825
1: -35.7569427, 43.2586746, -35.7569427, 43.2586746, -79.0156097, 79.0156097
2: -30.5907345, 44.9403458, -30.5907345, 44.9403458, -75.5310822, 75.5310822
3: -35.0810623, 49.6928406, -35.0810623, 49.6928406, -84.7739029, 84.7739029
4: -40.9552193, 51.8817978, -40.9552193, 51.8817978, -92.8370209, 92.8370132
5: -34.2261429, 47.9517632, -34.2261429, 47.9517632, -82.1779022, 82.1779022
6: -70.8315430, 41.0467148, -70.8315430, 41.0467148, -111.8782501, 111.8782578
7: -43.1463509, 45.9498024, -43.1463509, 45.9498024, -89.0961456, 89.0961533
8: -48.8081589, 63.1175346, -48.8081589, 63.1175346, -111.9256897, 111.9256897
9: -41.9876938, 45.6585083, -41.9876938, 45.6585083, -87.6461945, 87.6462021
10: -63.0807076, 57.5979233, -63.0807076, 57.5979233, -120.6786346, 120.6786270
11: -60.7652092, 33.4686699, -60.7652092, 33.4686699, -94.2338791, 94.2338791
12: -67.9794083, 38.1531181, -67.9794083, 38.1531181, -106.1325226, 106.1325226
13: -65.2255249, 61.3555145, -65.2255249, 61.3555145, -126.5810394, 126.5810394
14: -101.7791443, 45.9462204, -101.7791443, 45.9462204, -147.7253571, 147.7253723
15: -49.4616508, 45.3876305, -49.4616508, 45.3876305, -94.8492737, 94.8492813
16: -63.1231842, 43.5457993, -63.1231842, 43.5457993, -106.6689758, 106.6689758
17: -97.0565567, 40.5237808, -97.0565567, 40.5237808, -137.5803375, 137.5803223
18: -64.3055725, 40.1181068, -64.3055725, 40.1181068, -104.4236755, 104.4236755
19: -46.9785995, 28.0002918, -46.9785995, 28.0002918, -74.9788895, 74.9788895
20: -45.3601379, 29.8137321, -45.3601379, 29.8137321, -75.1738663, 75.1738663
21: -59.0123825, 31.9141216, -59.0123825, 31.9141216, -90.9265060, 90.9265060
22: -58.7619171, 34.4748764, -58.7619171, 34.4748764, -93.2367859, 93.2367935
23: -46.6954193, 36.9088135, -46.6954193, 36.9088135, -83.6042328, 83.6042328
24: -57.0340157, 37.7480583, -57.0340157, 37.7480583, -94.7820740, 94.7820740
25: -50.1265450, 38.0982132, -50.1265450, 38.0982132, -88.2247543, 88.2247620
26: -70.8370209, 45.6955872, -70.8370209, 45.6955872, -116.5326080, 116.5326080
27: -57.3656578, 39.1326370, -57.3656578, 39.1326370, -96.4982910, 96.4982910
28: -46.4588852, 38.8290672, -46.4588852, 38.8290672, -85.2879486, 85.2879486
29: -60.1362267, 32.4817734, -60.1362267, 32.4817734, -92.6179962, 92.6179962
30: -58.1264915, 41.4087524, -58.1264915, 41.4087524, -99.5352478, 99.5352402
31: -62.8394241, 38.8694992, -62.8394241, 38.8694992, -101.7089233, 101.7089157
32: -68.7925339, 35.4185715, -68.7925339, 35.4185715, -104.2111053, 104.2111053
33: -87.5928345, 47.7316704, -87.5928345, 47.7316704, -135.3244934, 135.3244934
34: -74.3134232, 31.5479107, -74.3134232, 31.5479107, -105.8613281, 105.8613281
35: -72.0159607, 37.8115234, -72.0159607, 37.8115234, -109.8274689, 109.8274841
36: -75.3385239, 38.7858200, -75.3385239, 38.7858200, -114.1243439, 114.1243362
37: -106.4644547, 30.5911770, -106.4644547, 30.5911770, -137.0556335, 137.0556335
38: -88.1569672, 41.6150589, -88.1569672, 41.6150589, -129.7720337, 129.7720184
39: -98.3843613, 45.1909943, -98.3843613, 45.1909943, -143.5753326, 143.5753479
40: -81.9499817, 30.8376102, -81.9499817, 30.8376102, -112.7875900, 112.7875900
41: -69.2990265, 43.2153130, -69.2990265, 43.2153130, -112.5143280, 112.5143356
42: -50.8413391, 35.0825958, -50.8413391, 35.0825958, -85.9239349, 85.9239349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=412, inp2_unstable=412, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 872
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 521
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 1548
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 748

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 626

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0326864, upper bound: 64.0560332
time: 91.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0479037, upper bound: 64.0408014
time: 271.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -74.4441681, 51.2501221, -74.4441681, 51.2501221, -125.6942902, 125.6942825
1: -35.7569427, 43.2586746, -35.7569427, 43.2586746, -79.0156097, 79.0156097
2: -30.5907345, 44.9403458, -30.5907345, 44.9403458, -75.5310822, 75.5310822
3: -35.0810623, 49.6928406, -35.0810623, 49.6928406, -84.7739029, 84.7739029
4: -40.9552193, 51.8817978, -40.9552193, 51.8817978, -92.8370209, 92.8370132
5: -34.2261429, 47.9517632, -34.2261429, 47.9517632, -82.1779022, 82.1779022
6: -70.8315430, 41.0467148, -70.8315430, 41.0467148, -111.8782501, 111.8782578
7: -43.1463509, 45.9498024, -43.1463509, 45.9498024, -89.0961456, 89.0961533
8: -48.8081589, 63.1175346, -48.8081589, 63.1175346, -111.9256897, 111.9256897
9: -41.9876938, 45.6585083, -41.9876938, 45.6585083, -87.6461945, 87.6462021
10: -63.0807076, 57.5979233, -63.0807076, 57.5979233, -120.6786346, 120.6786270
11: -60.7652092, 33.4686699, -60.7652092, 33.4686699, -94.2338791, 94.2338791
12: -67.9794083, 38.1531181, -67.9794083, 38.1531181, -106.1325226, 106.1325226
13: -65.2255249, 61.3555145, -65.2255249, 61.3555145, -126.5810394, 126.5810394
14: -101.7791443, 45.9462204, -101.7791443, 45.9462204, -147.7253571, 147.7253723
15: -49.4616508, 45.3876305, -49.4616508, 45.3876305, -94.8492737, 94.8492813
16: -63.1231842, 43.5457993, -63.1231842, 43.5457993, -106.6689758, 106.6689758
17: -97.0565567, 40.5237808, -97.0565567, 40.5237808, -137.5803375, 137.5803223
18: -64.3055725, 40.1181068, -64.3055725, 40.1181068, -104.4236755, 104.4236755
19: -46.9785995, 28.0002918, -46.9785995, 28.0002918, -74.9788895, 74.9788895
20: -45.3601379, 29.8137321, -45.3601379, 29.8137321, -75.1738663, 75.1738663
21: -59.0123825, 31.9141216, -59.0123825, 31.9141216, -90.9265060, 90.9265060
22: -58.7619171, 34.4748764, -58.7619171, 34.4748764, -93.2367859, 93.2367935
23: -46.6954193, 36.9088135, -46.6954193, 36.9088135, -83.6042328, 83.6042328
24: -57.0340157, 37.7480583, -57.0340157, 37.7480583, -94.7820740, 94.7820740
25: -50.1265450, 38.0982132, -50.1265450, 38.0982132, -88.2247543, 88.2247620
26: -70.8370209, 45.6955872, -70.8370209, 45.6955872, -116.5326080, 116.5326080
27: -57.3656578, 39.1326370, -57.3656578, 39.1326370, -96.4982910, 96.4982910
28: -46.4588852, 38.8290672, -46.4588852, 38.8290672, -85.2879486, 85.2879486
29: -60.1362267, 32.4817734, -60.1362267, 32.4817734, -92.6179962, 92.6179962
30: -58.1264915, 41.4087524, -58.1264915, 41.4087524, -99.5352478, 99.5352402
31: -62.8394241, 38.8694992, -62.8394241, 38.8694992, -101.7089233, 101.7089157
32: -68.7925339, 35.4185715, -68.7925339, 35.4185715, -104.2111053, 104.2111053
33: -87.5928345, 47.7316704, -87.5928345, 47.7316704, -135.3244934, 135.3244934
34: -74.3134232, 31.5479107, -74.3134232, 31.5479107, -105.8613281, 105.8613281
35: -72.0159607, 37.8115234, -72.0159607, 37.8115234, -109.8274689, 109.8274841
36: -75.3385239, 38.7858200, -75.3385239, 38.7858200, -114.1243439, 114.1243362
37: -106.4644547, 30.5911770, -106.4644547, 30.5911770, -137.0556335, 137.0556335
38: -88.1569672, 41.6150589, -88.1569672, 41.6150589, -129.7720337, 129.7720184
39: -98.3843613, 45.1909943, -98.3843613, 45.1909943, -143.5753326, 143.5753479
40: -81.9499817, 30.8376102, -81.9499817, 30.8376102, -112.7875900, 112.7875900
41: -69.2990265, 43.2153130, -69.2990265, 43.2153130, -112.5143280, 112.5143356
42: -50.8413391, 35.0825958, -50.8413391, 35.0825958, -85.9239349, 85.9239349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Conv2D, total=2048, inp1_unstable=412, inp2_unstable=412, delta_unstable=2045
- layer_idx=2, type=LayerType.Conv2D, total=1024, inp1_unstable=563, inp2_unstable=563, delta_unstable=1024
- layer_idx=4, type=LayerType.Linear, total=50, inp1_unstable=27, inp2_unstable=27, delta_unstable=45
- layer_idx=6, type=LayerType.Linear, total=43, inp1_unstable=43, inp2_unstable=43, delta_unstable=43

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1641
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 622
type: RSZ, layer: 1, pos: 1596
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 984
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1002
type: RSZ, layer: 1, pos: 997
type: RSZ, layer: 1, pos: 1017
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 1672
type: RSZ, layer: 1, pos: 1785
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 649
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 595
type: RSZ, layer: 1, pos: 1419
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 839
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1762
type: RSZ, layer: 1, pos: 841
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 551
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 871
type: RSZ, layer: 1, pos: 554
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 638
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 654
type: RSZ, layer: 1, pos: 1657
type: RSZ, layer: 1, pos: 932
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 680
type: RSZ, layer: 1, pos: 858
type: RSZ, layer: 1, pos: 1566
type: RSZ, layer: 1, pos: 1015
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1018
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 1665
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 559
type: RSZ, layer: 1, pos: 526
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 1385
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 607
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 1746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 658
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 687
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 567
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 885
type: RSZ, layer: 1, pos: 569
type: RSZ, layer: 1, pos: 1633
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 965
type: RSZ, layer: 1, pos: 1611
type: RSZ, layer: 1, pos: 1595
type: RSZ, layer: 1, pos: 1615
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 541
type: RSZ, layer: 1, pos: 609
type: RSZ, layer: 1, pos: 1628
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 626
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1780
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 1686
type: RSZ, layer: 1, pos: 1583
type: RSZ, layer: 1, pos: 1682
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 1464
type: RSZ, layer: 1, pos: 707
type: RSZ, layer: 1, pos: 842
type: RSZ, layer: 1, pos: 1698
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1761
type: RSZ, layer: 1, pos: 1777
type: RSZ, layer: 1, pos: 587
type: RSZ, layer: 1, pos: 1620
type: RSZ, layer: 1, pos: 1635
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 986
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 590
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1613
type: RSZ, layer: 1, pos: 608
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1713
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 657
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 553
type: RSZ, layer: 1, pos: 892
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 985
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 1567
type: RSZ, layer: 1, pos: 572
type: RSZ, layer: 1, pos: 573
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 916
type: RSZ, layer: 1, pos: 674
type: RSZ, layer: 1, pos: 1016
type: RSZ, layer: 1, pos: 1604
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 571
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 1546
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 1649
type: RSZ, layer: 1, pos: 840
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 606
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1669
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 853
type: RSZ, layer: 1, pos: 527
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 1765
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 855
type: RSZ, layer: 1, pos: 1656
type: RSZ, layer: 1, pos: 1701
type: RSZ, layer: 1, pos: 558
type: RSZ, layer: 1, pos: 589
type: RSZ, layer: 1, pos: 523
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 966
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 874
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 1629
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 1012
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 1565
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1019
type: RSZ, layer: 1, pos: 1580
type: RSZ, layer: 1, pos: 604
type: RSZ, layer: 1, pos: 648
type: RSZ, layer: 1, pos: 999
type: RSZ, layer: 1, pos: 1612
type: RSZ, layer: 1, pos: 1791
type: RSZ, layer: 1, pos: 596
type: RSZ, layer: 1, pos: 1014
type: RSZ, layer: 1, pos: 679
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 550
type: RSZ, layer: 1, pos: 1599
type: RSZ, layer: 1, pos: 1592
type: RSZ, layer: 1, pos: 884
type: RSZ, layer: 1, pos: 641
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 552
type: RSZ, layer: 1, pos: 1593
type: RSZ, layer: 1, pos: 621
type: RSZ, layer: 1, pos: 513
type: RSZ, layer: 1, pos: 1000
type: RSZ, layer: 1, pos: 1614
type: RSZ, layer: 1, pos: 1435
type: RSZ, layer: 1, pos: 1654
type: RSZ, layer: 1, pos: 543
type: RSZ, layer: 1, pos: 1563
type: RSZ, layer: 1, pos: 1551
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 611
type: RSZ, layer: 1, pos: 1764
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 857
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1697
type: RSZ, layer: 1, pos: 1403
type: RSZ, layer: 1, pos: 610
type: RSZ, layer: 1, pos: 1466
type: RSZ, layer: 1, pos: 566
type: RSZ, layer: 1, pos: 564
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 1631
type: RSZ, layer: 1, pos: 517
type: RSZ, layer: 1, pos: 1670
type: RSZ, layer: 1, pos: 664
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1745
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 574
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 901
type: RSZ, layer: 1, pos: 1730
type: RSZ, layer: 1, pos: 1590
type: RSZ, layer: 1, pos: 1666
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 522
type: RSZ, layer: 1, pos: 575
type: RSZ, layer: 1, pos: 557
type: RSZ, layer: 1, pos: 875
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 593
type: RSZ, layer: 1, pos: 519
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 1003
type: RSZ, layer: 1, pos: 556
type: RSZ, layer: 1, pos: 642
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 537
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 1549
type: RSZ, layer: 1, pos: 539
type: RSZ, layer: 1, pos: 856
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 524
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 1747
type: RSZ, layer: 1, pos: 582
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 565
type: RSZ, layer: 1, pos: 868
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 1640
type: RSZ, layer: 1, pos: 1001
type: RSZ, layer: 1, pos: 1681
type: RSZ, layer: 1, pos: 1591
type: RSZ, layer: 1, pos: 516
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 525
type: RSZ, layer: 1, pos: 1705
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 542
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 1790
type: RSZ, layer: 1, pos: 1606
type: RSZ, layer: 1, pos: 670
type: RSZ, layer: 1, pos: 663
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 1597
type: RSZ, layer: 1, pos: 873
type: RSZ, layer: 1, pos: 870
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 1702
type: RSZ, layer: 1, pos: 1550
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 536
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 555
type: RSZ, layer: 1, pos: 1763
type: RSZ, layer: 1, pos: 1562
type: RSZ, layer: 1, pos: 1598
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1778
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1685
type: RSZ, layer: 1, pos: 1729
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 588
type: RSZ, layer: 1, pos: 1579
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 917
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 540
type: RSZ, layer: 1, pos: 1545
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 970
type: RSZ, layer: 1, pos: 998
type: RSZ, layer: 1, pos: 1547
type: RSZ, layer: 1, pos: 639
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1013
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 690
type: RSZ, layer: 1, pos: 580
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 1714
type: RSZ, layer: 1, pos: 869
type: RSZ, layer: 1, pos: 1581
type: RSZ, layer: 1, pos: 1582
type: RSZ, layer: 1, pos: 950
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1564
type: RSZ, layer: 1, pos: 719
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 854
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 594
type: RSZ, layer: 1, pos: 647
type: RSZ, layer: 1, pos: 1659
type: RSZ, layer: 1, pos: 665
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 623
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 538
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1605
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1467
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 1465
type: RSZ, layer: 1, pos: 625
type: RSZ, layer: 1, pos: 605
type: RSZ, layer: 1, pos: 983
type: RSZ, layer: 1, pos: 1594
type: RSZ, layer: 1, pos: 724
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 581
type: RSZ, layer: 1, pos: 570
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 838
type: RSZ, layer: 1, pos: 591
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 579
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1020
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 521

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1641

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0353853, upper bound: 64.0384120
time: 106.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -64.0353853, upper bound: 64.0384120
time: 142.52 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 250.98 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 250.98
Output dim: 8, lower bound: -63.9911935, upper bound: 64.0536370
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 250.98
Output dim: 8, lower bound: -64.0147409, upper bound: 64.0316461
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 250.98
Output dim: 8, lower bound: -64.0534827, upper bound: 63.9959682
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 250.98
Output dim: 8, lower bound: -64.0352234, upper bound: 64.0141974
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 250.98
Output dim: 8, lower bound: -64.0488579, upper bound: 64.0518486
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 250.98
Output dim: 8, lower bound: -64.0540604, upper bound: 64.0517925
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 250.98
Output dim: 8, lower bound: -64.0326864, upper bound: 64.0560332
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 250.98
Output dim: 8, lower bound: -64.0479037, upper bound: 64.0408014
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 250.98
Output dim: 8, lower bound: -64.0353853, upper bound: 64.0384120
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 250.98
Output dim: 8, lower bound: -64.0353853, upper bound: 64.0384120
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 250.98
Output dim: 8, lower bound: -64.0538760, upper bound: 64.0572269
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 250.98
Output dim: 8, lower bound: -64.0514104, upper bound: 64.0540362
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 250.98
Output dim: 8, lower bound: -64.0514104, upper bound: 64.0482994

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 148.12 + 3580.34 = 3728.46 seconds
