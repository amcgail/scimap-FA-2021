# Map of Science

This repository contains code and data for the FA 2021 Map of Science project.
We are aiming to depict the complex structure and history of scientific publication, and of mentorship.
This is a collaborative effort between data scientists and designers at UT Austin.

***

### Mentorship updates

Visual notations of the mentorship tree:

<div>
  <img src="https://github.com/JialingJia/scimap-FA-2021/blob/main/mentorship/tree_annotation.png" width = "400"><br><br>
</div>

#### Prototypical trees:
- the widest tree: (9.513573121515911, 'FRANCIS GALTON')
- the tallest tree: (9.442344726411353, 'FRANCIS GALTON')
- the most female-tilted tree: (60.61795796428965, 'DONALD REIFF')
- the most male-tilted tree: (-19.1015703851541, 'SEBALD RAU')
- the most curly tree: (0.1975998995944304, 'CHRISTIAN GOTTFRIED DANIEL NEES VON ESENBECK')
- the most female-tilted field: (79.85674230061032, 'nursing')
- the most male-tilted field: (-76.94368361544785, 'neuroscience')

*The most curly tree is found by the largest std of the tree curvature.*
*The most female/male-tilted field is found based on the average radius of trees among the selected fields.*

#### Data:
- [prototypical trees](https://github.com/JialingJia/scimap-FA-2021/tree/main/mentorship/data/proto)
- [visual metrics](https://github.com/JialingJia/scimap-FA-2021/tree/main/mentorship/visual_metrics)
- [mentorlist](https://github.com/JialingJia/scimap-FA-2021/blob/main/mentorship/data/mentoradjacentlist.zip)
- [menteelist](https://github.com/JialingJia/scimap-FA-2021/blob/main/mentorship/data/menteeadjacentlist.zip)
- [researcher](https://github.com/JialingJia/scimap-FA-2021/blob/main/mentorship/data/researcherGender1.zip)

Original dataset paper: Ke, Qing, Lizhen Liang, Ying Ding, Stephen V. David, and Daniel E. Acuna. **A dataset of mentorship in science with semantic and demographic estimations.** *arXiv preprint arXiv:2106.06487 (2021)* [Raw data](https://zenodo.org/record/4917086)

***

#### Forest:
<div align='center'>
  <img src="https://github.com/JialingJia/scimap-FA-2021/blob/main/mentorship/visualization/6_discipline.png" width = "800"><br><br>
</div>

***

# Collaboration

To contribute, fork the repository into your own GitHub, make changes, and submit a pull request.
Corresponding author: Alec McGail // am2873@cornell.edu
