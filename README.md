# football-prediction


---


### Current Issues
- For some matches there are no odds neither in the old database (table 'Match'), nor in the recent DB (table 'Odds'). The reason is likely to be in the loading/preprocessing of the data. One potential solution is to skip the old DB entirely, rename the raw data folders to country names (e.g. 'Germany') and then load the odds through matching of team names directly into the 'Odds' table.
- When visualizing the comparison of betting returns, in some cases an error occurs for the average plot.
- France 19/20 missing matchdays and partially complete matchdays

### Future Work
- Scraping
  - Scrape more data on existing structure
  - Add manager data as TeamManager table with start and end dates of stints
  - Add games stats such as shots, possession etc. from game details
- Modeling
  - For neural network, test if it works to pretrain with entire dataset and then retrain for each league.
  - Add features: % Attendance, Team rating based on different attributes (goals, shots, see paper), incorporate opponent strength of prior matches into team form
  - Properly preprocess computed features, especially standard scale
  - Grid/random search valid feature combinations
  - Do research on sports prediction modeling techniques