import numpy as mp
import pandas as pd
import random
import time

# List of seasons to scrape
seasons = [str(season) for season in range(2014,2024)]
print(f'number of seasons = {len(seasons)}')

# List of teams
teams = ['crd', 'atl', 'rav', 'buf', 'car', 'chi', 'cin', 'cle', 'dal', 
         'gnb', 'htx', 'clt', 'jax', 'kan', 'sdg', 'ram', 'rai', 'mia', 
         'min', 'nwe', 'nor', 'nyg', 'nyj', 'phi', 'pit', 'sea', 'sfo', 
          'tam', 'oti', 'was', 'den', 'det']

# Create dataframe
df = pd.DataFrame()

df = pd.read_csv('nfl_gamelogs_2014-2024.csv')
print(df.shape)

# Drop unneccesary columns
nfl_df = df.drop(df.columns[12:], axis = 1)
nfl_df = nfl_df.drop(nfl_df.columns[5:6], axis=1)
print(nfl_df.shape)

# Rename columns for clarity
col_names = {'Unnamed: 4':'Win', 'Unnamed: 6':'Home', 'Tm':'Off_Pts', 'Opp.1':'Def_Pts'}
nfl_df = nfl_df.rename(columns=col_names)
print(nfl_df.info(verbose=True))

# Map 'Opp' to 3 - letter abbreviations
team_dict = {'Arizona Cardinals':'CRD', 'Atlanta Falcons':'ATL', 'Baltimore Ravens':'RAV',
             'Buffalo Bills':'BUF', 'Carolina Panthers':'CAR','Chicago Bears':'CHI',
             'Cincinnati Bengals':'CIN','Cleveland Browns':'CLE', 'Dallas Cowboys':'DAL',
             'Denver Broncos':'DEN', 'Detroit Lions':'DET', 'Green Bay Packers':'GNB',
             'Houston Texans':'HTX', 'Indianapolis Colts':'CLT', 'Jacksonville Jaguars':'JAX',
             'Kansas City Chiefs':'KAN', 'Los Angeles Chargers':'SDG', 'Los Angeles Rams':'RAM',
             'Las Vegas Raiders':'RAI','Oakland Raiders':'RAI', 'Miami Dolphins':'MIA', 
             'Minnesota Vikings':'MIN', 'New England Patriots':'NWE', 'New Orleans Saints':'NOR',
             'New York Giants':'NYG', 'New York Jets':'NYJ', 'Philadelphia Eagles':'PHI',
             'Pittsburgh Steelers':'PIT', 'St. Louis Rams':'RAM', 'San Diego Chargers':'SDG',
             'San Francisco 49ers':'SFO', 'Seattle Seahawks':'SEA', 'Tampa Bay Buccaneers':'TAM',
             'Tennessee Titans':'OTI', 'Washington Commanders':'WAS', 'Washington Football Team':'WAS',
             'Washington Redskins':'WAS'} 
nfl_df = nfl_df.replace({'Opp':team_dict})

# Convert Wins, OT, and Home to Binary
nfl_df['Win'] = nfl_df['Win'].apply(lambda x: 1 if x =='W' else 0)
nfl_df['OT'] = nfl_df['OT'].apply(lambda x: 1 if x =='OT' else 0)
nfl_df['Home'] = nfl_df['Home'].apply(lambda x: 0 if x =='@' else 1)
print(nfl_df)

vegas_df = pd.read_csv('nfl_vegas_lines_2014-2024.csv')

print(vegas_df.shape)
# Drop unnecessary columns
vegas_df = vegas_df.drop(vegas_df.columns[6:], axis=1)
print(vegas_df.info(verbose=True))

# Rename columns
col_names = {'G#':'G', 'Over/Under':'Total'}
vegas_df = vegas_df.rename(columns=col_names)
print(vegas_df.info(verbose=True))

# Drop playoff games
vegas_df = vegas_df.query('(Season <= 2020 and G < 17) or (Season >= 2021 and G < 18)')
print(vegas_df.shape)

# Create Home column based on Opp column
vegas_df['Home'] = vegas_df['Opp'].apply(lambda x: 0 if x[0] == '@' else 1)

# Remove @ symbol
vegas_df['Opp'] = vegas_df['Opp'].apply(lambda x: x[1:] if x[0] == '@' else x)

# Replace incorrect abbreviations
abbr_dict = {'OAK':'RAI', 'LVR':'RAI', 'STL':'RAM', 'LAR':'RAM', 'LAC':'SDG',
             'IND':'CLT', 'HOU':'HTX', 'BAL':'RAV', 'ARI':'CRD', 'TEN':'OTI'}
vegas_df = vegas_df.replace({'Opp':abbr_dict})

print(nfl_df.shape)
print(vegas_df.shape)

# Merge the data sets
merged_df = pd.merge(nfl_df, vegas_df, on=['Season', 'Team', 'Opp', 'Home'])

# Print example season
print(nfl_df.query('Season == 2014 and Team == "BUF"'))
print(vegas_df.query('Season == 2014 and Team == "BUF"'))
print(merged_df.query('Season == 2014 and Team == "BUF"'))

merged_df.to_csv('nfl_pts_and_vegas_2014-2024.csv', index=False)

        
        






