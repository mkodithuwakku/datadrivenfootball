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

# For each season and team
for season in seasons:
    for team in teams:
        # Alter source url based on current season & team
        url = 'https://www.pro-football-reference.com/teams/' + team + '/' + season + '/gamelog/'
        print(url)

        # Get offensive and defensive stats, combine, add season & team
        off_df = pd.read_html(url, header = 1, attrs = {'id':'gamelog' + season})[0]
        def_df = pd.read_html(url, header = 1, attrs = {'id':'gamelog_opp' + season})[0]
        team_df = pd.concat([off_df, def_df], axis = 1)
        team_df.insert(loc = 0, column = 'Season', value = season)
        team_df.insert(loc = 2, column = 'Team', value = team.upper())

        # Concat to main dataframe
        df = pd.concat([df, team_df], ignore_index = True)

        # Pause program to bypass request limit
        time.sleep(random.randint(7,8))

print(df)

# Save to csv and reload
df.to_csv('nfl_gamelogs_2014-2024.csv', index=False)
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

# Scrape Vegas lines for each season and team
vegas_df = pd.DataFrame()
for season in seasons:
    for team in teams:
        url = 'https://www.pro-football-reference.com/teams/' + team + '/' + season + '_lines.htm'
        lines_df = pd.read_html(url, header = 0, attrs={'id':'vegas_lines'})[0]
        lines_df.insert(loc=0, column='Season', value=season)
        lines_df.insert(loc=2, column='Team', value=team.upper())
        vegas_df = pd.concat([vegas_df, lines_df], ignore_index=True)
        time.sleep(random.randint(7,8))

# Export the csv
print(vegas_df)
vegas_df.to_csv('nfl_vegas_lines_2014-2024.csv', index=False)
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

        
        






