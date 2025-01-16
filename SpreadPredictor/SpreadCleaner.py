import pandas as pd

# Everything above this line is commented out to skip unnecessary reprocessing
# Retaining necessary read_csv lines to load data from previous steps

# Load the game logs data
df = pd.read_csv('nfl_gamelogs_2014-2024.csv')
print(df.shape)

# Process the game logs data
nfl_df = df.drop(df.columns[12:], axis=1)  # Drop unnecessary columns
nfl_df = nfl_df.drop(nfl_df.columns[5:6], axis=1)
print(nfl_df.shape)

# Rename columns for clarity
col_names = {'Unnamed: 4': 'Win', 'Unnamed: 6': 'Home', 'Tm': 'Off_Pts', 'Opp.1': 'Def_Pts'}
nfl_df = nfl_df.rename(columns=col_names)
print(nfl_df.info(verbose=True))

# Load the Vegas lines data
vegas_df = pd.read_csv('nfl_vegas_lines_2014-2024.csv')
print(vegas_df.shape)

# Drop unnecessary columns from Vegas lines
vegas_df = vegas_df.drop(vegas_df.columns[6:], axis=1)
print(vegas_df.info(verbose=True))

# Rename columns in Vegas lines
col_names = {'G#': 'G', 'Over/Under': 'Total'}
vegas_df = vegas_df.rename(columns=col_names)
print(vegas_df.info(verbose=True))

# Drop playoff games
vegas_df = vegas_df.query('(Season <= 2020 and G < 17) or (Season >= 2021 and G < 18)')
print(vegas_df.shape)

# Create Home column based on Opp column
vegas_df['Home'] = vegas_df['Opp'].apply(lambda x: 0 if x[0] == '@' else 1)

# Remove @ symbol from the 'Opp' column
vegas_df['Opp'] = vegas_df['Opp'].apply(lambda x: x[1:] if x[0] == '@' else x)

# Replace incorrect abbreviations
abbr_dict = {'OAK': 'RAI', 'LVR': 'RAI', 'STL': 'RAM', 'LAR': 'RAM', 'LAC': 'SDG',
             'IND': 'CLT', 'HOU': 'HTX', 'BAL': 'RAV', 'ARI': 'CRD', 'TEN': 'OTI'}
vegas_df = vegas_df.replace({'Opp': abbr_dict})

# Ensure 'Home' columns in both DataFrames are of the same type
nfl_df['Home'] = nfl_df['Home'].astype(int)  # Convert to integer
vegas_df['Home'] = vegas_df['Home'].astype(int)  # Convert to integer

# Merge the data sets
merged_df = pd.merge(nfl_df, vegas_df, on=['Season', 'Team', 'Opp', 'Home'])

# Print example season
print(nfl_df.query('Season == 2014 and Team == "BUF"'))
print(vegas_df.query('Season == 2014 and Team == "BUF"'))
print(merged_df.query('Season == 2014 and Team == "BUF"'))

# Save the final merged data
merged_df.to_csv('nfl_pts_and_vegas_2014-2024.csv', index=False)
