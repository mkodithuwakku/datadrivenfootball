import pandas as pd
import glob
import matplotlib.pyplot as plt


# Get names of all quaterback data files
all_files = glob.glob("/Users/mkodi/Personal Coding/DDF/Passer Rating Comparison/Code/datadrivenfootball/qbSeasons" + "/*.xls")
df_list = []

# Convert files to data frames and add to list
for file in all_files:
        print(file)
        df_list.append(pd.read_excel(file, header = 1))

# Concatenate dataframe list into single dataframe (ignoring existing index, in this case the passer rating rank)
df = pd.concat(df_list, ignore_index=True)

# Calculate average Rate per season, returns an array same length as df with each season averge for each entry
season_avg = df.groupby("Season")["Rate"].transform("mean")
    
# Create the 'Rating+' column
df["Rating+"] = df["Rate"] / season_avg