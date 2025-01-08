import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import math


# Get names of all quaterback data files
all_files = glob.glob("/Users/mkodi/Personal Coding/DDF/Passer Rating Comparison/Code/datadrivenfootball/qbSeasons" + "/*.xlsx")
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

# Create new table where player is the index, and each column is a respective season and that player's respect rating+
# With the final column being that players average rating+ throughout a career.
pivot_df = df.pivot_table(index="Player", columns="Season", values="Rating+",aggfunc="first")

# Column for the Average rating+ throughout the player's career
pivot_df["Avg Rating+"] = pivot_df.mean(axis=1, skipna=True)

# Column for cumulative rating+ throughout the player's career - #of years played (lower than average seasons are negative)

seasons_only = pivot_df.drop(columns=["Avg Rating+"], errors="ignore") # Drop average rating column
pivot_df["Career Rating+"] = seasons_only.sum(axis=1, skipna=True) - seasons_only.count(axis=1)

# Add a new row to display the season averages
pivot_df.loc["Average Rating"] = pivot_df.mean(numeric_only=True, skipna=True)

#-------------PLOTTING---------------
plot_df = pivot_df

# Identify top 10 players for "Avg Rating+" and "Career Rating+"
top10_avg = plot_df["Avg Rating+"].nlargest(10).index
top10_career = plot_df["Career Rating+"].nlargest(10).index

# Drop non-season columns so only actual seasons remain
plot_seasons_df = plot_df.drop(columns=["Avg Rating+", "Career Rating+"], errors="ignore")

# Transpose so each column is a player (or "Average Rating"), each row is a season
transposed = plot_seasons_df.transpose()
##############################################################################
# 1) Convert the transposed index to integers if your seasons are strings.
#    This lets us create integer-based x-axis ticks for each year.
##############################################################################
# If your Season column is already numeric, you can skip this step.
transposed.index = transposed.index.astype(int)

# 2) Determine min/max year
min_year = transposed.index.min()
max_year = transposed.index.max()

##############################################################################
# 3) Determine min/max for Rating+ so we can set y-ticks in increments of 0.05
##############################################################################
y_min = transposed.min().min()  # smallest Rating+ for any player/season
y_max = transposed.max().max()  # largest Rating+ for any player/season

# Round them nicely so we start/end on a multiple of 0.05
y_min_rounded = math.floor(y_min * 20) / 20.0  # e.g. 1.03 -> 1.00
y_max_rounded = math.ceil(y_max * 20) / 20.0   # e.g. 1.88 -> 1.90

# Create an array of ticks from y_min_rounded to y_max_rounded in steps of 0.05
y_ticks = np.arange(y_min_rounded, y_max_rounded + 0.05, 0.05)

plt.figure(figsize=(14, 8))  # Make the figure larger for clarity

for player in transposed.columns:
    data = transposed[player]
    
    # Determine color based on membership in top-10 or if "Average Rating"
    if player == "Average Rating":
        color = "yellow"
    elif player in top10_avg and player in top10_career:
        color = "magenta"  # top-10 in both
    elif player in top10_avg:
        color = "green"
    elif player in top10_career:
        color = "blue"
    else:
        color = "gray"
    
    # Plot with markers at each data point
    plt.plot(
        data.index, 
        data.values, 
        marker="o", 
        color=color,
        linewidth=1
    )
    
    # Label the final data point if highlighted (not gray)
    if color != "gray":
        last_x = data.index[-1]
        last_y = data.values[-1]
        plt.text(
            x=last_x + 0.1,  # small shift to the right
            y=last_y, 
            s=player, 
            color=color,
            fontsize=9,
            va="center",
            ha="left"
        )

##############################################################################
# 4) Set the x-axis ticks for every year in [min_year, max_year]
##############################################################################
all_years = range(min_year, max_year + 1)
plt.xticks(all_years, all_years)

##############################################################################
# 5) Set the y-axis ticks in increments of 0.05 with large spacing
##############################################################################
plt.yticks(y_ticks)

plt.title("Quarterback Rating+ by Season")
plt.xlabel("Season")
plt.ylabel("Rating+")
plt.grid(axis="y", linestyle="--", alpha=0.5)  # optional: horizontal grid lines
plt.tight_layout()
plt.show()

pivot_df.to_excel("Pivot_Quarterback_Stats.xlsx", sheet_name="PivotData")
transposed.to_excel("Transposed_Quarterback_Stats.xlsx", sheet_name="TransposedData")