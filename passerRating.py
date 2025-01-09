import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.lines as mlines

# Dark Mode
plt.style.use("dark_background")

# Get names of all quarterback data files
all_files = glob.glob("/Users/mkodi/Personal Coding/DDF/Passer Rating Comparison/Code/datadrivenfootball/qbSeasons/*.xlsx")
df_list = []

# Append excels to dataframe list
for file in all_files:
    #print(file)
    df_list.append(pd.read_excel(file, header=1))

# Combine all datagrames in list
df = pd.concat(df_list, ignore_index=True)

# Calculate average Rate per season
season_avg = df.groupby("Season")["Rate"].transform("mean")

# Create the 'Rating+' column
df["Rating+"] = df["Rate"] / season_avg

# Create pivot: Player as index, Season as columns, values = Rating+
pivot_df = df.pivot_table(
    index="Player", columns="Season", values="Rating+", aggfunc="first"
)

# Column for the average Rating+ over the player's career
pivot_df["Avg Rating+"] = pivot_df.mean(axis=1, skipna=True)

# Column for “Career Rating+”: sum of all seasons minus number of seasons
seasons_only = pivot_df.drop(columns=["Avg Rating+"], errors="ignore")
pivot_df["Career Rating+"] = (
    seasons_only.sum(axis=1, skipna=True) - seasons_only.count(axis=1)
)

# Add a row for the overall average across all players
pivot_df.loc["Average Rating"] = pivot_df.mean(numeric_only=True, skipna=True)

# Identify the top 10 players for “Career Rating+”
top10_career = pivot_df["Career Rating+"].nlargest(10).index

# Drop summary columns before plotting (Avg Rating+, Career Rating+)
plot_seasons_df = pivot_df.drop(
    columns=["Avg Rating+", "Career Rating+"], errors="ignore"
)

# Transpose so rows = seasons, columns = players
transposed = plot_seasons_df.transpose()
transposed.index = transposed.index.astype(int)

min_year = transposed.index.min()
max_year = transposed.index.max()

# Determine y-axis range in steps of 0.05
y_min = transposed.min().min()
y_max = transposed.max().max()

y_min_rounded = math.floor(y_min * 20) / 20.0
y_max_rounded = math.ceil(y_max * 20) / 20.0
y_ticks = np.arange(y_min_rounded, y_max_rounded + 0.05, 0.05)

# --- PLOT ---
plt.figure(figsize=(45, 30))

for player in transposed.columns:
    # Extract this player’s Rating+ series, dropping NaN
    data = transposed[player].dropna().sort_index()
    
    # Determine color & alpha for highlight vs. non-highlight
    if player == "Average Rating":
        color = "white"         
        alpha = 1.0
    elif player in top10_career:
        color = "orchid"   
        alpha = 1.0
    else:
        color = "lightgray"
        alpha = 0.25
    
    if len(data) == 0:
        continue 

    seasons = list(data.index)
    ratings = list(data.values)
    
    # Plot each point
    plt.scatter(seasons, ratings, color=color, alpha=alpha, marker=".")
    
    # Connect consecutive points
    for i in range(len(seasons) - 1):
        x1, x2 = seasons[i], seasons[i + 1]
        y1, y2 = ratings[i], ratings[i + 1]
        
        # If the difference in seasons is exactly 1, solid line, else dotted
        linestyle = "-" if (x2 - x1) == 1 else "--"
        
        plt.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=1, linestyle=linestyle)

    # If player in top 10 career, label last valid data point
    if player in top10_career:
        last_season = seasons[-1]
        last_rating = ratings[-1]
        plt.text(
            x=last_season + 0.2,
            y=last_rating,
            s=player,
            color=color,
            fontsize=10,
            va="center",
            ha="left",
            weight="bold"
        )

# X-axis: set ticks for each year, rotate to avoid overlap
all_years = range(min_year, max_year + 1)
plt.xticks(all_years, all_years, rotation=45)

# Y-axis: increments of 0.05
plt.yticks(y_ticks)

plt.title("Quarterback Rating+ by Season", color="white")
plt.xlabel("Season", color="white")
plt.ylabel("Rating+", color="white")

# Dark background style also inverts default spines, etc.
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Create Custom Legend
import matplotlib.lines as mlines

non_top10_line = mlines.Line2D(
    [], [], color="lightgray", marker=".", linestyle="-", alpha=0.5, 
    label="Non Top 10 Career"
)

top10_line = mlines.Line2D(
    [], [], color="orchid", marker=".", linestyle="-", 
    label="Top 10 Career"
)

gap_line = mlines.Line2D(
    [], [], color="lightgray", linestyle="--", label="Gap in Seasons"
)

plt.legend(
    handles=[non_top10_line, top10_line, gap_line], 
    loc="lower right",
    fontsize=10,
    facecolor="black",
    edgecolor="white" 
)

plt.show()

# Export if needed
pivot_df.to_excel("Pivot_Quarterback_Stats.xlsx", sheet_name="PivotData")
transposed.to_excel("Transposed_Quarterback_Stats.xlsx", sheet_name="TransposedData")
