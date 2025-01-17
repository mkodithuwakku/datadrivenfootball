import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import scikit-learn machine learning model
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

# Import scikit-learn metrics and preprocessing libraries
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the main dataframe
nfl_df = pd.read_csv('nfl_pts_and_vegas_2014-2024.csv')
print(nfl_df)

# Add True Total, and binary indicators for Over, Under, and Push
nfl_df['True_Total'] = nfl_df['Off_Pts'] + nfl_df['Def_Pts']
nfl_df['Over'] = np.where(nfl_df['True_Total'] > nfl_df['Total'], 1, 0)
nfl_df['Under'] = np.where(nfl_df['True_Total'] < nfl_df['Total'], 1, 0)
nfl_df['Push'] = np.where(nfl_df['True_Total'] == nfl_df['Total'], 1, 0)
print(nfl_df)

for season in range(2021, 2024):
    print(nfl_df.query('Season == @season and Week == 1')['Under'].mean())
# Sort the data by Season, then by Week
nfl_df = nfl_df.sort_values(by=['Season','Week']).reset_index(drop=True)

# Create and Evaluate a Model for NFL Totals (1 = Under, 0 = Over or Push)
# Set the dataframe
df = nfl_df.query('Home == 1').reset_index(drop=True)

# Set the features and the target variable
# Features is what the algorithm uses to predict the future
# target is the variable we want to determine in the future
features = ['Spread','Total']
target = 'Under'


# Iterate over the last three seasons
for season in [2021, 2022, 2023]:
    # Display the season
    print(f'\nResults for {season}:')

    # Initialize the season aggregates
    y_preds = []
    y_trues = []

    # Iterate over the weeks in the season
    for week in range(1, 19):

        # Display the current Week
        print(f' Week {week:>2}:', end=' ')

        # Create training set
        train_df = df.query('Season < @season or (Season == @season and Week < @week)')
        
        # Create testing set
        test_df = df.query('Season == @season and Week == @week and True_Total != Total')

        # Create X_train, y_train, X_test, y_test
        X_train = train_df[features]
        y_train = train_df[target]
        X_test = test_df[features]
        y_test = test_df[target]

        # Set the model
        model = KNeighborsClassifier(n_neighbors=9)

        # Train the model
        clf = model.fit(X_train, y_train)

        # Get the predictions
        y_pred = clf.predict(X_test)

        # Get the true values
        y_true = y_test

        # Display the accuracy score for the current week
        print(f'accuracy score={accuracy_score(y_true, y_pred):.2%}')

        # Update the season aggregates
        y_preds += list(y_pred)
        y_trues += list(y_true)

    # Display the total accuracy score for the current season
    print(f'Season {season}: Total accuracy score={accuracy_score(y_trues, y_preds):.2%}')

    # Display the classification report for the current season
    print(f'\nClassification Report for {season}:')
    print(classification_report(y_trues, y_preds, target_names=['Over','Under']))

    # Display the confusion matrix for the current season
    cm = confusion_matrix(y_trues, y_preds)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Over','Under'])
    display.plot()
    plt.grid(False)
    plt.show()


# Get and display the neighbors for games in Season 2023, Week 18
# Iterate through the test instances and find their nearest neighbors
index = 0
for game_index, spread, total in X_test.itertuples():
    # Get the neighbor distances and indices in the dataframe
    nbr_distance = clf.kneighbors(X_test)[0][index]
    nbr_index = clf.kneighbors(X_test)[1][index]
    index += 1

    # Display the games, the distances, and the neighbors
    print(f'GAME')
    print(df.iloc[[game_index],:][['Season','Week','Team','Opp','Spread','Total','True_Total','Under']])
    print(f'\nNEAREST NEIGHBORS (distances={[round(value, 2) for value in nbr_distance]})')
    print(df.iloc[nbr_index,:][['Season','Week','Team','Opp','Spread','Total','True_Total','Under']])
    print('\n')

# Make predictions for NFL Totals (Season = 2024, Divisional Round)
# Set the dataframe
df = nfl_df.query('Home == 1').reset_index(drop=True)

# Set the features and the target variable
features = ['Spread','Total']
target = 'Under'

# Set the season and the week
season = 2024
week = 18

# Create training set
train_df = df.query('Season < @season or (Season == @season and Week < @week)')

# Create X_train and y_train
X_train = train_df[features]
y_train = train_df[target]

# Two-dimensional list with upcoming game data from the home team's perspective
week1 = [
['Texans @ Chiefs', -8.5, 41.5],
['Commanders @ Lions', -9.0, 55.5],
['Rams @ Eagles', -6.0, 44.0],
['Ravens @ Bills', +1.0, 51.5]
]

# Create X_new dataframe from the upcoming game data
X_new = pd.DataFrame(week1, columns=['Game','Spread','Total'])

# Set the model
model = KNeighborsClassifier(n_neighbors=9)

# Fit the classifier
clf = model.fit(X_train, y_train)

# Make the predictions
y_pred = clf.predict(X_new[features])

# Add predictions to the dataframe
X_new['KNC(9)'] = y_pred
X_new['KNC(9)'] = X_new['KNC(9)'].apply(lambda x: 'Under' if x == 1 else 'Over')

# Display the dataframe with the predictions
print(f'MODEL PREDICTIONS FOR WEEK {week} OF THE {season} NFL SEASON\n')
print(X_new[['Game','Spread','Total','KNC(9)']])