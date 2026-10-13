# ⚽ Football Player Market Value Prediction

A machine learning project that predicts football players' end-of-season market values using player statistics and starting market value. The project compares Linear Regression and Random Forest models and provides a menu-driven application for player valuation, model comparison, and scouting analysis.

Author: Sukrit Yadav

## Project Overview

Football player market values are influenced by factors such as age, playing time, goals, assists, and existing market value. This project uses machine learning to estimate a player's market value at the end of a season based on their starting market value and season statistics.

The project includes two regression models:

- **Linear Regression**
- **Random Forest Regressor**

The models are evaluated using Mean Absolute Error (MAE) and the R² score. A Python command-line application allows users to explore predictions, compare models, and identify players whose actual market values differ from the model's estimates.

## Project Objectives

- Predict end-of-season football player market values.
- Compare Linear Regression and Random Forest performance.
- Explore the relationship between player statistics and market valuation.
- Identify players whose actual ending values are below the model's estimates.
- Analyse predicted market value increases and decreases.
- Provide custom predictions using user-entered player statistics.

## Dataset

The dataset was obtained from **Kaggle** and consists of CSV files containing football player match statistics and market values.

Link: https://www.kaggle.com/datasets/davidcariboo/player-scores

The project uses football player match statistics and market-value data from the dataset on Kaggle. The raw data is cleaned and processed into player-season records for model training and evaluation.

The features used by the models are:

| Feature | Description |
|---|---|
| Appearances | Number of appearances during the season |
| Yellow Cards | Total yellow cards received |
| Red Cards | Total red cards received |
| Goals | Total goals scored |
| Assists | Total assists recorded |
| Minutes Played | Total minutes played |
| Age | Player's age |
| Starting Market Value | Player's market value at the beginning of the season |

**Target variable:** The player's market value at the end of the season.

The models use season statistics and starting market value to estimate the ending market value.

## Machine Learning Models

### 1. Linear Regression

Linear Regression is used as a baseline model to estimate the relationship between the input features and the player's ending market value.

### 2. Random Forest Regressor

Random Forest combines predictions from multiple decision trees to model relationships between player statistics and market value. It was the better-performing model in the reported evaluation.

## Model Performance

The models were evaluated using a time-based split, with the 2025 season held out for testing and earlier seasons used for training.

| Evaluation Metric | Linear Regression | Random Forest |
|---|---:|---:|
| Mean Absolute Error (MAE) | €2,413,199.75 | €1,728,343.45 |
| R² Score | 0.8132 | 0.8810 |

### Results

**Random Forest performed better than Linear Regression** on the held-out test data.

- **Lower MAE:** Random Forest's predictions differed from actual ending market values by approximately €1.73 million on average, measured using absolute error.
- **Higher R²:** Its R² score of approximately 0.881 indicates that the model explains around 88.1% of the variation in ending market values in the test dataset.

These metrics describe performance on the test data; R² is not the percentage of predictions that are correct. The results also do not guarantee the same performance on future seasons.

## Application Features

The project includes a menu-driven Python application with the following functionality:

1. **Feature Importance:** Display the Random Forest model's feature importance scores.
2. **Player Statistics:** View a player's season statistics.
3. **Player Valuation:** Compare starting market value, predicted ending value, and actual ending value.
4. **Custom Prediction:** Enter player statistics and starting market value to generate a predicted ending value.
5. **Full Player Report:** View player statistics alongside predicted and actual valuations.
6. **Scouting Shortlist:** Find players whose actual ending market values are below the model's estimates.
7. **Predicted Market Value Risers:** Rank players by their predicted increase in market value.
8. **Predicted Market Value Fallers:** Rank players by their predicted decrease in market value.
9. **Model Comparison:** Compare Linear Regression and Random Forest predictions for a player.
10. **Actual vs. Predicted Visualisation:** Plot Random Forest predictions against actual ending market values.

The scouting shortlist is intended to highlight players for further investigation. A model estimate above a player's actual market value does not necessarily mean the player is genuinely undervalued.

## Technologies Used

- **Python** — application and machine learning workflow
- **Pandas** — data manipulation and analysis
- **Scikit-learn** — model training and evaluation
- **Joblib** — saving and loading trained models
- **Matplotlib** — data visualisation
- **Visual Studio Code** — development environment

## Project Structure

The application expects saved models and prediction CSV files to be available at the paths used in the source code.

```text
football-player-market-value-prediction/
├── .vscode/
├── data/
│   ├── final/
│   ├── raw/
│   ├── training/
│   └── updated/
├── Models/
│   ├── Linear_regression_model.pkl
│   └── random_forests_model.pkl
├── data_cleaning.py
├── data_processing.py
├── data_processing_2.py
├── Linear_regression.py
├── Main.py
├── Random_forest.py
├── requirements.txt
├── .gitignore
└── README.md
```

### File Descriptions

| File or Folder | Purpose |
|---|---|
| `data/` | Contains the project's datasets at different processing stages. |
| `data/raw/` | Stores the original datasets. |
| `data/training/` | Contains data used for model training. |
| `data/updated/` | Stores updated datasets. |
| `data/final/` | Contains final datasets used by the application. |
| `Models/` | Stores the trained Linear Regression and Random Forest models. |
| `data_cleaning.py` | Handles data cleaning. |
| `data_processing.py` | Processes the dataset. |
| `data_processing_2.py` | Performs additional data processing. |
| `Linear_regression.py` | Contains the Linear Regression model workflow. |
| `Random_forest.py` | Contains the Random Forest model workflow. |
| `Main.py` | Runs the menu-driven application. |
| `requirements.txt` | Lists the Python dependencies. |
| `.gitignore` | Specifies files Git should ignore. |
| `README.md` | Documents the project. |



## Installation and Usage

### Prerequisites

- Python 3 installed
- Git (optional, for cloning the repository)

### 1. Clone the repository

```bash
git clone https://github.com/Sukrit-ydv/football-player-market-value-prediction.git
cd football-player-market-value-prediction
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

Install the libraries used by the project:

```bash
pip install pandas scikit-learn joblib matplotlib
```

### 4. Run the application

Ensure that the saved model files and required prediction CSV files exist at the paths expected by the application.

Run:

```bash
python Main.py
```

Follow the on-screen menu to explore the available features.

## Limitations

- **Dependence on starting market value:** Starting market value is the model's most influential feature. Since it already reflects existing market expectations, the model may rely heavily on that information when estimating ending value.
- **Limited player statistics:** The current feature set focuses on appearances, cards, goals, assists, minutes played, age, and starting market value. It does not include several potentially useful measures for evaluating defenders and goalkeepers.
- **Unusual custom inputs:** Random Forest predictions are limited by the patterns represented in the training data. Predictions for unrealistic or rarely observed combinations of statistics may be unreliable, and the model does not extrapolate beyond its learned patterns in the same way a linear model can.
- **Market value is not determined by statistics alone:** Real-world valuations can also be affected by factors not included in the model, such as injuries, contract length, transfer demand, and club circumstances.
- **Predictions are estimates:** The application supports analysis and scouting research, but its outputs should not be treated as guaranteed future values or definitive assessments of a player's worth.

## Future Improvements

Potential improvements include:

- Adding defensive metrics such as tackles, interceptions, clearances, and blocks.
- Adding goalkeeper-specific statistics such as saves, save percentage, and goals conceded.
- Incorporating contract information, injury history, and other relevant valuation factors where reliable data is available.
- Evaluating performance across additional held-out seasons.
- Comparing additional regression models and tuning hyperparameters.
- Improving validation for custom inputs and displaying warnings for unrealistic statistics.

## Disclaimer

This project was developed for educational and analytical purposes. The predicted market values are model estimates and should not be interpreted as official valuations, financial advice, or guarantees of actual transfer prices.

## Author

**Sukrit Yadav**

GitHub: [Sukrit Yadav](https://github.com/Sukrit-ydv)

