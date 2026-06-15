# FDA Car Price Prediction Project

This is a simple VS Code and GitHub-ready project for the **Fundamentals of Data Analytics** assignment.

## What this project does

The project uses the `auto.csv` dataset to create:

- raw/unclean data visualisations
- cleaned data visualisations
- missing-value handling
- outlier treatment
- feature engineering
- horsepower binning
- feature normalisation
- Linear Regression
- Random Forest Regression
- Logistic Regression classification

## Simple raw vs clean control

Open this file:

```text
src/main.py
```

At the top, change this line:

```python
VISUALISATION_MODE = "both"
```

You can use:

```python
VISUALISATION_MODE = "raw"
```

This creates only raw/unclean visualisations.  
No cleaning is applied.

```python
VISUALISATION_MODE = "clean"
```

This cleans the data first, then creates cleaned visualisations and models.

```python
VISUALISATION_MODE = "both"
```

This creates raw visualisations first, then cleaned visualisations and models.

You can also disable modelling:

```python
RUN_MODELS = False
```

## Project structure

```text
FDA_Car_Price_GitHub_Project_Simple_Toggle/
├── data/
│   └── auto.csv
├── src/
│   ├── __init__.py
│   └── main.py
├── outputs/
│   ├── figures/
│   └── tables/
├── run.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## How to run in VS Code terminal

Go into the project folder:

```bash
cd FDA_Car_Price_GitHub_Project_Simple_Toggle
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install packages:

```bash
python -m pip install -r requirements.txt
```

Run the project:

```bash
python run.py
```

## Output location

Figures will be saved here:

```text
outputs/figures/
```

Tables will be saved here:

```text
outputs/tables/
```
