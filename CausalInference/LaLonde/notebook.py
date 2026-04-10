import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset Info

    The study looked at the effectiveness of a job training program (the treatment) on the real earnings of an individual, a couple years after completion of the program.

    The data consists of a number of demographic variables (age, race, academic background, and previous real earnings), as well as a treatment indicator, and the real earnings in the year 1978 (the response).

    - treat is the treatment assignment (1=treated, 0=control).
    - age is age in years.
    - educ is education in number of years of schooling.
    - black is an indicator for African-American (1=African-American, 0=not).
    - hispan is an indicator for being of Hispanic origin (1=Hispanic, 0=not).
    - married is an indicator for married (1=married, 0=not married).
    - nodegree is an indicator for whether the individual has a high school degree (1=no degree, 0=degree).
    - re74 is income in 1974, in U.S. dollars.
    - re75 is income in 1975, in U.S. dollars.
    - re78 is income in 1978, in U.S. dollars.
    """)
    return


@app.cell
def _():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    return pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv('lalonde_data.csv')
    df.head()
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    # Look at the final earnings of the control at treated groups
    df.groupby('treat')['re78'].agg(['median','mean'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At first glance, we don't see a very clear difference between the treated group and the control, at best a decrease in earnings, wich is the oposite of the initial hypthesis.
    Before jumping to conclusiones, let's explroe the dataset to see how the experiment was perfromed
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Exploration
    """)
    return


@app.cell
def _(df, plt):
    plt.hist(df[df['treat'] == 0]['re78'], bins=20, alpha=0.8, label='0')
    plt.hist(df[df['treat'] == 1]['re78'], bins=20, alpha=0.8, label='1')
    plt.title("1978 Salary Histograms")
    plt.legend(title="Treatment")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From this resulting histogram, we can see that both groups are quite different. Lets explore the previous years to see if the differentiation is because of the treatment or a flaw from the beggining
    """)
    return


@app.cell
def _(df, plt, sns):
    # reshape from wide → long
    df_long = df.melt(id_vars='treat', 
                      value_vars=['re74', 're75', 're78'],
                      var_name='Treatment', 
                      value_name='Salary')

    # plot
    sns.boxplot(x='Treatment', y='Salary', hue='treat', data=df_long)
    plt.xlabel('Year')
    plt.ylabel('Salary')
    plt.legend(title='Treatment')

    plt.show()
    return


@app.cell
def _(df, plt, sns):
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))

    # --- Plot 1: age (top-left) ---
    axes[0, 0].hist(df[df['treat'] == 0]['age'], bins=20, alpha=0.8, label='Control (0)')
    axes[0, 0].hist(df[df['treat'] == 1]['age'], bins=20, alpha=0.8, label='Treated (1)')
    axes[0, 0].set_title("Age Distribution by Treatment")
    axes[0, 0].set_xlabel("Age")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].legend()

    # --- Plot 2: educ (top-right) ---
    axes[0, 1].hist(df[df['treat'] == 0]['educ'], bins=20, alpha=0.8, label='Control (0)')
    axes[0, 1].hist(df[df['treat'] == 1]['educ'], bins=20, alpha=0.8, label='Treated (1)')
    axes[0, 1].set_title("Education Distribution by Treatment")
    axes[0, 1].set_xlabel("Years of Education")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()

    # --- Plot 3: share of black (bottom-left) ---
    sns.barplot(x='treat', y='black', data=df, estimator='mean', ax=axes[1, 0])
    axes[1, 0].set_title("Share of Black Individuals by Treatment")
    axes[1, 0].set_xlabel("Treatment")
    axes[1, 0].set_ylabel("Proportion Black")

    # --- Plot 4: share of nodegree (bottom-right) ---
    sns.barplot(x='treat', y='nodegree', data=df, estimator='mean', ax=axes[1, 1])
    axes[1, 1].set_title("Share of No Degree Individuals by Treatment")
    axes[1, 1].set_xlabel("Treatment")
    axes[1, 1].set_ylabel("Proportion No Degree")

    # --- Plot 5: share of married (bottom-left) ---
    sns.barplot(x='treat', y='married', data=df, estimator='mean', ax=axes[2, 0])
    axes[2, 0].set_title("Share of married Individuals by Treatment")
    axes[2, 0].set_xlabel("Treatment")
    axes[2, 0].set_ylabel("Proportion Married")

    # --- Plot 4: share of nodegree (bottom-right) ---
    sns.barplot(x='treat', y='hispan', data=df, estimator='mean', ax=axes[2, 1])
    axes[2, 1].set_title("Share of hispanic Individuals by Treatment")
    axes[2, 1].set_xlabel("Treatment")
    axes[2, 1].set_ylabel("Proportion Hispanic")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With these plots it is clear that the groups were very different from the beggining, meaning that we can't just say that the treatment had no effect, we need to control for all the variables (confounding variables)
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
