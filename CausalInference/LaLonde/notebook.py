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
def _():
    import warnings
    warnings.filterwarnings('ignore')
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Simple Approach
    The simplest type of model we can use is a linear model:

    Y0=α+βX+ϵ

    Y1=Y0+γD


    If this is accurate, fitting the following model to the data using linear regression will give us an estimate of the Average Treatment Effect (ATE):
    Y=α+βX+γD


    ϵ
      is called a residual and represents the noise
    """)
    return


@app.cell
def _():
    covariates = ['age', 'educ', 'black', 'hispan', 'married', 'nodegree', 're74', 're75']
    return (covariates,)


@app.cell
def _(covariates, df):
    # https://pypi.org/project/CausalInference/
    from causalinference import CausalModel

    causal = CausalModel(
        Y=df['re78'].values, 
        D=df['treat'].values, 
        X=df[covariates].values)

    causal.est_via_ols(adj=1)
    # adj=1 corresponds to the simplicity of the model we entered
    # This is called a "constant treatment effect"

    print(causal.estimates)
    return CausalModel, causal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This model predicts that the Average Treatment Effect (ATE, the job training) is $1548 extra annual earnings. This is very different from our previous raw results predicting that the job training had negative effects on earnings.

    Assuming that our model accurately describes the counterfactual X, CausalModel provides the 95% confidence interval. What this means is that, if we were to repeat this treatment experiment, in 95% of the cases the Average Treatment Effect would be within that interval. That doesn't mean that the true value is within that interval.

    Based on the assumption that the residuals are normally distributed, the 95% confidence interval is calculated as:
    AVG±1.96∗STD/√‾n


    In practice, as the confidence interval is very large, my interpretation is that the experiment should have had more people if a better estimate of the extra earnings was desired. Ways to control the standard deviation could also be explored.

    Overall, assuming that we controlled for all the effects and did it well, it seems that the job training had a positive effect on earnings. Indeed, although the standard deviation is very large, the p value of 0.035 rejects the null hypothesis (no effect) with a confidence level of 97.5%. However, the truth is that we don't know if we modelled the counterfactual well, and this could change everything... As we will see later, estimators such as the Ordinary Least Square (OLS) estimator can behave poorly when there is not enough covariate overlap, and that's because the estimator needs to extrapolate too much from one group to another.

    A more structured approach as we will see below can allow us to increase our confidence that the covariants are well controlled for. We will see many steps, but one simple idea is the technique of matching: the idea is to find for each sample which received the treatment a similar sample in the control group, and to directly compare these values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # More complete approach
    ### Pre-processing phase:
    assess covariate balance
    estimate propensity score
    trim sample
    stratify sample

    ### Estimation phase:
    blocking estimator or/and
    matching estimator
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pre-processing phase
    In the pre-processing phase, the data is inspected and manipulated to allow credible analysis to be conducted on it.

    As we discussed in the previous section, one key method for disantangling the treatment effect from the covariant effects is the matching technique. In this technique we compare subjects that have similar covariate values (i.e. same age, rage, income etc). However, our ability to compare such pairs depends heavily on the degree of overlap of the covariates between the treatment and control group. This is called covariate balance.

    Said otherwise, to control the effect of education, one way is to look at people in the tested group and in the non-tested group that all have the same level of education, say a bachelor degree. However, if nobody in the test group has a bachelor degree while many do in the non-test group, this procedure is impossible.

    (1) assess covariate balance to assess whether how easily people can be matched. If there is too much unbalance, direct matching will rarely be possible, and we may need to use more complex techniques, if at all possible.
    """)
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(causal):
    print(causal.summary_stats)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The aim here is to assess the overlap between the control and treatment groups. It can be seen that X2, X4, and X6 (black, married, revenue in 1974) have a large normalized difference, beyond 0.5. This can be interpreted as an imbalance. Concretely, there are way more black people, less married people and lower income in 1974 in the treatment group than in the control group, as previously shown in the exploratory section.

    The impact of imbalance is to make the matching technique harder to apply. We'll see later how we can try to correct for it (however, ideally the study would be more balanced!).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    (2) Propensity Score - the probability of receiving the treatment, conditional on the covariates.

    Propensity is useful for assessing and improving covariate balance. Indeed a theorem by Rosenbaum and Rubin in 1983, proves that, for subjects that share the same propensity score (even if their covariate vectors are different), the difference between the treated and the control units actually identifies a conditional average treatment effect.

    Thus, instead of matching on the covariate vectors X themselves, we can also match on the single-dimensional propensity score p(X), aggregate across subjects, and still arrive at a valid estimate of the overall average treatment effect.
    """)
    return


@app.cell
def _(causal):
    causal.est_propensity_s()
    print(causal.propensity)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    (3) Trim sample. This excludes subjects with extreme propensity scores. Indeed it will be very hard to match those extreme subjects, so the usual strategy is to focus attention on the remaining units that exhibit a higher degree of covariate balance.
    """)
    return


@app.cell
def _(causal):
    # extreme propensity is a very high probability to be either in the control group or the treatment group
    # that makes matching difficult

    causal.trim_s()
    print(causal.summary_stats)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this new subset, the normal difference for most variables is rather balanced. Only X2 (number of black people) is still unbalanced.

    It is worth noting that the initial sample of 614 people (429 controls, 185 treated) has been drastically trimmed to 297 people (157 controls, 140 treated).

    In this more balanced sub-sample, without using any model, the average earnings in 1978 is more like what we would expect: populations that received training (treated) earn in average $875 more than the control group.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    (4) Stratify sample - group similar subjects together. People are grouped in layers of similar propensity scores. These bins should have an improved covariate balance, and we should be able to compare and match samples within those bins.
    """)
    return


@app.cell
def _(causal):
    causal.stratify_s()
    print(causal.strata)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Within bins, the raw difference in outcome should be a good representation of the real treatment effect. For example:

    People in group 1 are unlikely to be in the treatment group (well off?). For them, the training improved their earnings by $1399 in average.

    People in group 4 are likely to be in the treatment group (poor?). For them, the training improved their earnings even more, with a mean of $2211 for that year 1978.

    Something that looks quite bad is that outcomes for the group 3 are totally different from that of the other groups. The trend seems to be that the higher the propensity score, the higher the raw difference in outcome for each stratum. but this one shows opposite results... This may be a sign that we haven't controlled for enough factors (or that the propensity calculation is wrong?). Or it might also be a true representation or reality: some people may benefit from the job training, while other may not. It might also be random and the reflection that we are working with a relatively small sample (74 elements in bin 3).

    Let's see in the analysis phase if regressions within each stratum will be able to control for confounding variables better.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estimation phase
    In the estimation phase, treatment effects of the training can be estimated in several ways.

    (1) The blocking estimator - although each layer of the stratum is pretty balanced and gives reasonable raw results, this estimator goes further and controls for the confounding factors within each layer of the stratum. More precisely, this estimator uses a least square estimate within each propensity bin, and from this produces an overall average treatment effect estimate.
    """)
    return


@app.cell
def _():
    # causal.est_via_blocking()
    # print(causal.estimates)

    # for some reason I'm having a singular matrix when calculating this blocking estimator
    # on one of the stratum
    # I've tried changing the stratum structure and the set of variables,
    # however, the singularity persists when calculating the covariance matrix

    # this is one of the issue of this causalinference package:
    # it needs to invert large matrixes, which can fail

    # The cause of singular matrix is due to not normalizing X, as the big difference in values can cause problems.
    # We have created another model called causal_norm and it works if we eliminate the trimming (wich intoroduces more problems)
    return


@app.cell
def _(covariates, df):
    import statsmodels.api as sm
    df_X = df[covariates]
    df_Y = df['re78']

    # Add constant manually just for this test
    X_test = sm.add_constant(df_X)
    model = sm.OLS(df_Y, X_test)
    results = model.fit() 

    # Statsmodels will either:
    # 1. Drop the offending variable for you
    # 2. Give you a 'Design Matrix is Singular' error with a list of offending indices.
    print(results.summary())
    return (df_X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The Problem: Variable Scaling
    Looking at the variables, there is a massive discrepancy in scales:
    Dummy variables (black, hispan, married): Range from 0 to 1.
    Earnings variables (re74, re75): Range from 0 to tens of thousands.
    """)
    return


@app.cell
def _(CausalModel, df, df_X):
    from sklearn.preprocessing import StandardScaler

    # List of continuous variables that have large scales
    continuous_vars = ['age', 'educ', 're74', 're75']

    # Initialize scaler
    scaler = StandardScaler()

    # Create a copy of your dataframe to keep it clean
    X_scaled = df_X.copy()

    # Scale only the continuous variables
    X_scaled[continuous_vars] = scaler.fit_transform(X_scaled[continuous_vars])

    # Now run the causal model with X_scaled
    causal_norm = CausalModel(
        Y=df['re78'].values, 
        D=df['treat'].values, 
        X=X_scaled.values)
    return (causal_norm,)


@app.cell
def _(causal_norm):
    causal_norm.est_propensity_s()
    print(causal_norm.propensity)
    return


@app.cell
def _():
    # causal_norm.trim_s()
    # print(causal_norm.summary_stats)

    # This is what is causing the Singular Matrix error for causal_norm
    return


@app.cell
def _(causal_norm):
    causal_norm.stratify_s()
    print(causal_norm.strata)

    return


@app.cell
def _(causal_norm):
    causal_norm.est_via_blocking()
    print(causal_norm.estimates)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    (2) The matching estimator - although each layer of the stratum is pretty balanced and gives reasonable raw results, this matching estimator controls for the confounding factors by matching even more thinely samples within each layer of the stratum. More precisely, this pairing is done via nearest-neighborhood matching. If the matching is imperfect, biias correction is recommended.

    If issues arrive with least square, such as excessive extrapolation, this matching estimator pushes until the end the unconfoundedness assumption and nonparametrically matches subjects with similar covariate values together. In other words, if the confounding factors are equal for both element of a pair, the difference between the two will be the real treatment effect. In the causalinference package, samples are weighted by the inverse of the standard deviation of the sample covariate, so as to normalize.

    Where matching discrepancy exist, least square will be used, but very locally, so large extrapolations should be less of a problem.
    """)
    return


@app.cell
def _(causal):
    causal.est_via_matching(bias_adj=True)
    print(causal.estimates)
    return


@app.cell
def _(causal):
    # allowing several matches
    causal.est_via_matching(bias_adj=True, matches=4)
    print(causal.estimates)

    return


@app.cell
def _(causal_norm):
    causal_norm.est_via_matching(bias_adj=True)
    causal_norm.est_via_ols(adj=1)

    print(causal_norm.estimates)
    return


@app.cell
def _(causal_norm):
    # allowing several matches
    causal_norm.est_via_matching(bias_adj=True, matches=4)
    print(causal_norm.estimates)

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
