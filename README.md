# Sherlock - the Automatic Data Inspector (Shadi)

SherlockAI is my attempt to design a tool which inspects and generates 
statistically tested data-driven insights from the data automatically. 
There are over tens, if not hundreds, of such tool out there, 
but SherlockAI is different!
Here is what is not quite right with other existing tools (e.g., ???)
* **Stability of features**: Insights out of multivariate analyses is usually 
connected to analysing the importance of variables in describing a response pattern
given a model. Given the fact that the given population is just a limited 
representative of reality, plus there might be outliers in the population, 
it is risky to use the whole population for feature selection. A more robust
approach would be to perform bootstrap aggregation on not only the samples 
but also the variables. See [????] for more information and detailed 
statistical discussions.

* **Statistics of big data**: Usual stat methods such as Mann-Whytney and t-test
do not work when then sample size is large (over 1000 samples!). These tests were 
designed during word war II mostly for psychology-oriented experiments
which had a small population sizes (order of 10s to 100s). If you have a 
population of 5000 samples, p-value of t-test and Mann-Whytney is always 
smaller than 0.05. Alternatives are Cohen's *d* test and direct 
calculation of overlapping percentage between distributions [???]. This 
is usually overlooked in ML research and in AutoML tools.

* **Issue of small sample size**: 
* **Auto anomally detection**: 

**Assumptions**
1. The inputs are all numerical (Numbers, no categories in variables)
2. Focus is on tabular data for now (time series is in vision, images are far away)

**High level functionalities**

* Sherlock Auto Explorer (SAD)
    * Explore each variable in isolation: (Done)
        * Is normal (Done)
        * Range (Done)
        * Percentiles (Done)
        * Bootstrap median/mean (Done)
        * Vis: Variables distributions (box plot) (Done)
    * With response (Done)
        * Categorical response
            * OVL of category pairs (Done)
            * p_val of category pairs (report Benferoni correction level) (Done)
            * Cohen's d of category pairs (Done)
            * Chi2 (Done)
            * Vis: Box plot (Done)
        * Continuous response (Parked)
            * Make it categorical and do the categorical analyses
            * Correlation
            * ?
    * Explore transformations impact (Parked, so what)
        * Include Log/Box-cox
        * Impact on normality
        * Impact on "Describing the response?"
        * Vis: original vs transformeds
    * Explore variable pairs (Done)
        * Correlation coefficient (Done)
        * Vis: Network connected (Med as size/relationship as edge) (Done)
        * ? 

* Sherlock Auto Insight 
    * Provide variable importance (Done)
        * Importance stability factor (ISF) (Done)
        * Bagged importance level (BIL) (Done)
        * Vis: Network of pairs working together (ISF as edges, Degree centrality as color)
        * Vis: Un-connected network (ISF as size, BIL color) (Done)
    * Auto regression: Provides variable importance (Parked)
        * ????
        * 
    * Auto dimensionality reduction
        * Supervised
            * LDA
        * Unsupervised
            * PCA
            * AutoEncoder 

* Sherlock Auto Visualizer (is done gradually as part of others)

* Sherlock Auto Modeller (Parked)

* Sherlock Auto Cleaner
    * Anommally detection
        * one class SVM
        * Autoencoder
        * Six sigma
        * 

* Sherlock Auto Characterizer (Parked)

* Sherlock engine
    * Stat engine
        * OVL
        * Cohen's d
        * 
    * Models engine
        * Twin SVM
        