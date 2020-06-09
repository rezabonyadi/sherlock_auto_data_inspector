from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer


available_models={'LDA': LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
            ,'SVM_l1': LinearSVC(penalty='l1', dual=False, max_iter=40000, C=1.0)
            ,'LR': LogisticRegression(solver='lbfgs')
            ,'SVM': LinearSVC(max_iter=4000)
            ,'GPC': GaussianProcessClassifier(kernel=1.0 * DotProduct(sigma_0=1.0)**2)
            ,'ENET': SGDClassifier(penalty='elasticNet')
            ,'DTR': DecisionTreeClassifier()
            ,'RNF': RandomForestClassifier(n_estimators=100)
            ,'GBT': GradientBoostingClassifier(n_estimators=100)
            ,'EXT': ExtraTreesClassifier()
            ,'QDA': QuadraticDiscriminantAnalysis()
           }

available_models_preprocessing = {'Standardization': StandardScaler(), 'MinMaxScaler': MinMaxScaler(),
                           'RobustScaler': RobustScaler()}

available_random_selections = ['stratified', 'keep_order', 'random']
available_networks_layouts = ['spring']

