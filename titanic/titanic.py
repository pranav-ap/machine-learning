# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC, LinearSVC
#from sklearn.ensemble import RandomForestClassifier #, GradientBoostingClassifier

# Modelling Helpers
#from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split #, StratifiedKFold
#from sklearn.feature_selection import RFECV

##### Visualisation #####
#import matplotlib as mpl
import matplotlib.pyplot as plt
#import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
#%matplotlib inline
#mpl.style.use( 'ggplot' )
#sns.set_style( 'white' )
#pylab.rcParams[ 'figure.figsize' ] = 8 , 6

def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

# df is the y-axis, var is x-axis, target is the set of curves,  
def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

# target is y-axis, cat is x-axis
def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = titanic.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 0 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )
    
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))

################################### Titanic ####################################

# Importing the dataset
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
titanic = train.append( test, ignore_index = True )

### FEATURE ENGINEERING

# Make new features named Title and Status based on Name
title_status = {
                    "Capt":       "Officer", 
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }

title = pd.DataFrame()
title['Title'] = titanic[ 'Title' ] = titanic[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
title['Status'] = titanic['Status'] = titanic['Title'].map( lambda title: title_status[title])

# Extract cabin category from cabin number
cabin = pd.DataFrame()
titanic['Cabin'] = titanic.Cabin.fillna('U')
titanic['Cabin'] = titanic['Cabin'].map(lambda c: c[0])

# Fill missing ages with mean
age = pd.DataFrame()
age['Age'] = titanic['Age'] = titanic.Age.fillna(titanic.Age.mean())


fare = pd.DataFrame()
fare['Fare'] = titanic['Fare'] = titanic.Fare.fillna(titanic.Fare.mean())

# Extract prefix of a ticket, returns 'XXX' if no prefix
def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' ).replace( '/' , '' ).split()
    ticket = map( lambda t : t.strip() , ticket)
    ticket = list( filter( lambda t : not t.isdigit(), ticket ) )
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'

ticket = pd.DataFrame()
titanic['Ticket'] = titanic['Ticket'].map( cleanTicket )

# Transform Sex into binary values 0 and 1
sex = pd.DataFrame()
sex['Sex'] = titanic['Sex'] = titanic['Sex'].map( lambda s: 1 if s == 'male' else 0)

# Make new feature named Family size and category
family = pd.DataFrame()
family['FamilySize'] = titanic['FamilySize'] = titanic[ 'Parch' ] + titanic[ 'SibSp' ] + 1; # +1 is to include the person considered  

family['Family_Single'] = titanic['Family_Single'] = titanic['FamilySize'].map( lambda s : 1 if s == 1 else 0 )
family['Family_Small'] = titanic['Family_Small']  = titanic['FamilySize'].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family['Family_Large'] = titanic['Family_Large']  = titanic['FamilySize'].map( lambda s : 1 if 5 <= s else 0 )

# Dummy encoding and avoiding dummy variable trap
prefix_dict = {
               'Cabin': 'Cabin', 
               'Embarked': 'Embarked', 
               'Ticket': 'Ticket'
              }

titanic = pd.get_dummies( titanic , columns = ['Cabin', 'Embarked', 'Ticket'], drop_first = False, prefix = prefix_dict )

# separate titanic into dependent and independent features
# titanic['Embarked_C'], titanic['Embarked_Q'], titanic['Embarked_S'],
train_test_X = pd.concat([titanic['Age'], titanic['Fare'], titanic['Sex'], titanic['Cabin_A'], titanic['Cabin_B'], titanic['Cabin_C'], titanic['Cabin_D'], titanic['Cabin_E'], titanic['Cabin_F'], titanic['Cabin_G'], titanic['Cabin_T'], titanic['Cabin_U'] ], axis = 1)
train_test_y = titanic.loc[:, ['Survived']]

# make valid sets
train_valid_X = train_test_X[:891]
train_valid_y = train_test_y[:891]

### optimization - build optimal model using backward elimination
import statsmodels.formula.api as sm
## add a column of ones 
train_valid_X = np.append(arr = np.ones((891, 1)).astype(int), values = train_valid_X, axis = 1)

# lower the p-value the more significant the variable is wrt independent variable 

## start from all predictors
X_opt = train_valid_X#[:, range(0, 16)]
## fit the model 
regressor_Logit = sm.Logit(endog = train_valid_y, exog = X_opt).fit()
regressor_Logit.summary() # FYI default SL = 5% = 0.05

# pass 2
X_opt = train_valid_X[:, [1,2,3]]
regressor_Logit = sm.Logit(endog = train_valid_y, exog = X_opt).fit()
regressor_Logit.summary()

# pass 3
X_opt = train_valid_X[:, [2,3]]
regressor_Logit = sm.Logit(endog = train_valid_y, exog = X_opt).fit()
regressor_Logit.summary()

# make test and train sets
test_X = train_test_X[891:]
test_X = pd.concat([test_X['Fare'], test_X['Sex']], axis = 1)

train_X, valid_X, train_y, valid_y = train_test_split( X_opt , train_valid_y , train_size = 0.7, random_state = 0 )

#### Modeling 
model = LogisticRegression()
model.fit( train_X , train_y )


# Score the model
print( model.score( train_X , train_y ) , model.score( valid_X , valid_y ) )

test_y = model.predict( test_X )
passenger_id = titanic[891:].PassengerId
prediction_DataFrame = pd.DataFrame({'PassengerId': passenger_id, 'Survived': test_y })
prediction_DataFrame.to_csv('titanic_pred.csv', index = False)
