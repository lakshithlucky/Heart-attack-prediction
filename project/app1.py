from flask import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import MySQLdb
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score

filename = 'diabetes.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
page = "login.html"



@app.route('/')
def home():
	return render_template('home.html',src=page)

@app.route('/title.html')
def title():
    return render_template('title.html')

@app.route('/body1.html')
def body1():
    return render_template('body1.html')

@app.route('/login.html')
def logins():
    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def login():
    msg = ''
    db = MySQLdb.connect("localhost","root","","heartdisease" )
    if request.method == 'POST' and 'user' in request.form and 'pwd' in request.form:
        username_patient = request.form['user']
        password_patient = request.form['pwd']
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username_patient, password_patient,))
        account = cursor.fetchone()
        if account:
            msg = 'Logged in successfully!'
            return render_template('predict.html', msg=msg)
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)
 
@app.route('/predict.html')
def predicts():
    return render_template('predict.html')

@app.route('/register.html')
def registers():
    return render_template('register.html')

@app.route('/results', methods=['GET', 'POST'])
def register():
    msg = ''
    db = MySQLdb.connect("localhost","root","","heartdisease" )
    if request.method == 'POST':
        username_patient = request.form['user']
        password_patient = request.form['pwd']
        cnfpassword_patient = request.form['cpwd']
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        sql = "SELECT * FROM users WHERE username = '%s'" % (username_patient)
        cursor.execute(sql)
        account = cursor.fetchone()
        if account:
            msg = 'Account already exists!'
            return render_template('register.html',msg=msg)
        else:
            cursor.execute('insert into users values("%s", "%s", "%s")' % \
             (username_patient,password_patient,cnfpassword_patient))
            db.commit()
            msg = 'You have successfully registered!'
            return render_template('login.html',msg=msg)


@app.route('/result', methods=['POST','GET'])
def predict():
    db = MySQLdb.connect("localhost","root","","heartdisease" )
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        cp = request.form['pericarditis']
        trestbps= request.form['bloodpressure']
        chol = request.form['cholestrol']
        fbs = request.form['sugar']
        restecg = request.form['resting']
        thalach = request.form['heartrate']
        exang = request.form['exercise']
        oldpeak = float(request.form['oldpeaks'])
        slope = request.form['oldpeak']
        ca = request.form['carotid']
        thal = request.form['thal']
        
        
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        targets = classifier.predict(data)
        for i in targets:
            target = i
        cursor = db.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('insert into dataset values("%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s")' % \
             (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target))
        db.commit()
        return render_template('result.html',prediction=targets)

@app.route('/result.html',methods=['POST'])
def result():
    return render_template('result.html')

dataset = pd.read_csv('D:/mini1/data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
  
dataset['ca']=pd.to_numeric(dataset['ca'],errors='coerce')
dataset['thal']=pd.to_numeric(dataset['thal'],errors='coerce')
dataset['slope']=pd.to_numeric(dataset['slope'],errors='coerce')
dataset['chol']=pd.to_numeric(dataset['chol'],errors='coerce')
dataset['restecg']=pd.to_numeric(dataset['restecg'],errors='coerce')
dataset['thalach']=pd.to_numeric(dataset['thalach'],errors='coerce')
dataset['fbs']=pd.to_numeric(dataset['fbs'],errors='coerce')
dataset['exang']=pd.to_numeric(dataset['exang'],errors='coerce')
dataset['trestbps']=pd.to_numeric(dataset['trestbps'],errors='coerce')
 
    
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)      
    
 
    
X = pd.DataFrame(dataset.iloc[:,:-1].values)
xt = DataFrameImputer().fit_transform(X)   
x_train,x_test,y_train,y_test = train_test_split(xt,y,test_size=0.2,random_state=0)
sc = StandardScaler()
x_train  = sc.fit_transform(x_train)

x_test= sc.transform(x_test)



classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)
y_pred= classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
v= accuracy_score(y_test,y_pred)

filename = 'diabetes.pkl'
pickle.dump(classifier, open(filename, 'wb'))

if __name__ == '__main__':
	app.run(debug=True)
