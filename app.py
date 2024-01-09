from flask import Flask, render_template, request, session, url_for, redirect
import pymysql
import joblib
import pandas as pd #
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import string
import random
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import tree
    
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from PIL import Image, ImageDraw, ImageFont

from sklearn.metrics import roc_curve, roc_auc_score
import plotly.graph_objects as go

app = Flask(__name__)

app.secret_key = 'any random string'

def dbConnection():
    try:
        connection = pymysql.connect(host="localhost", user="root", password="root", database="algorithmvisualizer",port=3307)
        return connection
    except:
        print("Something went wrong in database Connection")


def dbClose():
    try:
        dbConnection().close()
    except:
        print("Something went wrong in Close DB Connection")
                
con = dbConnection()
cursor = con.cursor()

# ----------------------------------------------------------------------------------------------------------------

Student_df=pd.read_csv("StudentPerformance.csv")

df=pd.read_csv("preprocess.csv")
df_train=df.drop(['Unnamed: 0'],axis=1)
to_scale = [col for col in df_train.columns if df_train[col].max()>1]
scaler = RobustScaler()
scaled =scaler.fit_transform(df_train[to_scale])
scaled = pd.DataFrame(scaled, columns=to_scale)

# replace original columns with scaled columns
for col in scaled:
    df_train[col] = scaled[col]
    
DecisionTree_Classifier_model = joblib.load("models/DecisionTree_Classifier_Classify_model.joblib")
KNN_Classifier_model = joblib.load("models/KNeighbors_Classifier_Classify_model.joblib")
RandomForest_Classifier_model = joblib.load("models/RandomForestClassifier_Classify_model.joblib")
SVC_linear_Classifier_model = joblib.load("models/SVC_linear_Classify_model.joblib")
Logistic_Regression_Classifier_model = joblib.load("models/Logistic_Regression_Classify_model.joblib")

DecisionTree_Regressor_model = joblib.load("models/DecisionTree_Regressor_model.joblib")
KNN_Regressor_model = joblib.load("models/KNN_Regressor_model.joblib")
RandomForest_Regressor_model = joblib.load("models/RandomForest_Regressor_model.joblib")
SVC_Regressor_model = joblib.load("models/SVC_Regressor_model.joblib")
Logistic_Regression_Regressor_model = joblib.load("models/Logistic_Regression_model.joblib")

print('all models load')
    
def dynamic_label_encode(df):
    encoded_df = df.copy()
    label_encoders = {}

    for column in df.select_dtypes(include=['object']).columns:
        le = preprocessing.LabelEncoder()
        encoded_df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    return encoded_df, label_encoders

# ----------------------------------------------------------------------------------------------------------------
def getRandomfilepath():   
    N = 10
    res = ''.join(random.choices(string.ascii_letters, k=N))    
    if os.path.exists("static/confusion_matrix/"+res+".png"):
        getRandomfilepath()
    else:
        return "static/confusion_matrix/"+res+".png"
    
def getRandomhtmlfilepath():   
    N = 10
    res = ''.join(random.choices(string.ascii_letters, k=N))    
    if os.path.exists("static/confusion_matrix/"+res+".html"):
        getRandomhtmlfilepath()
    else:
        return "static/confusion_matrix/"+res+".html"

@app.route('/')
def index():
    return render_template('login.html') 

@app.route('/register')
def register():
    return render_template('register.html') 

@app.route('/login')
def login():
    return render_template('login.html')  

@app.route('/teacherlogin')
def teacherlogin():
    return render_template('teacherlogin.html') 

def plotScatterPlot(data,actual,path): 
    plt.figure(figsize=(8, 6))
    new_predictions = KNN_Regressor_model.predict(data) 
    plt.scatter(actual, new_predictions, c='blue', label='Scatter Plot')

    # Add labels and title
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('K-Nearest Neighbors Scatter Plot')
    
    # Show the legend
    plt.legend(loc='best')
    plt.savefig(path)
    
    
    
def decision_boundary(path): 
    X=df_train[['DBMS score','Software Engineering score']].values
    y=df_train['Pass/Fail'].values
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Fit the classifier to the data
    knn.fit(X, y)
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict the class for each point in the mesh grid
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='k')
    plt.title(f'KNN Decision Boundary (k={k})')
    plt.xlabel('DBMS score')
    plt.ylabel('Software Engineering score')
    plt.savefig(path)
    
def plot_lr_featurebar(path,col):
    feature_importances = Logistic_Regression_Classifier_model.coef_[0]
    plt.figure(figsize=(10, 6))
    
    plt.barh(col,feature_importances)
    plt.xlabel('Feature Importance')
    plt.title('Logistic Regression Feature Importance')
    plt.savefig(path)
    
def plot_dt_featurebar(path,X_train_new):
    feature_importances = DecisionTree_Classifier_model.feature_importances_
    feature_names = X_train_new.columns
    sorted_idx = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(14, 12))
    plt.title("Feature Importance")
    plt.bar(range(X_train_new.shape[1]), feature_importances[sorted_idx], align="center")
    plt.xticks(range(X_train_new.shape[1]), [feature_names[i] for i in sorted_idx], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.savefig(path)
    
def plot_tree_growth(path,X_train_new):
    plt.figure(figsize=(10, 6))
    tree.plot_tree(DecisionTree_Classifier_model, feature_names=X_train_new.columns)
    
    #Two  lines to make our compiler able to draw:
    plt.savefig(path)
    
def plot_roc_curve(path,sample):
    encoded_df1 = df_train[:int(sample)]
    a=encoded_df1.drop(["Pass/Fail",'Current Sem percentage','Student_id'],axis=1)
    b=encoded_df1["Pass/Fail"]
    y_pred_proba = Logistic_Regression_Classifier_model.predict_proba(a)[::,1]
    fpr, tpr, _ = metrics.roc_curve(b,y_pred_proba)
    plt.figure(figsize=(10, 6))
    #create ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.savefig(path)
    
def plot_svm_decisionboundary(path,X_train_new,y_train_new):
    # Apply PCA to reduce the dimensionality to 2 for visualization
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_train_new)
    y=y_train_new
    # Create an SVM classifier
    clf = SVC(kernel='linear', C=1)
    clf.fit(X_reduced, y)
    
    # Create a mesh grid to visualize the decision boundary
    h = .02
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and support vectors
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('SVM Decision Boundary with PCA')

    plt.savefig(path)
    
def plot_svm_hyperplane(path,X_train_new,y_train_new):
    clf = SVC(kernel='linear', C=1)
    clf.fit(X_train_new, y_train_new)
    
    # Create a 2D scatter plot of the data points
    def plot_data():
        plt.scatter(X_train_new.iloc[:, 5], X_train_new.iloc[:, 6], c=y_train_new, cmap=plt.cm.Paired)
    
    # Create a function to plot the hyperplane
    def plot_hyperplane(coef, intercept):
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
        plt.contour(xx, yy, zz, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    
    # Create an animation to move the hyperplane
    fig, ax = plt.subplots()
    plt.colorbar(plot_data(), ax=ax)
    plt.savefig(path)
    
def plot_feature_importances_graph(path,X_train_new):
    feature_importances = RandomForest_Classifier_model.feature_importances_
    # Get the names of the features
    feature_names = X_train_new.columns
    
    # Sort the features by importance
    sorted_idx = np.argsort(feature_importances)
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Random Forest')
    plt.savefig(path)
    
def plot_feature_importances_image(path,X_train_new):
    # Create a blank white image
    width, height = 800, 600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 30)
    font1 = ImageFont.truetype("arial.ttf", 25)

    # Define starting position
    x, y = 10, 10
    # You can also get feature importances
    feature_importances = RandomForest_Classifier_model.feature_importances_
    print('Feature Importances:')
    draw.text((x, y),'Feature Importances:', fill="black", font=font)
    y += 40 
    for feature, importance in zip(X_train_new.columns, feature_importances):
        draw.text((x, y), str(feature)+': '+str(importance), fill="black", font=font1)
        y += 30  # Move down for the next iteration
    image.save(path)

@app.route('/home')
def home():
    username = session.get("username")
    cursor.execute('SELECT * FROM register WHERE username = %s',(username))
    row = cursor.fetchone()
    
    Studdff=Student_df[Student_df['Student_id'] == int(row[0])][['Software Engineering score','Machine Learningscore','DBMS score','Elective 1 score','Elective 2 score']]
    column_headers = list(Studdff.columns.values)
    col_vals=Studdff.values.tolist()[0]   
    
    Studdff=df_train.iloc[int(row[0])-1].drop(["Pass/Fail",'Current Sem percentage','Student_id'])
    var2 = [Studdff]    
    new_predictions1 = KNN_Classifier_model.predict(var2) 
    
    print(column_headers)
    print(col_vals)
    
    # data = [{'x': i, 'y': y} for i, (x, y) in enumerate(zip(column_headers, col_vals))]
    
    new_predictions2 = KNN_Regressor_model.predict(var2)
    print(new_predictions2)
    
    encoded_df1 = df_train[:999]
    X_test=encoded_df1.drop(["Pass/Fail",'Current Sem percentage','Student_id'],axis=1)
    var1=X_test
    new_predictions = KNN_Classifier_model.predict(var1)
    matrixresult = confusion_matrix(encoded_df1["Pass/Fail"],new_predictions, labels=[1,0])
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrixresult, annot=True, fmt="d", cmap="Blues", xticklabels=["1", "0"], yticklabels=["1", "0"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix Of KNN")     
    savepath = getRandomfilepath()    
    plt.savefig(savepath)
    
    var4=df_train.drop(["Pass/Fail",'Current Sem percentage','Student_id'],axis=1)
    scatterplotsavepath = getRandomfilepath()
    plotScatterPlot(var4,df_train["Current Sem percentage"], scatterplotsavepath)
    
    decision_boundary_path = getRandomfilepath() 
    decision_boundary(decision_boundary_path)
    
    if new_predictions1[0] == 0:
        status = 'Fail'
    else:
        status = 'Pass'
    
    return render_template('index.html',userrow=row,confusion_matrix=savepath,column_headers=column_headers,col_vals=col_vals,statusofstud=status,percentofstud=round(new_predictions2[0], 2),scatter_plot=scatterplotsavepath,decision_boundary=decision_boundary_path)


@app.route('/home1')
def home1():
    cursor.execute('SELECT * FROM register WHERE id = %s',(1))
    row = cursor.fetchone()
    
    Studdff=Student_df[Student_df['Student_id'] == int(row[0])][['Software Engineering score','Machine Learningscore','DBMS score','Elective 1 score','Elective 2 score']]
    column_headers = list(Studdff.columns.values)
    col_vals=Studdff.values.tolist()[0]   
    
    Studdff=df_train.iloc[int(row[0])-1].drop(["Pass/Fail",'Current Sem percentage','Student_id'])
    var2 = [Studdff]    
    new_predictions1 = KNN_Classifier_model.predict(var2) 
    
    print(column_headers)
    print(col_vals)
    
    # data = [{'x': i, 'y': y} for i, (x, y) in enumerate(zip(column_headers, col_vals))]
    
    new_predictions2 = KNN_Regressor_model.predict(var2)
    print(new_predictions2)
    
    encoded_df1 = df_train[:999]
    X_test=encoded_df1.drop(["Pass/Fail",'Current Sem percentage','Student_id'],axis=1)
    var1=X_test
    new_predictions = KNN_Classifier_model.predict(var1)
    matrixresult = confusion_matrix(encoded_df1["Pass/Fail"],new_predictions, labels=[1,0])
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrixresult, annot=True, fmt="d", cmap="Blues", xticklabels=["1", "0"], yticklabels=["1", "0"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix Of KNN")     
    savepath = getRandomfilepath()    
    plt.savefig(savepath)
    
    var4=df_train.drop(["Pass/Fail",'Current Sem percentage','Student_id'],axis=1)
    scatterplotsavepath = getRandomfilepath()
    plotScatterPlot(var4,df_train["Current Sem percentage"], scatterplotsavepath)
    
    decision_boundary_path = getRandomfilepath() 
    decision_boundary(decision_boundary_path)
    
    if new_predictions1[0] == 0:
        status = 'Fail'
    else:
        status = 'Pass'       

    cursor.execute('SELECT id FROM register')
    idrow = cursor.fetchall()  
    flat_list = [item for sublist in idrow for item in sublist]
    return render_template('index1.html',userrow=row,confusion_matrix=savepath,column_headers=column_headers,col_vals=col_vals,statusofstud=status,percentofstud=round(new_predictions2[0], 2),scatter_plot=scatterplotsavepath,decision_boundary=decision_boundary_path,flat_list=flat_list)

@app.route('/admin')
def admin():
    return render_template('admin.html')



@app.route('/profile',methods=['POST','GET'])
def profile():
    username = session.get("username")
    student_id = session.get("student_id")
    
    cursor.execute('SELECT * FROM register WHERE username = %s',(username))
    row = cursor.fetchone()
    if request.method == "POST":
        details = request.form
       
        name = details['name']
        email = details['email']
        password1 = details['password']
        mobile= details['mobno']
        address = details['address']
        
        print("Update")
        sql1 = "UPDATE register SET fname = %s,email = %s,password = %s,mobileno = %s,address = %s WHERE username = %s AND id = %s ;"
        val1 = (name,email,password1, mobile, address, username, student_id)
        cursor.execute(sql1,val1)
        con.commit()
        print("username",username)
        message = "Upgrade your profile successfully."+" "+"username is-"+name
       
        return render_template('profile.html',message=message,userrow=row)  
    return render_template('profile.html',userrow=row)

@app.route('/forgotpass')
def addhotel():
    return render_template('forgotpassword.html')

@app.route('/userregistration',methods=['POST','GET'])
def userregistration():
    if request.method == "POST":
        details = request.form
        Stuid = details['Stuid']
        name = details['name']
        username = details['username']
        email = details['email']
        password1 = details['password']
        mobile= details['mobno']
        address = details['address']
        
        cursor.execute('SELECT * FROM register WHERE username = %s',(username))
        count = cursor.rowcount
        
        if count > 0: 
            return "User already exist !"
        else:
            sql2  = "INSERT INTO register(id,username,email,password,mobileno,address,fname) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            val2 = (int(Stuid), str(username), str(email), str(password1), str(mobile), str(address), str(name))
            cursor.execute(sql2,val2) 
            con.commit()
            print("username",username)
           
            return render_template('login.html')  
        
@app.route('/userlogin', methods=["GET","POST"])
def userlogin():
    msg = ''
    if request.method == "POST":      
        username = request.form.get("username")
        print ("username",username)
        password = request.form.get("password")       
        cursor.execute('SELECT * FROM register WHERE username = %s AND password = %s' , (username, password))
        result = cursor.fetchone()
        print ("result",result)
        if result:
            session['student_id'] = result[0]
            session['username'] = result[1]
            return redirect(url_for('home'))
        else:
            msg = 'Incorrect username/password!'
            return msg
      
    return render_template('register.html')

@app.route('/teacherlogin1', methods=["GET","POST"])
def teacherlogin1():
    msg = ''
    if request.method == "POST":      
        username = request.form.get("username")
        print ("username",username)
        password = request.form.get("password")  
        if username == 'admin' and password == 'admin':
            return redirect(url_for('home1'))
        else:
            msg = 'Incorrect username/password!'
            return msg
      
    return render_template('teacherlogin.html')

@app.route('/forgotpass', methods=["GET","POST"])
def forgotpass():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        cpassword = request.form.get("cpassword")
        
        cursor.execute('SELECT * FROM register WHERE username = %s',(username))
        count = cursor.rowcount
        
        if count > 0: 
            if password == cpassword:                
                sql_update_query = "Update register set password = %s where username = %s"
                input_data = (password, username)
                cursor.execute(sql_update_query, input_data)
                con.commit()
                return redirect(url_for('login'))
            else:
                return "Password and Confirm password does not match !"
        else:
            return "User not exist !"
        
@app.route('/getLoadData', methods=["GET","POST"])
def getLoadData():
    if request.method == "POST":
        samplesize = request.form.get("value1")
        algorithm = request.form.get("value2")       

        encoded_df1 = df_train[:int(samplesize)]
        X_test=encoded_df1.drop(["Pass/Fail",'Current Sem percentage','Student_id'],axis=1)
        var1=X_test
        y=encoded_df1['Pass/Fail']
        
        username = session.get("username")
        cursor.execute('SELECT * FROM register WHERE username = %s',(username))
        row = cursor.fetchone() 
        
        Studdff=df_train.iloc[int(row[0])-1].drop(["Pass/Fail",'Current Sem percentage','Student_id'])
        var2 = [Studdff]
        
        if algorithm == 'Knn':
            new_predictions = KNN_Classifier_model.predict(var1)  
            new_predictions1 = KNN_Classifier_model.predict(var2)   
            new_predictions3 = KNN_Regressor_model.predict(var2) 
            
            image1path = getRandomfilepath()    
            plotScatterPlot(var1,df_train["Current Sem percentage"][:int(samplesize)], image1path)
            image2path = getRandomfilepath()     
            decision_boundary(image2path)
            
            htmlfilepath=''
        elif algorithm == 'Logistic regression':
            new_predictions = Logistic_Regression_Classifier_model.predict(var1)  
            new_predictions1 = Logistic_Regression_Classifier_model.predict(var2)  
            new_predictions3 = Logistic_Regression_Regressor_model.predict(var2)   
            
            image1path = getRandomfilepath()    
            plot_lr_featurebar(image1path,var1.columns)
            image2path = getRandomfilepath()    
            plot_roc_curve(image2path,int(samplesize))            
            
            htmlfilepath = getRandomhtmlfilepath()             
            y_true = np.array(new_predictions)
            y_scores = np.array(df_train["Pass/Fail"][:int(samplesize)].values)
            
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = roc_auc_score(y_true, y_scores)
            
            # Create the ROC curve figure using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc:.2f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
            
            fig.update_layout(
                title='ROC Curve',
                xaxis=dict(title='False Positive Rate'),
                yaxis=dict(title='True Positive Rate'),
            )
            
            # Save the figure to an HTML file
            fig.write_html(htmlfilepath)
        elif algorithm == 'Decision tree':
            new_predictions = DecisionTree_Classifier_model.predict(var1)    
            new_predictions1 = DecisionTree_Classifier_model.predict(var2)   
            new_predictions3 = DecisionTree_Regressor_model.predict(var2)   
            
            image1path = getRandomfilepath()    
            plot_dt_featurebar(image1path,var1)
            image2path = getRandomfilepath()    
            plot_tree_growth(image2path,var1)
            
            htmlfilepath=''
            
            
        elif algorithm == 'SVM':            
            new_predictions = SVC_linear_Classifier_model.predict(var1)
            new_predictions1 = SVC_linear_Classifier_model.predict(var2)   
            new_predictions3 = SVC_Regressor_model.predict(var2)         
            
            image1path = getRandomfilepath()    
            plot_svm_decisionboundary(image1path,var1,y)
            image2path = getRandomfilepath()    
            plot_svm_hyperplane(image2path,var1,y)
            
            htmlfilepath=''
            
        elif algorithm == 'Random Forest':
            new_predictions = RandomForest_Classifier_model.predict(var1)
            new_predictions1 = RandomForest_Classifier_model.predict(var2)   
            new_predictions3 = RandomForest_Regressor_model.predict(var2)  
            
            image1path = getRandomfilepath()    
            plot_feature_importances_graph(image1path,var1)
            image2path = getRandomfilepath()    
            plot_feature_importances_image(image2path,var1) 
            
            htmlfilepath=''         
            
        matrixresult = confusion_matrix(encoded_df1["Pass/Fail"],new_predictions, labels=[1,0])
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrixresult, annot=True, fmt="d", cmap="Blues", xticklabels=["1", "0"], yticklabels=["1", "0"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix of "+algorithm)           
        savepath = getRandomfilepath() 
        plt.savefig(savepath)
        
        percentage = round(new_predictions3[0], 2)
        
        if new_predictions1[0] == 0:
            status = 'Fail'
        else:
            status = 'Pass'
        
        jsonObject = json.dumps([savepath,status,str(percentage),image1path,image2path,htmlfilepath])
        
        return jsonObject
    
@app.route('/getLoadData1', methods=["GET","POST"])
def getLoadData1():
    if request.method == "POST":
        samplesize = request.form.get("value1")
        algorithm = request.form.get("value2")       

        encoded_df1 = df_train[:int(samplesize)]
        X_test=encoded_df1.drop(["Pass/Fail",'Current Sem percentage','Student_id'],axis=1)
        var1=X_test
        y=encoded_df1['Pass/Fail']
        if algorithm == 'Knn':
            new_predictions = KNN_Classifier_model.predict(var1) 
            
            image1path = getRandomfilepath()    
            plotScatterPlot(var1,df_train["Current Sem percentage"][:int(samplesize)], image1path)
            image2path = getRandomfilepath()     
            decision_boundary(image2path)    
            
            htmlfilepath=''       
        elif algorithm == 'Logistic regression':
            new_predictions = Logistic_Regression_Classifier_model.predict(var1)    
            
            image1path = getRandomfilepath()    
            plot_lr_featurebar(image1path,var1.columns)
            image2path = getRandomfilepath()    
            plot_roc_curve(image2path,int(samplesize))            
            
            htmlfilepath = getRandomhtmlfilepath() 
            
            # Sample data (you should replace this with your actual data)
            y_true = np.array(new_predictions)
            y_scores = np.array(df_train["Pass/Fail"][:int(samplesize)].values)
            
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = roc_auc_score(y_true, y_scores)
            
            # Create the ROC curve figure using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc:.2f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
            
            fig.update_layout(
                title='ROC Curve',
                xaxis=dict(title='False Positive Rate'),
                yaxis=dict(title='True Positive Rate'),
            )
            
            # Save the figure to an HTML file
            fig.write_html(htmlfilepath)
        elif algorithm == 'Decision tree':
            new_predictions = DecisionTree_Classifier_model.predict(var1)   
            
            image1path = getRandomfilepath()    
            plot_dt_featurebar(image1path,var1)
            image2path = getRandomfilepath()    
            plot_tree_growth(image2path,var1) 
            
            htmlfilepath=''        
        elif algorithm == 'SVM':            
            new_predictions = SVC_linear_Classifier_model.predict(var1)     
            
            image1path = getRandomfilepath()    
            plot_svm_decisionboundary(image1path,var1,y)
            image2path = getRandomfilepath()    
            plot_svm_hyperplane(image2path,var1,y)
            
            htmlfilepath=''
        elif algorithm == 'Random Forest':
            new_predictions = RandomForest_Classifier_model.predict(var1)
            
            image1path = getRandomfilepath()    
            plot_feature_importances_graph(image1path,var1)
            image2path = getRandomfilepath()    
            plot_feature_importances_image(image2path,var1) 
            
            htmlfilepath='' 
        matrixresult = confusion_matrix(encoded_df1["Pass/Fail"],new_predictions, labels=[1,0])
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrixresult, annot=True, fmt="d", cmap="Blues", xticklabels=["1", "0"], yticklabels=["1", "0"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix of "+algorithm)           
        savepath = getRandomfilepath() 
        plt.savefig(savepath)
        
        
        jsonObject = json.dumps([savepath,image1path,image2path,htmlfilepath])
        
        return jsonObject
        
        # return savepath     
        
@app.route('/getLoadData2', methods=["GET","POST"])
def getLoadData2():
    if request.method == "POST":
        samplesize = request.form.get("value1")
        algorithm = request.form.get("value2")
        stuid = request.form.get("value3")       

        encoded_df1 = df_train[:int(samplesize)]
        X_test=encoded_df1.drop(["Pass/Fail",'Current Sem percentage','Student_id'],axis=1)
        var1=X_test
        y=encoded_df1['Pass/Fail']
        
        cursor.execute('SELECT * FROM register WHERE id = %s',(stuid))
        row = cursor.fetchone() 
        
        Studdff=df_train.iloc[int(row[0])-1].drop(["Pass/Fail",'Current Sem percentage','Student_id'])
        var2 = [Studdff]
        
        if algorithm == 'Knn':
            new_predictions = KNN_Classifier_model.predict(var1)  
            new_predictions1 = KNN_Classifier_model.predict(var2)   
            new_predictions3 = KNN_Regressor_model.predict(var2) 
            
            image1path = getRandomfilepath()    
            plotScatterPlot(var1,df_train["Current Sem percentage"][:int(samplesize)], image1path)
            image2path = getRandomfilepath()     
            decision_boundary(image2path)
            
            htmlfilepath=''
        elif algorithm == 'Logistic regression':
            new_predictions = Logistic_Regression_Classifier_model.predict(var1)  
            new_predictions1 = Logistic_Regression_Classifier_model.predict(var2)  
            new_predictions3 = Logistic_Regression_Regressor_model.predict(var2)    
            
            image1path = getRandomfilepath()    
            plot_lr_featurebar(image1path,var1.columns)
            image2path = getRandomfilepath()    
            plot_roc_curve(image2path,int(samplesize))                        
            
            htmlfilepath = getRandomhtmlfilepath() 
            
            # Sample data (you should replace this with your actual data)
            y_true = np.array(new_predictions)
            y_scores = np.array(df_train["Pass/Fail"][:int(samplesize)].values)
            
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = roc_auc_score(y_true, y_scores)
            
            # Create the ROC curve figure using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc:.2f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
            
            fig.update_layout(
                title='ROC Curve',
                xaxis=dict(title='False Positive Rate'),
                yaxis=dict(title='True Positive Rate'),
            )
            
            # Save the figure to an HTML file
            fig.write_html(htmlfilepath)
        elif algorithm == 'Decision tree':
            new_predictions = DecisionTree_Classifier_model.predict(var1)    
            new_predictions1 = DecisionTree_Classifier_model.predict(var2)   
            new_predictions3 = DecisionTree_Regressor_model.predict(var2)   
            
            image1path = getRandomfilepath()    
            plot_dt_featurebar(image1path,var1)
            image2path = getRandomfilepath()    
            plot_tree_growth(image2path,var1)
            
            htmlfilepath=''
            
            
        elif algorithm == 'SVM':            
            new_predictions = SVC_linear_Classifier_model.predict(var1)
            new_predictions1 = SVC_linear_Classifier_model.predict(var2)   
            new_predictions3 = SVC_Regressor_model.predict(var2)         
            
            image1path = getRandomfilepath()    
            plot_svm_decisionboundary(image1path,var1,y)
            image2path = getRandomfilepath()    
            plot_svm_hyperplane(image2path,var1,y)
            
            htmlfilepath=''
            
        elif algorithm == 'Random Forest':
            new_predictions = RandomForest_Classifier_model.predict(var1)
            new_predictions1 = RandomForest_Classifier_model.predict(var2)   
            new_predictions3 = RandomForest_Regressor_model.predict(var2)  
            
            image1path = getRandomfilepath()    
            plot_feature_importances_graph(image1path,var1)
            image2path = getRandomfilepath()    
            plot_feature_importances_image(image2path,var1) 
            
            htmlfilepath=''         
            
        matrixresult = confusion_matrix(encoded_df1["Pass/Fail"],new_predictions, labels=[1,0])
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrixresult, annot=True, fmt="d", cmap="Blues", xticklabels=["1", "0"], yticklabels=["1", "0"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix of "+algorithm)           
        savepath = getRandomfilepath() 
        plt.savefig(savepath)
        
        print('new_predictions3')
        print(new_predictions3)
        
        percentage = round(new_predictions3[0], 2)
        
        if new_predictions1[0] == 0:
            status = 'Fail'
        else:
            status = 'Pass'
        
        jsonObject = json.dumps([savepath,status,str(percentage),image1path,image2path,row,htmlfilepath])
        
        return jsonObject
    
@app.route('/getLoadData3', methods=["GET","POST"])
def getLoadData3():
    if request.method == "POST":
        samplesize = request.form.get("value1")
        algorithm = request.form.get("value2") 
        stuid = request.form.get("value3")  
        
        print(stuid)

        encoded_df1 = df_train[:int(samplesize)]
        X_test=encoded_df1.drop(["Pass/Fail",'Current Sem percentage','Student_id'],axis=1)
        var1=X_test
        y=encoded_df1['Pass/Fail']
        if algorithm == 'Knn':
            new_predictions = KNN_Classifier_model.predict(var1) 
            
            image1path = getRandomfilepath()    
            plotScatterPlot(var1,df_train["Current Sem percentage"][:int(samplesize)], image1path)
            image2path = getRandomfilepath()     
            decision_boundary(image2path)     
            
            htmlfilepath=''      
        elif algorithm == 'Logistic regression':
            new_predictions = Logistic_Regression_Classifier_model.predict(var1)    
            
            image1path = getRandomfilepath()    
            plot_lr_featurebar(image1path,var1.columns)
            image2path = getRandomfilepath()    
            plot_roc_curve(image2path,int(samplesize))            
            
            htmlfilepath = getRandomhtmlfilepath() 
            
            # Sample data (you should replace this with your actual data)
            y_true = np.array(new_predictions)
            y_scores = np.array(df_train["Pass/Fail"][:int(samplesize)].values)
            
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = roc_auc_score(y_true, y_scores)
            
            # Create the ROC curve figure using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc:.2f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
            
            fig.update_layout(
                title='ROC Curve',
                xaxis=dict(title='False Positive Rate'),
                yaxis=dict(title='True Positive Rate'),
            )
            
            # Save the figure to an HTML file
            fig.write_html(htmlfilepath)
        elif algorithm == 'Decision tree':
            new_predictions = DecisionTree_Classifier_model.predict(var1)   
            
            image1path = getRandomfilepath()    
            plot_dt_featurebar(image1path,var1)
            image2path = getRandomfilepath()    
            plot_tree_growth(image2path,var1)  
            
            htmlfilepath=''       
        elif algorithm == 'SVM':            
            new_predictions = SVC_linear_Classifier_model.predict(var1)     
            
            image1path = getRandomfilepath()    
            plot_svm_decisionboundary(image1path,var1,y)
            image2path = getRandomfilepath()    
            plot_svm_hyperplane(image2path,var1,y)
            
            htmlfilepath=''
        elif algorithm == 'Random Forest':
            new_predictions = RandomForest_Classifier_model.predict(var1)
            
            image1path = getRandomfilepath()    
            plot_feature_importances_graph(image1path,var1)
            image2path = getRandomfilepath()    
            plot_feature_importances_image(image2path,var1)  
            
            htmlfilepath=''
        matrixresult = confusion_matrix(encoded_df1["Pass/Fail"],new_predictions, labels=[1,0])
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrixresult, annot=True, fmt="d", cmap="Blues", xticklabels=["1", "0"], yticklabels=["1", "0"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix of "+algorithm)           
        savepath = getRandomfilepath() 
        plt.savefig(savepath)
        
        
        jsonObject = json.dumps([savepath,image1path,image2path,htmlfilepath])
        
        return jsonObject 
        
@app.route('/uploadMarks', methods=["GET","POST"])
def uploadMarks():
    if request.method == "POST":
        gender = request.form.get("gender")
        levelofedu = request.form.get("levelofedu")
        testcourse = request.form.get("testcourse")
        SEscore = request.form.get("SEscore")
        MLscore = request.form.get("MLscore")
        DBMSscore = request.form.get("DBMSscore")
        E1score = request.form.get("E1score")
        E2score = request.form.get("E2score")
        attendance = request.form.get("attendance")
        stuhour = request.form.get("stuhour")
        assigmarks = request.form.get("assigmarks")
        socialmedia = request.form.get("socialmedia")
        extaactivity = request.form.get("extaactivity")
        presemmarks = request.form.get("presemmarks")
        
        data = {
            "gender": [gender],
            "parental level of education": [levelofedu],
            "test preparation course": [testcourse],
            "Software Engineering score": [int(SEscore)],
            "Machine Learningscore": [int(MLscore)],
            "DBMS score": [int(DBMSscore)],
            "Elective 1 score": [int(E1score)],
            "Elective 2 score": [int(E2score)],
            "Attendance": [float(attendance)],
            "Study hours/Day": [int(stuhour)],
            "Assignment marks": [int(assigmarks)],
            "Time spent on social media in mins": [int(socialmedia)],
            "Participation in extra cirricular activities": [extaactivity],
            "Previous sem percentage": [int(presemmarks)]
        }

        df = pd.DataFrame(data)
        
        dummydf = Student_df.drop(["Pass/Fail",'Current Sem percentage','Student_id'], axis=1)

        concatenated_df = pd.concat([df, dummydf], ignore_index=True)
        encoded_df, label_encoders = dynamic_label_encode(concatenated_df)
        
        to_scale = [col for col in encoded_df.columns if encoded_df[col].max()>1]
        scaler = RobustScaler()
        scaled =scaler.fit_transform(encoded_df[to_scale])
        scaled = pd.DataFrame(scaled, columns=to_scale)
        
        # replace original columns with scaled columns
        for col in scaled:
            encoded_df[col] = scaled[col]

        # Keep only the part that was originally in df1
        result_df = encoded_df.iloc[:len(df), :]
        
        status = DecisionTree_Classifier_model.predict(result_df)
        marks = DecisionTree_Regressor_model.predict(result_df)
        
        if status[0] == 0:
            passorfail = 'Fail'
        else:
            passorfail = 'Pass'
            
        print(passorfail)
        print(marks)
        
        return str(passorfail)+'|'+str(round(marks[0], 2))
    return render_template('uploadMarks.html') 
        
 
if __name__ == "__main__":
    app.run("0.0.0.0")    
    # app.run(debug=True)