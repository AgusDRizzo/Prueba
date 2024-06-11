import sklearn
import pickle
import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
model_file = 'model_C=1.0.bin'
 
 
with open(model_file, 'rb') as f_in:
     model_rl = pickle.load(f_in)
 
 
def main():
    image3 = Image.open('Logo-4.png')
    st.image(image3)
    #st.markdown("<h1 style='text-align: center; color: grey;'>RetainIT</h1>", unsafe_allow_html=True)
    st.title("Predicting Customer Churn")
    
    #image = Image.open('images/icone.png') 
    image2 = Image.open('Logo-3.tif')
    #st.image(image,use_column_width=False) 
    st.sidebar.image(image2)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    
    
    if add_selectbox == 'Online':
        Age = st.number_input('Age:', min_value=19, max_value=80, value=19)
        Tenure = st.number_input(' Tenure in months:', min_value=0, max_value=72, value=0 )
        refunds = st.number_input(' Total refunds:', min_value=0, max_value=51, value=0)
        dependents = st.number_input(' Customer has  dependents:', min_value=0, max_value=9, value=0)
        phoneservice = st.selectbox(' Customer has phoneservice:', ['Yes', 'No'])
        multiplelines = st.selectbox(' Customer has multiple lines:', ['Yes', 'No'])
        internetservice= st.selectbox(' Customer has internet service:', ['Yes', 'No'])
        deviceprotection = st.selectbox(' Customer has device protection:', ['Yes', 'No'])
        monthlycharges= st.number_input('Monthly charges :', min_value=0, max_value=119, value=0)
        longdistance = st.number_input('Total Long distance charges :', min_value=0, max_value=3600, value=0)
        offer = st.selectbox(' Customer has accepted offer:', ['Yes', 'No'])
        north = st.selectbox(' Customer lives north:', ['Yes', 'No'])
        output= ""
        output_prob = ""
        input_dict={
                "Age":Age,
                "Number of Dependents": dependents,
                "Tenure in Months": Tenure,
                "Phone Service": phoneservice,
                "Multiple Lines": multiplelines,
                "Device Protection Plan": deviceprotection,
                "Monthly Charge": monthlycharges,
                "Total Refunds": refunds,
                "Total Long Distance Charges": longdistance,
                "Offer": offer,
                "Internet Service": internetservice,
                "North": north
            }
        if st.button("Predict"):
            
            X = pd.DataFrame.from_dict([input_dict])
            X[['Phone Service', 'Multiple Lines','Device Protection Plan', 'Internet Service', 'Offer', 'North']] = X[['Phone Service', 'Multiple Lines','Device Protection Plan', 'Internet Service', 'Offer', 'North']].replace({'Yes': 1, 'No': 0})
            y_pred = model_rl.predict(X)
            churn = bool(y_pred)
            output_prob = model_rl.predict_proba(X).max().round(2)
            output = churn
  
        st.success('Stay: {0}, Risk Score: {1}'.format(output, output_prob))
 
    if add_selectbox == 'Batch':
 

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            X = dv.transform([data])
            y_pred = model_rl.predict(X)
            churn = y_pred 
            churn = bool(churn)
            st.write(churn)
 
 
if __name__ == '__main__':
    main()