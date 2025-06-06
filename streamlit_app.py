"""
Created on Sun Jun  1 20:52:02 2025

@author: Lazzie
"""

#import numpy as np
import pickle 
import streamlit as st
import pandas as pd 
import shap
import matplotlib.pyplot as plt
#import numpy as np

#load saved model
#XG BOOST
#xgb_model = pickle.load(open('Models_/XGBoost_model_pkl','rb'))
#LightGBM
lgbm_model = pickle.load(open('Models_/LightGBM_model_pkl','rb'))

# Initialize session state
if 'client_data' not in st.session_state:
    st.session_state.client_data = None
if 'last_model' not in st.session_state:
    st.session_state.last_model = None

# Cache SHAP explainer
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

# Default prediction function (no SHAP)
def prediction_default(input_data, model, threshold=0.5):
    feature_names =[
    'GENDER_SUITE_ENCODED','EDUCATION_OCCUPATION_ENCODED',
    'REL_EDUCATION_ENCODED','NAME_HOUSING_TYPE_ENCODED',

    'AGE_YEARS', 'OWN_CAR_AGE', 'EMPLOYED_TO_AGE_RATIO','YEARS_EMPLOYED',
 #PREV STAFF
     'PREV_APP_STATUS_REFUSED','PREV_APP_RATIO_PREV_APP_STATUS_UNUSED_OFFER',
    'PREV_APK_AMT_CREDIT_APPLICATION_RATIO_MEAN', 'PREV_APK_AMT_CREDIT_MEAN', 'PREV_APK_AMT_DECLINED_MIN',
 'PREV_APK_AMT_INTEREST_MAX','PREV_APK_AMT_CREDIT_MIN','PREV_APK_AMT_ANNUITY_MIN','PREV_AMT_PAYMENT_MIN','PREV_AMT_INTSALMENT_MAX',
    #products
    'COUNT_PREV_PROD_Card_Street',
    #instalments
    'NUM_INSTALMENTS_EARLY_PAYMENTS','NUM_INSTALMENTS_LATE_PAYMENTS','NUM_INSTALMENT_PARTIAL_PAYMENTS',

    #Alternstive
'AMT_ANNUITY','AMT_CREDIT','ANNUITY_CREDIT_RATIO','CNT_FAM_MEMBERS',  'YEARS_DETAILS_CHANGE_SUM',   
'WEIGHTED_EXT_SOURCE','EXT_SOURCE_MEAN', 
 'OBS_DEF_30_MUL','OBS_DEF_60_MUL','DEF_30_CREDIT_RATIO','DEF_60_CREDIT_RATIO',
 'CHILDREN_INCOME_RATIO','CREDIT_INCOME_RATIO','ANNUITY_INCOME_RATIO',
'REGION_RATING_MUL','REGION_POPULATION_RELATIVE','REGIONS_RATING_INCOME_MUL','FLAG_DOCUMENT_3'
]
    #input to numpy array
   # input_data_numpy = np.array(input_data).astype(np.float32)

    #reshape the input 
    #input_reshaped= input_data_numpy.reshape(1,-1)
    input_df = pd.DataFrame([input_data], columns = feature_names)
    explainer = get_explainer(model)
    shap_values = explainer(input_df)

    #predict using XG Boost
    #prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0][1]
    prediction = 1 if proba >= threshold else 0
    proba_percentage = round(proba*100,0)
    print(prediction) 
    if prediction == 1:
        return '❌ Will not pay back the loan',proba_percentage,shap_values
    else:
        return '✅ Will pay back the loan',proba_percentage,shap_values
     # Save input for SHAP
    st.session_state.client_data = input_df
    st.session_state.last_model = model
    
# SHAP computation
def compute_shap_values():
    if st.session_state.client_data is not None and st.session_state.last_model is not None:
        explainer = get_explainer(st.session_state.last_model)
        shap_values = explainer(st.session_state.client_data)
        
        st.subheader("🔍 SHAP Value Summary")
        shap.plots.waterfall(shap_values[0], max_display=10)
        st.pyplot(bbox_inches='tight')
    else:
        st.warning("⚠️ Please make a prediction first.")
#Label Encoding  Mappings
Housing_mapping = {
    'Co-op apartment': 0,
    'House_or_apartment': 1,
    'Municipal apartment': 2,
    ' Office apartment': 3,
    'Rented apartment' : 4,
    'With parents':5
}

education_occupation = {
'Academic degree_Accountants' :0
,'Academic degree_Cooking staff' :1
,'Academic degree_Core staff' :2
,'Academic degree_Drivers' :3
,'Academic degree_Laborers' :4
,'Academic degree_Managers' :5
,'Academic degree_Private service staff' :6
,'Academic degree_Sales staff' :7
,'Academic degree_XNA' :8
,'Academic degree_IT staff' :82
,'Higher education_Accountants' :9
,'Higher education_Cleaning staff' :10
,'Higher education_Cooking staff' :11
,'Higher education_Core staff' :12
,'Higher education_Drivers' :13
,'Higher education_HR staff' :14
,'Higher education_High skill tech staff' :15
,'Higher education_IT staff' :16
,'Higher education_Laborers' :17
,'Higher education_Low-skill Laborers' :18
,'Higher education_Managers' :19
,'Higher education_Medicine staff' :20
,'Higher education_Private service staff' :21
,'Higher education_Realty agents' :22
,'Higher education_Sales staff' :23
,'Higher education_Secretaries' :24
,'Higher education_Security staff' :25
,'Higher education_Waiters/barmen staff' :26
,'Higher education_XNA' :27
,'Incomplete higher_Accountants' :28
,'Incomplete higher_Cleaning staff' :29
,'Incomplete higher_Cooking staff' :30
,'Incomplete higher_Core staff' :31
,'Incomplete higher_Drivers' :32
,'Incomplete higher_HR staff' :33
,'Incomplete higher_High skill tech staff' :34
,'Incomplete higher_IT staff' :35
,'Incomplete higher_Laborers' :36
,'Incomplete higher_Low-skill Laborers' :37
,'Incomplete higher_Managers' :38
,'Incomplete higher_Medicine staff' :39
,'Incomplete higher_Private service staff' :40
,'Incomplete higher_Realty agents' :41
,'Incomplete higher_Sales staff' :42
,'Incomplete higher_Secretaries' :43
,'Incomplete higher_Security staff' :44
,'Incomplete higher_Waiters/barmen staff' :45
,'Incomplete higher_XNA' :46
,'Lower secondary_Accountants' :47
,'Lower secondary_Cleaning staff' :48
,'Lower secondary_Cooking staff' :49
,'Lower secondary_Core staff' :50
,'Lower secondary_Drivers' :51
,'Lower secondary_High skill tech staff' :52
,'Lower secondary_Laborers' :53
,'Lower secondary_Low-skill Laborers' :54
,'Lower secondary_Managers' :55
,'Lower secondary_Medicine staff' :56
,'Lower secondary_Private service staff' :57
,'Lower secondary_Sales staff' :58
,'Lower secondary_Security staff' :59
,'Lower secondary_Waiters/barmen staff' :60
,'Lower secondary_XNA' :61
,'Secondary_Accountants' :62
,'Secondary_Cleaning staff' :63
,'Secondary_Cooking staff' :64
,'Secondary_Core staff' :65
,'Secondary_Drivers' :66
,'Secondary_HR staff' :67
,'Secondary_High skill tech staff' :68
,'Secondary_IT staff' :69
,'Secondary_Laborers' :70
,'Secondary_Low-skill Laborers' :71
,'Secondary_Managers' :72
,'Secondary_Medicine staff' :73
,'Secondary_Private service staff' :74
,'Secondary_Realty agents' :75
,'Secondary_Sales staff' :76
,'Secondary_Secretaries' :77
,'Secondary_Security staff' :78
,'Secondary_Waiters/barmen staff' :79,
'Secondary_XNA' :80
}

#Mapping for GENDER_SUITE:
gender_suite = {
'F_Children' : 0
,'F_Family' : 1
,'F_Group of people' : 2
,'F_Other_A' : 3
,'F_Other_B' : 4
,'F_Spouse_partner' : 5
,'F_Unaccompanied' : 6
,'F_XNA' : 7
,'M_Children' : 8
,'M_Family' : 9
,'M_Group of people' : 10
,'M_Other_A' : 11
,'M_Other_B' : 12
,'M_Spouse_partner' : 13
,'M_Unaccompanied' : 14
,'M_XNA' : 15}

#Mapping for REL_EDUCATION:
rel_education = {
'In_Relationship_Academic degree' : 0
,'In_Relationship_Higher education' : 1
,'In_Relationship_Incomplete higher' : 2
,'In_Relationship_Lower secondary' : 3
,'In_Relationship_Secondary' : 4
,'Previously Married_Academic degree' : 5
,'Previously Married_Higher education' : 6
,'Previously Married_Incomplete higher' : 7
,'Previously Married_Lower secondary' : 8
,'Previously Married_Secondary' : 9
,'Single_Academic degree' : 10
,'Single_Higher education' : 11
,'Single_Incomplete higher' : 12
,'Single_Lower secondary' : 13
,'Single_Secondary' : 14}

#gender mapping 
gender_mapping = {
    'M': 1,
    'F':0 }
#Education
edu = ['Academic degree',
 'Higher education',
  'Incomplete higher',
  'Lower secondary',
  'Secondary']

#occupation
occu =['Laborers','Cooking staff','Sales staff','XNA','Managers'
,'Private service staff','Core staff','High skill tech staff'
,'Medicine staff','Drivers','Security staff','Low-skill Laborers'
,'Accountants','Cleaning staff','Realty agents','Secretaries'
,'Waiters/barmen staff','IT staff','HR staff']
  
#Relationship
rel = ['Single', 'Previously Married', 'In_Relationship']
#Gender 
gender = ['M', 'F']

#Suite 
suite = ['Unaccompanied', 'Family', 'Spouse_partner', 'Other_A', 'Children',
       'Other_B', 'XNA', 'Group of people']
def main():
    #Title
    #st.title('Default Prediction App')
    html_temp = """
    <div style = 'background-color: #FF4B4B; padding: 10px'> 
    <h2 style = 'color: white; text-align: center;'> 🏦 Default Risk Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    #st.markdown("<h1 style='color:tomato;'>🏦 Default Risk Prediction App</h1>", unsafe_allow_html=True)
   # st.markdown("<h3 style='color:orange;'>Model based on behavioural and alternative data</h3>", unsafe_allow_html=True)
   # st.subheader('Model based on behavioural and alternative data', divider = 'rainbow')
   # st.markdown('**Select the model of your choice**')
    tab1,tab2= st.tabs(['Model 1 - XGBoost','Model 2 - LightGBM'])
    
    with tab1:
        run_prediction_tab(xgb_model)
    with tab2:
        run_prediction_tab(lgbm_model)
   # st.markdown('Please fill in the blank spaces with client information')
        

def run_prediction_tab(model):
    #Get input data from user
    with st.form(key=f'Prediction_Form_{type(model).__name__}'):
        #st.markdown('**Client Personal Information**')
        st.markdown("<h5 style='color:#FF4B4B;'>👤  Client Personal Information</h5>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns(3)
        with col1:
            Housing = st.selectbox('Housing status', list(Housing_mapping.keys()))
        with col2:
            gen = st.selectbox("Gender", list(gender_mapping.keys()))  
        with col3:
            suit = st.selectbox('Suite', list(suite))
        NAME_HOUSING_TYPE_ENCODED = Housing_mapping[Housing]
        gender_sui = gen+'_'+suit
        GENDER_SUITE_ENCODED = gender_suite[gender_sui]
        
        #Education ocuupation 
        col1,col2,col3 = st.columns(3)
        with col1:
            Education = st.selectbox('Highest level of education:', list(edu))
        with col2:
            Occupation = st.selectbox('Occupation:', list(occu))
        with col3:
            relationship = st.selectbox('Relationship Status:', list(rel))
        occ_edu = Education + '_' + Occupation
        rel_edu = relationship + '_' + Education
        EDUCATION_OCCUPATION_ENCODED = education_occupation[occ_edu]
        REL_EDUCATION_ENCODED = rel_education[rel_edu]
       
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            Age = st.number_input('Age', min_value=18, max_value= 70)
        #own car age
        with col2:
            OWN_CAR_AGE = st.number_input("Age of car (if No:-1)", min_value= -1)
        with col3:
            children = st.number_input('Number of **Children**', min_value=0)
        with col4:
            CNT_FAM_MEMBERS = st.number_input('**Family Members** count', min_value=0)
            
        
        st.divider()
        #st.markdown('**💰 Client Employment and Income Incicators**')
        st.markdown("<h5 style='color:#FF4B4B;'>💰 Client Employment and Income Information</h5>", unsafe_allow_html=True)
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            YEARS_EMPLOYED = st.number_input('Years employed')
        if YEARS_EMPLOYED > Age:
            st.error('🚨 Invalid input: Years employed cannot exceed the age of the client')
        with col2:
            income = st.number_input('Annual Income(USD)', min_value=0)
        with col3:
            annuity = st.number_input('Amount annuity', min_value=0,max_value=2000)
        with col4:
            credit = st.number_input('Amount credit', min_value=450,max_value=50000)
        CHILDREN_INCOME_RATIO = (children/(income + 0.00001))
        ANNUITY_CREDIT_RATIO = (annuity/(credit+0.00001))
        ANNUITY_INCOME_RATIO= annuity / (income + 0.0000001)
        CREDIT_INCOME_RATIO = credit / (income + 0.0000001)
        #Age employment 
        EMPLOYED_TO_AGE_RATIO = (YEARS_EMPLOYED /(Age + 000.1))  
        
        st.divider()
        #prev behaviour
        #st.markdown('**📄📚 Client Loan History**')
        st.markdown("<h5 style='color:#FF4B4B;'>📚 Client Loan History</h5>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns(3)
        with col1:
            PREV_APP_COUNT = st.number_input('Number of previous applicatios',min_value=0)
        with col2:
            PREV_APP_STATUS_REFUSED = st.number_input('Number of **Refused** applications', min_value=0)  
        with col3:
                unused = st.number_input('Number of **Unused** applications',min_value=0)
        if (PREV_APP_STATUS_REFUSED + unused) > PREV_APP_COUNT:
            st.error('🚨 Invalid Input: Unused and refused applications should not exceed total applications')

        PREV_APP_RATIO_PREV_APP_STATUS_UNUSED_OFFER = (unused/(PREV_APP_COUNT + 0.00001))
        col1,col2,col3 = st.columns(3)
        with col1:
            application = st.number_input('Average **Amount Application**')
        with col2:
            PREV_APK_AMT_CREDIT_MEAN = st.number_input('Average **Amount Credit**')
        with col3:
            PREV_APK_AMT_DECLINED_MIN = st.number_input('Lowest *Application Amount Declined*')
         #calculation - credit and annuity
        PREV_APK_AMT_CREDIT_APPLICATION_RATIO_MEAN = application /(PREV_APK_AMT_CREDIT_MEAN + 0.00001)
        
        col1,col2,col3 = st.columns(3)
        with col1:
            PREV_APK_AMT_CREDIT_MIN = st.number_input('Lowest approved amount')
        with col2:
            PREV_APK_AMT_ANNUITY_MIN = st.number_input('Lowest monthly annuity')
        with col3:
            PREV_APK_AMT_INTEREST_MAX = st.number_input('Highest **Interest** collected from client')
        #Previous products applied and loan reason:
        COUNT_PREV_PROD_Card_Street= st.number_input('Number of times applicant applied for the **CARD STREET PRODUCT**', min_value=0)
        st.divider()
        #Instalments payments behaviour
        #st.markdown('**Installment Payment Behaviour**')
        st.markdown("<h5 style='color:#FF4B4B;'>📊 Installment Payment Behaviour</h5>", unsafe_allow_html=True)
        col1, col2,col3 = st.columns(3)
        with col1:
            NUM_INSTALMENTS_EARLY_PAYMENTS = st.number_input('Number of **EARLY** payments',min_value = 0)
        
        with col2:
            NUM_INSTALMENT_PARTIAL_PAYMENTS = st.number_input('Number of **PARTIAL** payments', min_value=0, max_value=140000)
        with col3:
            NUM_INSTALMENTS_LATE_PAYMENTS = st.number_input('Number of **LATE** payments', min_value = 0)
            
        col1,col2 = st.columns(2)
        with col1:
            PREV_AMT_PAYMENT_MIN = st.number_input('Lowest installments amount paid')
        with col2:
            PREV_AMT_INTSALMENT_MAX = st.number_input('Highest instalment amount paid')
        st.divider()
        #External Credit source
        #st.markdown('**Credit Scoring & Risk Indicators**')
        st.markdown("<h5 style='color:#FF4B4B;'>💳 Credit Scoring & Risk Indicators</h5>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns(3)
        with col1:
            ext_1 = st.number_input('External Credit score 1: ',0.00,1.)
        with col2:
            ext_2 = st.number_input('External Credit score 2: ',0.00,1.)
        with col3:
            ext_3 = st.number_input('External Credit score 3: ',0.00,1.)  
        
       # st.markdown('**Client Social Circle Information**')
        col1,col2 = st.columns(2)
        with col1:
            obs_60 = st.number_input("People in client social circle struggling to pay in time: last 2 months", 0,5)
        with col2:
            def_60 = st.number_input("People in client social circle who defaulted: last 2 months", 0,5)
        col1,col2 = st.columns(2)
        with col1:
            obs_30 = st.number_input("People in in in client social circle struggling to pay in time: last month", 0,5)
        with col2:
            def_30 = st.number_input("People client social circle who defaulted: previous month", 0,5)   
        
        FLAG_DOCUMENT_3 = st.slider('Applicant provided **Document 3**',0,1) 
          #weighted  and mean
        WEIGHTED_EXT_SOURCE = (ext_1*2) + (ext_2*3) +(ext_3*4)
        EXT_SOURCE_MEAN = (ext_1+ext_2+ext_3)/3
        
        #OBS and DEf 60 calculations
        OBS_DEF_60_MUL = obs_60 * def_60
        OBS_DEF_30_MUL = obs_30 * def_30
        DEF_30_CREDIT_RATIO = credit / (def_30 +  0.00001)
        DEF_60_CREDIT_RATIO = credit / (def_60 +  0.00001)
            
        #region rating
        st.divider()
        #st.markdown('**Client residential Region Information**')
        st.markdown("<h5 style='color:#FF4B4B;'>🌍 Client residential Region Information</h5>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns(3)
        with col1:
            REGION_RATING_CLIENT = st.slider('Rating of the **REGION**',1,3)
        with col2:
            REGION_RATING_CLIENT_W_CITY = st.slider('Rating of **CITY**',1,3)
        with col3:
            REGION_POPULATION_RELATIVE = st.number_input('Region Population', min_value=0.000,max_value=1.000)
        REGION_RATING_MUL = REGION_RATING_CLIENT * REGION_RATING_CLIENT_W_CITY
        REGIONS_RATING_INCOME_MUL = (REGION_RATING_CLIENT +REGION_RATING_CLIENT_W_CITY) * income
        #years details change 
        col1,col2,col3 = st.columns(3)
        with col1:
            Phone = st.number_input('Years since changed Phone', min_value=0.0)
        with col2:
            Registration = st.number_input('Years since registered any property:', min_value=0.0)
        with col3:
            ID = st.number_input('Years since ID published', min_value=0.0)
        YEARS_DETAILS_CHANGE_SUM= Phone + Registration + ID

        submitted = st.form_submit_button('Predict')
        if submitted:
            inputs = [
GENDER_SUITE_ENCODED,EDUCATION_OCCUPATION_ENCODED,
REL_EDUCATION_ENCODED,NAME_HOUSING_TYPE_ENCODED,Age, OWN_CAR_AGE, EMPLOYED_TO_AGE_RATIO,YEARS_EMPLOYED,
 #PREV STAFF
PREV_APP_STATUS_REFUSED,PREV_APP_RATIO_PREV_APP_STATUS_UNUSED_OFFER,
PREV_APK_AMT_CREDIT_APPLICATION_RATIO_MEAN, PREV_APK_AMT_CREDIT_MEAN, PREV_APK_AMT_DECLINED_MIN,
PREV_APK_AMT_INTEREST_MAX,PREV_APK_AMT_CREDIT_MIN,PREV_APK_AMT_ANNUITY_MIN,PREV_AMT_PAYMENT_MIN,PREV_AMT_INTSALMENT_MAX,COUNT_PREV_PROD_Card_Street,
    #instalments
    NUM_INSTALMENTS_EARLY_PAYMENTS,NUM_INSTALMENTS_LATE_PAYMENTS,NUM_INSTALMENT_PARTIAL_PAYMENTS,

    #Alternstive
annuity,credit,ANNUITY_CREDIT_RATIO,CNT_FAM_MEMBERS,  YEARS_DETAILS_CHANGE_SUM,WEIGHTED_EXT_SOURCE,EXT_SOURCE_MEAN, OBS_DEF_30_MUL,OBS_DEF_60_MUL,
DEF_30_CREDIT_RATIO,DEF_60_CREDIT_RATIO,CHILDREN_INCOME_RATIO,CREDIT_INCOME_RATIO,ANNUITY_INCOME_RATIO,REGION_RATING_MUL,REGION_POPULATION_RELATIVE,REGIONS_RATING_INCOME_MUL,FLAG_DOCUMENT_3
]
            #prediction  
            label,proba,shap_values = prediction_default(inputs,model)
            st.success(label)
            st.info(f'📈 Probability of Default: **{proba}%**')
            
            st.subheader("Computed SHAP explanation")
            st.markdown('**RED** coloured features increase the default probability while **BLUE** pulls it down.')
            fig,ax = plt.subplots()
            shap.plots.waterfall(shap_values[0],max_display= 10, show = False)
            st.pyplot(fig,bbox_inches='tight')
            
          

            
        

if __name__ == '__main__':
    main()


