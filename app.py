import pandas as pd
import streamlit as st

# region Functions

# endregion

# region UI
st.title('📊 AI Data Visualizer')

# TODO : file uploader, where we can only accept csv / excel files
uploaded_file = st.file_uploader("Upload your file here", type=["csv"])

# TODO : button to upload the file and then visualize the data
if st.button("Upload File", use_container_width=True):

    # TODO : method for validation (stelah smuanya validation casenya beres)

    # TODO : check if the file uploaded is None

    # TODO : check if the file is a valid extension file

    # TODO : check if the file size is more than maximum file size
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
    
    # TODO : ini untuk validation nya (ini apakah filenya ada yang exceed / ada file format yang ga valid)

    # TODO : ini untuk process file yang diupload dan di visualize

    # TODO : process the uploaded file and visualize the data into the excel
# endregion