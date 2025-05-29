import pandas as pd
import streamlit as st

# region Functions
def is_valid_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.size > 0:
            return True
        else:
            st.error("File size must not be empty.")
            return False
    else:
        st.error("File must not be empty. Please upload a file")
        return False
# endregion

# region UI
st.title('📊 AI Data Visualizer')

# TODO : file uploader, where we can only accept csv / excel files
uploaded_file = st.file_uploader("Upload your file here", type=["csv"], key="file_uploader")

# TODO : button to upload the file and then visualize the data
if st.button("Upload File", use_container_width=True):
    valid_uploaded_file = is_valid_uploaded_file(uploaded_file)

    if valid_uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df)
    
    # TODO : ini untuk process file yang diupload dan di visualize

    # TODO : process the uploaded file and visualize the data into the excel
# endregion