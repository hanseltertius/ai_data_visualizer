import pandas as pd
import streamlit as st

# region Variables
ALLOWED_FILE_TYPES = ["csv", "xlsx", "xls", "xlsm"]
# endregion

# region State Initialization
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
# endregion

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
    
def reset_uploaded_file():
    st.session_state.uploaded_file = None
# endregion

# region UI
st.title('📊 AI Data Visualizer')

uploaded_file = st.file_uploader(
    "Upload your file here", 
    type=ALLOWED_FILE_TYPES, 
    key="file_uploader",
    on_change=reset_uploaded_file
)

if st.button("Upload File", use_container_width=True):
    valid_uploaded_file = is_valid_uploaded_file(uploaded_file)

    if valid_uploaded_file:
        st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file is not None:
    uploaded_file = st.session_state.uploaded_file
    file_type = uploaded_file.type

    if file_type in [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/vnd.ms-excel.sheet.macroEnabled.12"
    ]:
        uploaded_file.seek(0) # Reset file upload
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names

        if len(sheet_names) > 1:
            # region Handle Multiple Sheets upload
            selected_sheet = st.selectbox(
                "Sheet Name",
                sheet_names,
                index=None,
                placeholder="Select a sheet to visualize",
                key="sheet_selector"
            )

            if selected_sheet is not None:
                st.markdown(f"Displayed data from sheet: ```{selected_sheet}```")
                df = pd.read_excel(excel_file, sheet_name=selected_sheet)
                st.dataframe(df)
            # endregion
        else:
            # region Handle Single sheet upload
            selected_sheet = sheet_names[0]
            st.markdown(f"Displayed data from sheet: ```{selected_sheet}```")
            df = pd.read_excel(excel_file, sheet_name=selected_sheet)
            st.dataframe(df)
            # endregion
    elif file_type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

# endregion