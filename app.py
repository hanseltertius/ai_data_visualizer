import pandas as pd
import streamlit as st
import openai

# region Initialize API key
openai.api_key = st.secrets.get("OPENAI_API_KEY")
# endregion

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

# TODO : create reusable function to display data frame
def display_dataframe(uploaded_file = None, selected_sheet_name = "", is_excel=True):
    if is_excel:
        st.markdown(f"Displayed data from sheet: ```{selected_sheet_name}```")
        df = pd.read_excel(excel_file, sheet_name=selected_sheet_name)
        st.dataframe(df)
        display_tabs(df)
    else:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        display_tabs(df)

# TODO : create reusable function to display tabs
def display_tabs(df):
    tab_summary, tab_insight, tab_chart = st.tabs(["Summary", "Insight", "Chart"])

    with tab_summary:
        # TODO : maybe using a summary again, inside tab there is another tab
        # TODO : summarize columns by (select columns)
        # TODO : do summary of the tabs
        # st.write(df.describe())
        # st.write(df.info())
        st.write("Summary")

        # TODO : maybe we can summarize from 2 columns and some stuff, maybe can select multiple columns to generate summary, ini mesti dapat info2 dari dataframe bgmn

    with tab_insight:
        user_input = st.text_area("Insight", key="insight_input", placeholder="Write your insight here... (Leading / Trailing whitespaces will be trimmed at the input generation)")

        if st.button("Generate Insight", use_container_width=True):
            reformatted_user_input = user_input.strip()
            if reformatted_user_input:
                with st.spinner("Generating insight..."):
                    # TODO : getting the dataframe
                    generated = generate_insight_from_openai(reformatted_user_input, df)
                    st.write(generated)
                # TODO : kita bakal generating dl (smua layout di disable (maybe))

                # TODO : response need to generate text to speech (use hugging face models)
            else:
                st.warning("Insight input cannot be empty.")
    with tab_chart:
        # TODO : tinggal generate the chart based on select box
        # TODO : setelah pake generate the chart based on select box, tinggal pake button, trus generate popup ny
        # TODO : generate summary
        st.write("Chart")
    # TODO : display tabs (summary, insight and charts)

def generate_insight_from_openai(user_input, df):
    csv_data = df.to_csv(index=False)
    prompt = f"""
    Given the following data table and user question, provide a data insight or analysis.
    Data Table: 
    {csv_data}
    User Question: 
    {user_input}
    Can you generate the the insight based on "User Question".
    """

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # TODO : it potentially get out of index, need to handle the feature
    result = response.choices[0].message.content.strip()

    return result
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
                display_dataframe(selected_sheet_name=selected_sheet)
            # endregion
        else:
            # region Handle Single sheet upload
            selected_sheet = sheet_names[0]
            display_dataframe(selected_sheet_name=selected_sheet)
            # endregion
    elif file_type == "text/csv":
        display_dataframe(uploaded_file=uploaded_file, is_excel=False)

# endregion