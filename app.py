import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import openai
import io
import base64

# region Initialize API key
openai.api_key = st.secrets.get("OPENAI_API_KEY")
# endregion

# region Variables
ALLOWED_FILE_TYPES = ["csv", "xlsx", "xls", "xlsm"]
# endregion

# region State Initialization
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "generated_insight" not in st.session_state:
    st.session_state.generated_insight = None

if "display_summarized_columns" not in st.session_state:
    st.session_state.display_summarized_columns = False

if "last_selected_columns" not in st.session_state:
    st.session_state.last_selected_columns = []
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

def format_column_value(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)

@st.dialog("Bar Graph Result", width="large")
def show_bar_graph(df, x_axis, y_axis, selected_file_name, selected_sheet_name = ""):
    # region Setup Bar Graph
    chart_title = f"Summary of {selected_file_name} in {selected_sheet_name}" if selected_sheet_name else f"Summary of {selected_file_name}"
    num_x = len(df[x_axis].unique())
    fig_width = min(max(8, num_x * 0.5), 40) # Dynamically set width: 0.5 inch per x label, min 8, max 40
    figure, axes = plt.subplots(figsize=(fig_width, 6))
    axes.bar(df[x_axis].astype(str), df[y_axis])
    axes.set_xlabel(x_axis)
    axes.set_ylabel(y_axis)
    axes.set_title(chart_title)
    plt.xticks(rotation=45, ha="center")
    plt.yticks(rotation=315, va="center")
    plt.tight_layout()
    # endregion

    # region Generate Image
    buf = io.BytesIO()
    figure.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    # Display image in a scrollable div
    dpi = 100  # Default matplotlib DPI
    img_width_px = int(fig_width * dpi)
    img_height_px = int(6 * dpi)

    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    st.markdown(
        f"""
        <div style="overflow-x: auto; overflow-y: auto; width: 100%; max-height: 800px; padding-bottom: 8px; margin-bottom: 8px">
            <img 
                style="display: block; min-width: {img_width_px}px; min-height: {img_height_px}px; width: auto; height: auto;" 
                src="data:image/png;base64,{img_base64}" />
        </div>
        """,
        unsafe_allow_html=True
    )
    # endregion

    # region Download Button
    st.download_button(
        "Download as PNG",
        data=buf,
        file_name="bar_chart.png",
        mime="image/png",
        use_container_width=True
    )
    # endregion

@st.dialog("Pie Chart Result", width="large")
def show_pie_chart(df, column, selected_file_name, selected_sheet_name = ""):
    # region Setup Pie Chart
    chart_title = f"Pie Chart of {column} in {selected_file_name}" if not selected_sheet_name else f"Pie Chart of {column} in {selected_file_name} ({selected_sheet_name})"
    value_counts = df[column].value_counts(dropna=False)
    figure, axes = plt.subplots(figsize=(8, 8))
    axes.pie(value_counts, labels=value_counts.index.astype(str), autopct='%1.1f%%', startangle=90)
    axes.set_title(chart_title)
    plt.tight_layout()
    # endregion

    # region Generate Image
    buf = io.BytesIO()
    figure.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    dpi = 100
    img_width_px = int(8 * dpi)
    img_height_px = int(8 * dpi)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    st.markdown(
        f"""
        <div style="overflow-x: auto; overflow-y: auto; width: 100%; max-height: 800px; margin-bottom: 8px">
            <img 
                style="display: block; min-width: {img_width_px}px; min-height: {img_height_px}px; width: auto; height: auto;" 
                src="data:image/png;base64,{img_base64}" />
        </div>
        """,
        unsafe_allow_html=True
    )
    # endregion

    # region Download Button
    st.download_button(
        "Download as PNG",
        data=buf,
        file_name="pie_chart.png",
        mime="image/png",
        use_container_width=True
    )
    # endregion

@st.dialog("Scatter Plot Result", width="large")
def show_scatter_plot(df, x_axis, y_axis, selected_file_name, selected_sheet_name=""):
    # region Setup Scatter Plot
    chart_title = f"Scatter Plot of {y_axis} vs {x_axis} in {selected_file_name}" if not selected_sheet_name else f"Scatter Plot of {y_axis} vs {x_axis} in {selected_file_name} ({selected_sheet_name})"
    figure, axes = plt.subplots(figsize=(8, 6))
    axes.scatter(df[x_axis], df[y_axis], alpha=0.7)
    axes.set_xlabel(x_axis)
    axes.set_ylabel(y_axis)
    axes.set_title(chart_title)
    plt.xticks(rotation=45, ha="center")
    plt.yticks(rotation=315, va="center")
    plt.tight_layout()
    # endregion

    # region Generate Image
    buf = io.BytesIO()
    figure.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    dpi = 100
    img_width_px = int(8 * dpi)
    img_height_px = int(6 * dpi)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    st.markdown(
        f"""
        <div style="overflow-x: auto; overflow-y: auto; width: 100%; max-height: 800px; padding-bottom: 8px; margin-bottom: 8px">
            <img 
                style="display: block; min-width: {img_width_px}px; min-height: {img_height_px}px; width: auto; height: auto;" 
                src="data:image/png;base64,{img_base64}" />
        </div>
        """,
        unsafe_allow_html=True
    )
    # endregion

    # region Download Button
    st.download_button(
        "Download as PNG",
        data=buf,
        file_name="bar_chart.png",
        mime="image/png",
        use_container_width=True
    )
    # endregion

def display_dataframe(uploaded_file = None, selected_sheet_name = "", selected_file_name = "", is_excel=True):
    if is_excel:
        # region Display DataFrame from Excel
        try:
            st.markdown(f"Displayed data from sheet: ```{selected_sheet_name}```")
            df = pd.read_excel(excel_file, sheet_name=selected_sheet_name)
            if df.empty or len(df.columns) == 0:
                st.error("The selected Excel sheet has no columns or data.")
                return
            st.dataframe(df)
            display_tabs(df, selected_sheet_name=selected_sheet_name, selected_file_name=selected_file_name)
        except Exception as e:
            st.error(f"Failed to read the Excel file or sheet: {e}")
        # endregion
    else:
        # region Display DataFrame from CSV
        # Check if file is empty or invalid before reading
        uploaded_file.seek(0)
        content = uploaded_file.read()
        if not content.strip():
            st.error("The uploaded CSV file is empty.")
            return
        # Reset pointer after reading
        uploaded_file.seek(0)  
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty or len(df.columns) == 0:
                st.error("The uploaded CSV file has no columns or data.")
                return
            st.dataframe(df)
            display_tabs(df, selected_sheet_name=selected_sheet_name, selected_file_name=selected_file_name)
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty or invalid.")
        # endregion

def display_insight(df):
    user_input = st.text_area(
        label="Insight",
        key="insight_input", 
        placeholder="Write your insight here... (Leading / Trailing whitespaces will be trimmed at the input generation)", 
        label_visibility="hidden"
    )

    if st.button("Generate Insight", use_container_width=True):
        reformatted_user_input = user_input.strip()
        if reformatted_user_input:
            generate_insight_from_openai(reformatted_user_input, df)
        else:
            st.error("Insight input cannot be empty.")

    if st.session_state.get("generated_insight"):
        st.markdown(st.session_state.generated_insight)
        st.download_button(
            label="Download Insight",
            data=st.session_state.generated_insight,
            file_name="generated_insight.txt",
            mime="text/plain",
            use_container_width=True
        )

def display_bar_chart(df, selected_file_name, selected_sheet_name=""):
    # Filter column if every data in a column is NaN / None
    numeric_cols = [column for column in df.select_dtypes(include="number").columns if not df[column].isna().all()]
    categorical_cols = [column for column in df.select_dtypes(exclude="number").columns if not df[column].isna().all()]
    if len(numeric_cols) == 0:
        st.warning("No numerical columns available, please re-upload the data with numeric columns")
    elif len(categorical_cols) == 0:
        st.warning("No categorical columns available, please re-upload the data with categorical columns")
    else:
        x_axis = st.selectbox(
            "X-axis", 
            categorical_cols,
            index=None,
            placeholder= "Select x-axis (categorical columns)",
            key="bar_x_axis"
        )
        y_axis = st.selectbox(
            "Y-axis", 
            numeric_cols,
            index=None,
            placeholder="Select y-axis (numeric columns)",  
            key="bar_y_axis"
        )

        if st.button("Display", key="display_bar_chart", use_container_width=True):
            if x_axis is None:
                st.error("X-axis must not be empty.")
            elif y_axis is None:
                st.error("Y-axis must not be empty.")
            else:
                show_bar_graph(df, x_axis, y_axis, selected_file_name, selected_sheet_name)

def display_pie_chart(df, selected_file_name, selected_sheet_name=""):
    # Filter column if every data in a column is NaN / None
    pie_cols = [col for col in df.columns if not df[col].isna().all()]
    pie_col = st.selectbox(
        "Pie Chart Column",
        pie_cols,
        index=None,
        placeholder="Select a column for pie chart",
        key="pie_chart_col"
    )
    if st.button("Display", key="display_pie_chart", use_container_width=True):
        if pie_col is None:
            st.error("Please select a column for the pie chart.")
        else:
            show_pie_chart(df, pie_col, selected_file_name=selected_file_name, selected_sheet_name=selected_sheet_name)

def display_scatter_plot(df, selected_file_name, selected_sheet_name=""):
    numeric_cols = [column for column in df.select_dtypes(include="number").columns if not df[column].isna().all()]
    categorical_cols = [column for column in df.select_dtypes(exclude="number").columns if not df[column].isna().all()]
    if len(numeric_cols) == 0:
        st.warning("No numerical columns available, please re-upload the data with numeric columns")
    elif len(categorical_cols) == 0:
        st.warning("No categorical columns available, please re-upload the data with categorical columns")
    else:                    
        x_axis = st.selectbox(
            "X-axis",
            numeric_cols,
            index=None,
            placeholder="Select X-axis (numeric column)",
            key="scatter_x_axis"
        )

        y_axis = st.selectbox(
            "Y-axis",
            categorical_cols,
            index=None,
            placeholder="Select Y-axis (numeric column)",
            key="scatter_y_axis"
        )

        if st.button("Display", key="display_scatter_plot", use_container_width=True):
            if x_axis is None:
                st.error("X-axis must not be empty.")
            elif y_axis is None:
                st.error("Y-axis must not be empty.")
            else:
                show_scatter_plot(df, x_axis, y_axis, selected_file_name, selected_sheet_name)

def display_chart(df, selected_file_name, selected_sheet_name = ""):
    displayed_data_choices = ["Bar Chart", "Pie Chart", "Scatter Plot"]

    selected_display_data = st.selectbox(
        "Display in", 
        displayed_data_choices, 
        index=None, 
        placeholder="Select data type to display", 
        key="displayed_data_selector"
    )

    if selected_display_data is not None:
        if selected_display_data == "Bar Chart":
            display_bar_chart(df, selected_file_name, selected_sheet_name)
        elif selected_display_data == "Pie Chart":
            display_pie_chart(df, selected_file_name, selected_sheet_name)
        elif selected_display_data == "Scatter Plot":
            display_scatter_plot(df, selected_file_name, selected_sheet_name)

def display_overall_summary(df):
    st.subheader("Overall Data Summary")

    with st.expander("Overall Summary", expanded=True):

        # region Table Information
        st.markdown("#### Table Information")
        num_rows = len(df)
        num_cols = len(df.columns)
        st.markdown(f"**Rows Count:** ```{num_rows}```")
        st.markdown(f"**Columns Count:** ```{num_cols}```")
        st.markdown('<hr style="border: 1px solid #bbb; margin-top: 8px; margin-bottom: 8px;">', unsafe_allow_html=True)
        # endregion

        # region Columns Information
        st.markdown("#### Columns Information")

        # region Numeric Columns
        st.markdown("##### Numeric Columns")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        num_numeric_cols = len(numeric_cols)
        st.markdown(f"**Numeric Columns Count:** ```{num_numeric_cols}```")
        if num_numeric_cols > 0:
            st.markdown("**List of Numeric Columns:**")
            st.markdown("\n".join([f"- {col_name}" for col_name in numeric_cols]))
        st.markdown('<hr style="border: 1px dotted #bbb; margin-top: 8px; margin-bottom: 8px;">', unsafe_allow_html=True)
        # endregion

        # region DateTime Columns
        st.markdown("##### DateTime Columns")
        datetime_cols = df.select_dtypes(include=["datetime", "datetime64[ns]"]).columns.tolist()
        num_datetime_cols = len(datetime_cols)
        st.markdown(f"**DateTime Columns Count:** ```{num_datetime_cols}```")
        if num_datetime_cols > 0:
            st.markdown("**List of DateTime Columns:**")
            st.markdown("\n".join([f"- {col_name}" for col_name in datetime_cols]))
        st.markdown('<hr style="border: 1px dotted #bbb; margin-top: 8px; margin-bottom: 8px;">', unsafe_allow_html=True)
        # endregion

        # region Categorical Columns
        st.markdown("##### Categorical Columns")
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_categorical_cols = len(categorical_cols)
        st.markdown(f"**Categorical Columns Count:** ```{num_categorical_cols}```")
        if num_categorical_cols > 0:
            st.markdown("**List of Categorical Columns:**")
            st.markdown("\n".join([f"- {col_name}" for col_name in categorical_cols]))
        # endregion
        
        st.markdown('<hr style="border: 1px solid #bbb; margin-top: 8px; margin-bottom: 8px;">', unsafe_allow_html=True)
        # endregion

        # region Column Data Types
        st.markdown("#### Column Data Types")
        datatypes_df = df.dtypes.reset_index()
        datatypes_df.columns = ["Name", "Datatype"]
        st.dataframe(datatypes_df, hide_index=True)
        st.markdown('<hr style="border: 1px solid #bbb; margin-top: 8px; margin-bottom: 8px;">', unsafe_allow_html=True)
        # endregion

        # region Missing Values
        st.markdown("#### Missing Values")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Name", "Count"]
        st.dataframe(missing_df, hide_index=True)
        # endregion

        # region Columns with the Most / Least Frequent Value Table
        most_least_summary = []
        for col in df.columns:
            value_counts = df[col].value_counts(dropna=False)
            if not value_counts.empty:
                most_frequent_count = value_counts.iloc[0]
                most_frequent_values = value_counts[value_counts == most_frequent_count].index.tolist()

                least_frequent_count = value_counts.iloc[-1]
                least_frequent_values = value_counts[value_counts == least_frequent_count].index.tolist()

                most_least_summary.append({
                    "Column": col,
                    "Most Frequent Value(s)": ", ".join(format_column_value(v) for v in most_frequent_values),
                    "Most Freq Count": round(most_frequent_count, 2) if isinstance(most_frequent_count, float) else most_frequent_count,
                    "Least Frequent Value(s)": ", ".join(format_column_value(v) for v in least_frequent_values),
                    "Least Freq Count": round(least_frequent_count, 2) if isinstance(least_frequent_count, float) else least_frequent_count
                })
            else:
                most_least_summary.append({
                    "Column": col,
                    "Most Frequent Value(s)": "N/A",
                    "Most Freq Count": "N/A",
                    "Least Frequent Value(s)": "N/A",
                    "Least Freq Count": "N/A"
                })
        most_least_df = pd.DataFrame(most_least_summary)
        st.markdown("#### Most/Least Frequent Value(s) per Column")
        st.dataframe(most_least_df, hide_index=True)
        st.markdown('<hr style="border: 1px solid #bbb; margin-top: 8px; margin-bottom: 8px;">', unsafe_allow_html=True)
        # endregion

        # region Statistics (Numeric Columns Only)
        if num_numeric_cols > 0:
            st.markdown('<hr style="border: 1px solid #bbb; margin-top: 8px; margin-bottom: 8px;">', unsafe_allow_html=True)
            st.markdown("#### Statistics (Display Numeric Columns Only)")
            st.write(df.describe())
        # endregion

def display_summary_by_columns(df):
    st.subheader("Summary by Selected Column(s)")
    selected_columns = st.multiselect(
        "Select column(s)",
        df.columns.tolist(),
        key="selected_columns"
    )

    # Reset summarized state if columns change
    if st.session_state.last_selected_columns != selected_columns:
        st.session_state.display_summarized_columns = False
    st.session_state.last_selected_columns = selected_columns

    if st.button("Summarize", key="summarize", use_container_width=True):
        st.session_state.display_summarized_columns = True
    
    # region Show Expanders if displayed summarized columns
    if st.session_state.display_summarized_columns and selected_columns:
        for col in selected_columns:
            with st.expander(f"Column {col}", expanded=True):
                # region General Summary
                st.markdown("#### General Summary")
                st.markdown(f"**Type:** `{df[col].dtype}`")
                st.markdown(f"**Missing values:** `{df[col].isnull().sum()}`")
                st.markdown(f"**Unique values:** `{df[col].nunique()}`")
                st.markdown('<hr style="border: 1px solid #bbb; margin-top: 8px; margin-bottom: 8px;">', unsafe_allow_html=True)
                # endregion

                # region Value Specific Summary
                st.markdown("#### Value Specific Summary")
                if pd.api.types.is_numeric_dtype(df[col]):
                    # region Numeric Columns

                    # region Numeric Values Summary
                    # Calculate percentiles and min/max
                    desc = df[col].describe()
                    percentiles = [
                        ("25%", desc["25%"]),
                        ("50%", desc["50%"]),
                        ("75%", desc["75%"]),
                        ("max", desc["max"]),
                        ("min", desc["min"]),
                        ("mean", desc["mean"])
                    ]
                    # Add "All" as the first option
                    options = [("All Values", percentiles)] + percentiles

                    selected = st.selectbox(
                        "Select percentile/statistic to display",
                        options=options,
                        format_func=lambda x: x[0]
                    )

                    st.markdown("##### Numeric Values Summary")
                    if selected[0] == "All Values":
                        for label, value in selected[1]:
                            st.markdown(f"- **{label}:** `{value}`")
                    else:
                        st.markdown(f"**{selected[0]} value:** `{selected[1]}`")
                    # endregion

                    # region Count Values
                    st.markdown("##### Count Values")
                    st.write(df[col].value_counts(dropna=False))
                    # endregion

                    # endregion
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    # region Datetime Columns

                    # region Datetime Values Summary

                    # region Calculate Values
                    earliest = df[col].min()
                    latest = df[col].max()
                    mode_date = df[col].mode()
                    most_freq_date = mode_date[0] if not mode_date.empty else None
                    mode_day = df[col].dt.day_name().mode()
                    most_freq_day = mode_day[0] if not mode_day.empty else None
                    # endregion

                    # region Format Values for Display
                    summary_tuples = [
                        ("Earliest", earliest.strftime('%A, %d %B %Y') if pd.notnull(earliest) else "N/A"),
                        ("Latest", latest.strftime('%A, %d %B %Y') if pd.notnull(latest) else "N/A"),
                        ("Most Frequent Date", most_freq_date.strftime('%A, %d %B %Y') if most_freq_date is not None and pd.notnull(most_freq_date) else "N/A"),
                        ("Most Frequent Day", most_freq_day if most_freq_day is not None else "N/A"),
                    ]
                    # Add "All" as the first option
                    options = [("All Values", summary_tuples)] + summary_tuples

                    selected = st.selectbox(
                        "Select datetime statistic to display",
                        options=options,
                        format_func=lambda x: x[0],
                        key=f"datetime_{col}"
                    )

                    st.markdown("##### Datetime Values Summary")
                    if selected[0] == "All Values":
                        for label, value in selected[1]:
                            st.markdown(f"- **{label}:** `{value}`")
                    else:
                        st.markdown(f"**{selected[0]}:** `{selected[1]}`")
                    # endregion

                    # endregion

                    # region Count Values
                    st.markdown("##### Count Values")
                    st.write(df[col].value_counts(dropna=False))
                    # endregion

                    # endregion
                else:
                    # region Categorical Columns
                    categorical_values_count = df[col].value_counts(dropna=False)

                    # region Categorical Values Summary

                    # region Prepare Values for Categorical Columns
                    value_counts = df[col].value_counts(dropna=False)
                    mode_values = df[col].mode()
                    min_count = value_counts.min()
                    min_values = value_counts[value_counts == min_count]
                    least_frequent_values = min_values.index.tolist() if not min_values.empty else []
                    # endregion

                    # region Format values for display
                    options = [
                        ("Most Frequent Value(s)", ", ".join(mode_values.tolist()) if not mode_values.empty else "N/A"),
                        ("Least Frequent Value(s)", ", ".join(least_frequent_values))
                    ]

                    options = [("All Values", options)] + options

                    selected = st.selectbox(
                        "Select datetime statistic to display",
                        options=options,
                        format_func=lambda x: x[0],
                        key=f"datetime_{col}"
                    )

                    st.markdown("##### Categorical Values Summary")
                    if selected[0] == "All Values":
                        for label, value in selected[1]:
                            st.markdown(f"- **{label}:** `{value}`")
                    else:
                        st.markdown(f"**{selected[0]}:** `{selected[1]}`")
                    # endregion

                    # endregion

                    # region Count Values
                    st.markdown("##### Count Values")
                    st.write(categorical_values_count)
                    # endregion

                    # endregion
                # endregion
    elif st.session_state.display_summarized_columns and not selected_columns:
        st.error("Please select at least 1 column to summarize.")
    # endregion

def display_tabs(df, selected_sheet_name = "", selected_file_name = ""):
    tab_summary, tab_insight, tab_chart = st.tabs(["Summary", "Insight", "Chart"])

    with tab_summary:
        overall_summary, summary_by_columns = st.tabs(["Overall", "Summary by Column(s)"])

        with overall_summary:
            display_overall_summary(df)
        with summary_by_columns:
            display_summary_by_columns(df)
    with tab_insight:
        display_insight(df)
    with tab_chart:
        display_chart(df, selected_file_name, selected_sheet_name)

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

    with st.spinner("Generating insight..."):
        response_placeholder = st.empty()
        full_response = ""
        try:
            stream = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )

            for chunk in stream:
                if chunk.choices and hasattr(chunk.choices[0], "delta"):
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        full_response += delta.content
                        response_placeholder.markdown(f"{full_response}▌")
            response_placeholder.empty()
            st.session_state.generated_insight = full_response
        except Exception as e:
            st.error(f"""
            Failed to generate insight: 
            {e}         
            """)
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
    file_name = uploaded_file.name

    if file_type in [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/vnd.ms-excel.sheet.macroEnabled.12"
    ]:
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names

            if not sheet_names:
                st.error("The uploaded Excel file contains no sheets.")
            else:
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
                        display_dataframe(selected_sheet_name=selected_sheet, selected_file_name=file_name)
                    # endregion
                else:
                    # region Handle Single sheet upload
                    selected_sheet = sheet_names[0]
                    display_dataframe(selected_sheet_name=selected_sheet, selected_file_name=file_name)
                    # endregion
        except Exception as e:
            st.error(f"Failed to read the Excel file: {e}")
    elif file_type == "text/csv":
        display_dataframe(
            uploaded_file=uploaded_file,
            selected_file_name=file_name,
            is_excel=False
        )

# endregion