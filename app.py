import pandas as pd
import streamlit as st
import openai
import requests
import time

from classes.barchart import BarChart
from classes.piechart import PieChart
from classes.scatterplot import ScatterPlot
from classes.datahandler import DataHandler

# region Initialize API key
openai.api_key = st.secrets.get("OPENAI_API_KEY")
# endregion

# region Variables
ALLOWED_FILE_TYPES = ["csv", "xlsx", "xls", "xlsm"]
OPEN_AI_MODEL = "gpt-4o"
HUGGING_FACE_AI_MODEL = "facebook/bart-large-cnn"
data_handler = DataHandler()
# endregion

# region State Initialization
def initialize_session_state():
    defaults = {
        "uploaded_file": None,
        "generated_insight": None,
        "display_summarized_columns": False,
        "last_selected_columns": [],
        "insight_generating": False,
        "insight_input_to_generate": None,
        "insight_df_to_generate": None,
        "last_selected_section": None,
        "is_loading_data": False,
        "insight_error_message": None,
        "summarized_insight": None,
        "summarized_insight_error_message": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Call this function at the top of your script
initialize_session_state()
# endregion

# region Functions
def is_empty_columns(df):
    return df.empty or len(df.columns) == 0

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
    st.session_state.generated_insight = None
    st.session_state.insight_error_message = None
    st.session_state.is_loading_data = False
    st.session_state.summarized_insight = None
    st.session_state.summarized_insight_error_message = None

def format_column_value(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)

def generate_summary_with_huggingface(text):
    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {st.secrets.get("HUGGINGFACE_API_KEY")}"}
    payload = {"inputs": text}
    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    if response.status_code == 200:
        summary = response.json()[0]['summary_text']
        return summary
    else:
        raise Exception(f"HuggingFace API Error: {response.text}")

@st.dialog("Bar Graph Result", width="large")
def show_bar_graph(df, x_axis, y_axis, selected_file_name, selected_sheet_name = ""):
    # region Setup Bar Graph
    bar_chart = BarChart()
    figure = bar_chart.generate(df, x_axis, y_axis, selected_file_name, selected_sheet_name)
    # endregion

    # region Generate Image
    buffer = bar_chart.get_buffer_image(figure)
    
    num_x = len(df[x_axis].unique())
    fig_width = min(max(8, num_x * 0.5), 40) # Dynamically set width: 0.5 inch per x label, min 8, max 40

    image_html = bar_chart.generate_image(buffer, fig_width, 6)

    st.markdown(image_html, unsafe_allow_html=True)
    # endregion

    # region Download Button
    generate_download_png_button(buffer, "bar_chart.png")
    # endregion

@st.dialog("Pie Chart Result", width="large")
def show_pie_chart(df, column, selected_file_name, selected_sheet_name = ""):
    # region Setup Pie Chart
    pie_chart = PieChart()
    figure = pie_chart.generate(df, column, selected_file_name, selected_sheet_name)
    # endregion

    # region Generate Image
    buffer = pie_chart.get_buffer_image(figure)
    image_html = pie_chart.generate_image(buffer, 8, 8)
    st.markdown(image_html, unsafe_allow_html=True)
    # endregion

    # region Download Button
    generate_download_png_button(buffer, "pie_chart.png")
    # endregion

@st.dialog("Scatter Plot Result", width="large")
def show_scatter_plot(df, x_axis, y_axis, selected_file_name, selected_sheet_name=""):
    # region Setup Scatter Plot
    scatter_plot = ScatterPlot()
    figure = scatter_plot.generate(df, x_axis, y_axis, selected_file_name, selected_sheet_name)
    # endregion

    # region Generate Image
    buffer = scatter_plot.get_buffer_image(figure)
    image_html = scatter_plot.generate_image(buffer, 8, 6)
    st.markdown(image_html, unsafe_allow_html=True)
    # endregion

    # region Download Button
    generate_download_png_button(buffer, "scatter_plot.png")
    # endregion

def generate_summarized_insight(text):
    try:
        with st.spinner("Generating summarized insight..."):
            summary = generate_summary_with_huggingface(text)
            response_placeholder = st.empty()
            displayed = ""
            for char in summary:
                displayed += char
                response_placeholder.markdown(displayed + "▌")
                time.sleep(0.003)
            response_placeholder.empty()
        st.session_state.summarized_insight = summary
    except Exception as e:
        st.session_state.summarized_insight_error_message = f"""
            Failed to summarize insight: 
            {e}
        """
    finally:
        st.session_state.summary_generating = False
        st.session_state.is_loading_data = False
        time.sleep(0.001)
        st.rerun()

def display_dataframe(uploaded_file = None, selected_sheet_name = "", selected_file_name = "", is_excel=True):
    if is_excel:
        # region Display DataFrame from Excel
        try:
            st.markdown(f"Displayed data from sheet: ```{selected_sheet_name}```")
            df = pd.read_excel(excel_file, sheet_name=selected_sheet_name)
            if is_empty_columns(df):
                st.error("The selected Excel sheet has no columns or data.")
                return
            data_handler.df = df
            st.dataframe(data_handler.df)
            display_segmented_control(data_handler.df, selected_sheet_name=selected_sheet_name, selected_file_name=selected_file_name)
        except Exception as e:
            st.error(f"Failed to read the Excel file or sheet: {e}")
        # endregion
    else:
        # region Display DataFrame from CSV
        uploaded_file.seek(0) # Check if file is empty or invalid before reading
        content = uploaded_file.read()
        if not content.strip():
            st.error("The uploaded CSV file is empty.")
            return
        uploaded_file.seek(0) # Reset pointer after reading
        try:
            df = pd.read_csv(uploaded_file)
            if is_empty_columns(df):
                st.error("The uploaded CSV file has no columns or data.")
                return
            data_handler.df = df
            st.dataframe(data_handler.df)
            display_segmented_control(data_handler.df, selected_sheet_name=selected_sheet_name, selected_file_name=selected_file_name)
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty or invalid.")
        # endregion

def display_insight(df):
    if is_empty_columns(df):
        st.warning("There are no columns to generate insight with. Please upload a valid file with data.")
    else:
        insight_input = st.text_area(
            label="Insight",
            key="insight_input", 
            placeholder="Write your insight here... (Leading / Trailing whitespaces will be trimmed at the input generation)", 
            label_visibility="collapsed",
            disabled=st.session_state.get("is_loading_data")
        )

        reformatted_insight_input = insight_input.strip()

        if st.button(
            "Generate Insight", 
            disabled=st.session_state.get("is_loading_data"), 
            use_container_width=True):
            if reformatted_insight_input:
                st.session_state.is_loading_data = True
                st.session_state.generated_insight = None
                st.session_state.insight_generating = True
                st.session_state.insight_input_to_generate = reformatted_insight_input
                st.session_state.insight_df_to_generate = df
                st.session_state.insight_error_message = None
                st.rerun()
            else:
                st.error("Insight input cannot be empty.")

        # region Generating Insight
        if st.session_state.get("insight_generating"):
            generate_insight_from_openai(st.session_state.get("insight_input_to_generate"), st.session_state.get("insight_df_to_generate"))
        # endregion

        # region Show Error from Generating Insight
        if st.session_state.get("insight_error_message"):
            st.error(st.session_state.insight_error_message)
        # endregion

        # region Show Generated Insight
        if st.session_state.get("generated_insight") and not st.session_state.get("insight_generating"):
            # Container component to disappear the generated insight layout while loading
            with st.container():
                st.markdown("#### Generated Insight")
                st.markdown(f"Insight generation powered by ```{OPEN_AI_MODEL}``` via OpenAI 🤖")
                st.markdown('<hr style="border: 1px solid #bbb; margin-top: 8px; margin-bottom: 8px;">', unsafe_allow_html=True)
                st.markdown(st.session_state.generated_insight)
                st.download_button(
                    label="Download Insight",
                    data=st.session_state.generated_insight,
                    file_name="generated_insight.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="generated_download_insight_btn",
                    disabled=st.session_state.get("is_loading_data")
                )

                if st.button(
                    "Summarize Insight", 
                    use_container_width=True, 
                    key="generated_summarized_insight_btn",
                    disabled=st.session_state.get("is_loading_data")
                ):
                    st.session_state.is_loading_data = True
                    st.session_state.summary_generating = True
                    st.session_state.summarized_insight = None
                    st.session_state.summarized_insight_error_message = None
                    st.rerun()
                    
                if st.session_state.get("summary_generating"):
                    generate_summarized_insight(st.session_state.generated_insight)

                # region Show Summarized insight
                if st.session_state.get("summarized_insight") and not st.session_state.get("summary_generating"):
                    # Container component to disappear the summarized insight layout while loading
                    with st.container():
                        st.markdown("#### Summarized Insight")
                        st.markdown(f"Summary generation powered by `{HUGGING_FACE_AI_MODEL}` via HuggingFace Inference API 🤖")
                        st.markdown('<hr style="border: 1px solid #bbb; margin-top: 8px; margin-bottom: 8px;">', unsafe_allow_html=True)
                        st.markdown(st.session_state.summarized_insight)
                        st.download_button(
                            label="Download Summary",
                            data=st.session_state.summarized_insight,
                            file_name="summarized_insight.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                # endregion

                # region Show Error from Summarized insight
                if st.session_state.get("summarized_insight_error_message"):
                    st.error(st.session_state.summarized_insight_error_message)
                # endregion
        # endregion

def display_bar_chart(df, selected_file_name, selected_sheet_name=""):
    # Filter column if every data in a column is NaN / None
    numeric_cols = data_handler.get_numeric_columns(df=df)
    categorical_cols = data_handler.get_categorical_columns(df=df)
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
            "Average Y-axis", 
            numeric_cols,
            index=None,
            placeholder="Select average y-axis value (numeric columns)",  
            key="bar_y_axis"
        )

        if st.button("Display", key="display_bar_chart", use_container_width=True):
            if x_axis is None:
                st.error("X-axis must not be empty.")
            elif y_axis is None:
                st.error("Y-axis must not be empty.")
            else:
                avg_df = data_handler.get_clean_df(columns=[x_axis, y_axis], df=df, is_get_average=True)
                show_bar_graph(avg_df, x_axis, y_axis, selected_file_name, selected_sheet_name)

def display_pie_chart(df, selected_file_name, selected_sheet_name=""):
    # Filter column if every data in a column is NaN / None
    pie_cols = data_handler.get_all_columns(df=df)
    if len(pie_cols) == 0:
        st.warning("No columns available, please re-upload the data with valid columns")
    else:
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
    numeric_cols = data_handler.get_numeric_columns(df=df)
    if len(numeric_cols) < 2:
        st.warning("At least two numerical columns are required to display a scatter plot. Please re-upload the data with more numeric columns.")
    else:                    
        x_axis = st.selectbox(
            "X-axis",
            numeric_cols,
            index=None,
            placeholder="Select X-axis (numeric column)",
            key="scatter_x_axis"
        )

        # Only show Y-axis select box if X-axis is selected
        if x_axis is not None:
            y_axis_options = [col for col in numeric_cols if col != x_axis]
            y_axis = st.selectbox(
                "Y-axis",
                y_axis_options,
                index=None,
                placeholder="Select Y-axis (numeric column)",
                key="scatter_y_axis"
            )
        else:
            y_axis = None

        if st.button("Display", key="display_scatter_plot", use_container_width=True):
            if x_axis is None:
                st.error("X-axis must not be empty.")
            elif y_axis is None:
                st.error("Y-axis must not be empty.")
            else:
                plot_df = data_handler.get_clean_df(columns=[x_axis, y_axis], df=df)
                show_scatter_plot(plot_df, x_axis, y_axis, selected_file_name, selected_sheet_name)

def display_chart(df, selected_file_name, selected_sheet_name = ""):
    if is_empty_columns(df):
        st.warning("There are no columns to display chart with. Please upload a file containing data.")
    else:
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
    if is_empty_columns(df):
        st.warning("There are no columns to summarize. Please upload a valid file with data.")
    else:
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
    if is_empty_columns(df):
        st.warning("There are no columns to summarize")
    else:
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

def display_segmented_control(df, selected_sheet_name = "", selected_file_name = ""):
    # Remove unnamed columns
    df_without_unnamed_columns = data_handler.remove_unnamed_columns(df=df)

    selected_section = st.segmented_control(
        "Select View",
        options=["Summary", "Insight", "Chart"],
        key="main_segmented_control",
        disabled=st.session_state.get("is_loading_data")
    )

    # Only clear when switching to Insight
    if selected_section == "Insight" and st.session_state.last_selected_section != "Insight":
        st.session_state.generated_insight = None
        st.session_state.summarized_insight = None
        st.session_state.insight_error_message = None

    st.session_state.last_selected_section = selected_section

    if selected_section == "Summary":
        overall_summary, summary_by_columns = st.tabs(["Overall", "Summary by Column(s)"])

        with overall_summary:
            display_overall_summary(df_without_unnamed_columns)
        with summary_by_columns:
            display_summary_by_columns(df_without_unnamed_columns)
    elif selected_section == "Insight":
        display_insight(df_without_unnamed_columns)
    elif selected_section == "Chart":
        display_chart(df_without_unnamed_columns, selected_file_name, selected_sheet_name)

def generate_download_png_button(buffer, file_name = ""):
    st.download_button(
        "Download as PNG",
        data=buffer,
        file_name=file_name,
        mime="image/png",
        use_container_width=True
    )

def generate_insight_from_openai(insight_input, df):
    # Generate Summary to handle large datasets
    csv_data = df.describe(include='all').to_csv()
    prompt = f"""
    Given the following data table and user question, provide a data insight or analysis.
    Data Table: 
    {csv_data}
    User Question: 
    {insight_input}
    Can you generate the the insight based on "User Question".
    """

    with st.spinner("Generating insight..."):
        response_placeholder = st.empty()
        full_response = ""
        try:
            # region Generate the response in the form of stream
            stream = openai.chat.completions.create(
                model=OPEN_AI_MODEL,
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
            # endregion

            st.session_state.generated_insight = full_response
        except Exception as e:
            st.session_state.insight_error_message = f"""
                Failed to generate insight: 
                {e}
            """
        finally:
            st.session_state.insight_generating = False
            st.session_state.insight_input_to_generate = None
            st.session_state.insight_df_to_generate = None
            st.session_state.is_loading_data = False
            time.sleep(0.001)
            st.rerun()
# endregion

# region UI
st.title('📊 AI Data Visualizer')

uploaded_file = st.file_uploader(
    "Upload your file here", 
    type=ALLOWED_FILE_TYPES, 
    key="file_uploader",
    on_change=reset_uploaded_file,
    disabled=st.session_state.get("is_loading_data")
)

if st.button("Upload File", use_container_width=True, disabled=st.session_state.get("is_loading_data")):
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
                        key="sheet_selector",
                        disabled=st.session_state.get("is_loading_data")
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