# ai_data_visualizer

### Project Title

AI Data Visualizer

### Table of Contents

### Purpose and Background

#### Purpose

The AI Data Visualizer is an interactive web application designed to help users
quickly explore, summarize, and gain insights from their tabular data (Excel or
CSV files) using the power of AI. With just a few clicks, users can upload their
datasets, visualize key statistics and charts, and generate AI-powered insights
and summaries—all within a user-friendly interface.

#### Background

Data analysis is a crucial step in understanding trends, patterns, and anomalies
within datasets. However, not everyone is proficient with data science tools or
programming languages. The AI Data Visualizer bridges this gap by providing:

- Automated Data Cleaning: Instantly removes unnecessary columns (like unnamed
  columns) for a cleaner view. Descriptive Summaries: Presents overall and
  column-wise summaries, including missing values, data types, and frequent
  values.
- Interactive Visualizations: Allows users to generate bar charts, pie charts,
  and scatter plots for deeper exploration.
- AI-Powered Insights: Integrates with OpenAI and HuggingFace APIs to generate
  natural language insights and summaries based on the uploaded data and user
  queries.
- Easy Export: Enables users to download generated insights and visualizations
  for reporting or further analysis.

This tool is ideal for educators, students, business analysts, and anyone who
wants to make sense of their data quickly—without writing code.

### Features

- File Uploader that accepts Excel file (.xls, .xlsx) and CSV file
- Display DataFrame from uploaded file
- DataFrame analysis
  - Summary
    - Generate Overall Summary
    - Generate Summary by Column(s)
  - Insight
    - Generate Insight using `gpt-4o` from OpenAI API
    - Generate Summarized Insight using `facebook/bart-large-cnn` from
      HuggingFace Inference API
  - Display chart using the `matplotlib` library
    - Display bar chart
      - X-axis : Categorical Column
      - Y-axis : Numeric Column (average from X-axis)
    - Display pie chart
    - Display scatter plot
      - X-axis : Numeric Column
      - Y-axis : Numeric Column (must be different from X-axis)

### How to run project locally

1. Clone the repository
   - `git clone https://github.com/hanseltertius/ai_data_visualizer.git`
2. Create Virtual Environment in Python
   - `python -m venv myenv`
   - To activate the virtual environment, we need to type:
     - If using Windows : `myenv\Scripts\activate`
     - If using Mac : `source myenv/bin/activate`
3. Install the required libraries (only if they are not installed globally):
   - `pip install pandas streamlit openai openpyxl matplotlib xlrd`
4. We need to [create an OpenAI API Key](#how-to-create-openai-api-key) into the
   secrets file.
5. We need to
   [create a HuggingFace Inference API Key](#how-to-create-huggingface-inference-api-key)
   into the secrets file.
6. Run `streamlit app.py`

### How to create OpenAI API Key

1. Open [Auth Open AI Website](https://auth.openai.com/log-in)

![Screenshot](screenshots/Open%20AI%20Sign%20Up.png)

2. Click "Sign Up" button

![Screenshot](screenshots/Open%20AI%20Create%20Account.png)

3. Try to enter Email Address / Continue with Google for easier registration
4. After Signed Up, try to click "Your profile" section:
   ![Screenshot](screenshots/Open%20AI%20Dashboard.png)
   - Click "Projects" ![Screenshot](screenshots/Open%20AI%20Projects.png)
   - Click "Create" button
     ![Screenshot](screenshots/Open%20AI%20Create%20New%20Project.png)
   - When Create New Project, input Name and then click "Create" button
     ![Screenshot](screenshots/Open%20AI%20Create%20Project%20Result.png)
5. After Create Project, try to click "API Keys" section:
   ![Screenshot](screenshots/Open%20AI%20API%20Keys%20Page.png)
   - Click "Create new secret key" button
     ![Screenshot](screenshots/Open%20AI%20Create%20Secret%20Key.png)
   - Input Name and Project, then click "Create secret key" button
     ![Screenshot](screenshots/Open%20AI%20Secret%20Key%20Result.png)
   - The website will prompt the key, try to click "Copy" button
6. Paste the key into `.streamlit/secrets.toml`
   ![Screenshot](screenshots/Open%20AI%20Installation%20Key%20in%20Streamlit%20Secrets.png)

### How to create HuggingFace Inference API Key

1. Open [HuggingFace](https://huggingface.co/) website
   ![Screenshot](screenshots/HuggingFace%20Home%20Page.png)
2. Sign Up HuggingFace Profile by clicking "Sign Up"
   ![Screenshot](screenshots/HuggingFace%20Sign%20Up%20Page.png)
3. After Creating Profile, click the profile at the top right hand corner, click
   "Access Tokens"
   ![Screenshot](screenshots/HuggingFace%20Click%20Access%20Tokens.png)
4. Click "Create New Token"
   ![Screenshot](screenshots/HuggingFace%20Access%20Token%20Page.png)
5. Allow User Permissions by tick the boxes
   ![Screenshot](screenshots/HuggingFace%20Click%20Permissions.png)
6. After that, try to click "Create Token"
   ![Screenshot](screenshots/HuggingFace%20Click%20Create%20Token.png)
7. We showed the popup that contains Access Token, please copy the respective
   Access Token ![Screenshot](screenshots/HuggingFace%20Copy%20Token.png)
8. Put the token into `.streamlit/secrets.toml`
   ![Screenshot](screenshots/HuggingFace%20Installation%20Key%20in%20Streamlit%20Secrets.png)

### Demo

#### Upload File

- This feature allows uploading CSV, XLS, and XLSX files up to 200 MB each.
- Uploaded data must include a header row for analysis features to work.

##### Upload CSV

![Screenshot](screenshots/Upload%20CSV%20File.png)

![Screenshot](screenshots/Upload%20CSV%20File%20Result.png)

We can show the Data Frame from the CSV as well as the menu to get detailed
analysis.

##### Upload XLS

![Screenshot](screenshots/Upload%20XLS%20File.png)

![Screenshot](screenshots/Upload%20XLS%20File%20Result.png)

We can show the Data Frame from the XLS as well as the menu to get detailed
analysis, we can also upload the file with multiple sheets.

##### Upload XLSX

![Screenshot](screenshots/Upload%20XLSX%20File.png)

![Screenshot](screenshots/Upload%20XLSX%20File%20Result.png)

We can show the Data Frame from the XLSX as well as the menu to get detailed
analysis, we can also upload the file with multiple sheets.

#### Summary

##### Select Overall Summary

- Select the "Summary" menu; by default, it is automatically selected.

![Screenshot](screenshots/Overall%20Summary%20Part%201.png)

![Screenshot](screenshots/Overall%20Summary%20Part%202.png)

![Screenshot](screenshots/Overall%20Summary%20Part%203.png)

- We can display some statistics for the Overall Summary, such as :
  - Table Information
  - Columns Information
  - Column Data Types
  - Missing Values
  - Most / least Frequent Value(s) per Column
  - Statistics (where there are Numeric Columns in the table)

##### Select Summary by Column(s)

![Screenshot](screenshots/Selected%20by%20Column%20Summary%20Part%201.png)

![Screenshot](screenshots/Selected%20by%20Column%20Summary%20Part%202.png)

- We can select multiple columns in the summary by selected columns, where each
  column can have:
  - General Summary
  - Value Specific Summary (can select column to display selected data by
    clicking on the select box.)

#### Insight

- We can generate insight based on the displayed data frame, which is auto
  generated by AI.

##### Generate Insight

![Screenshot](screenshots/Generate%20Insight.png)

- Type the text input and click "Generate Insight"

![Screenshot](screenshots/Generated%20Insight%20Part%201.png)

![Screenshot](screenshots/Generated%20Insight%20Part%202.png)

- We managed to generate the insight based on the selected input, which is done
  by OpenAI API.

##### Generate Summary from Insight

![Screenshot](screenshots/Generate%20Summarized%20Insight%20Button.png)

- Try to click the button "Summarize Insight".

![Screenshot](screenshots/Summarized%20Insight.png)

- We managed to generate the summarized insight based on the generated insight
  by OpenAI API, which is done by HuggingFace Inference API.

#### Chart

##### Bar Chart

![Screenshot](screenshots/Bar%20Chart.png)

- Click the "Bar Chart" as the selected value, and then try to select the X-axis
  as well as Y-axis

![Screenshot](screenshots/Bar%20Chart%20Result.png)

- Try to click on "Display" button, which displays the Bar Chart that is
  downloadable by clicking on the "Download as PNG" button.

##### Pie Chart

![Screenshot](screenshots/Pie%20Chart.png)

- Click the "Pie Chart" as the selected value, and then try to select the
  column.

![Screenshot](screenshots/Pie%20Chart%20Result%20Part%201.png)

![Screenshot](screenshots/Pie%20Chart%20Result%20Part%202.png)

- Try to click on "Display" button, which displays the Pie Chart that is
  downloadable by clicking on the "Download as PNG" button.

##### Scatter Plot

![Screenshot](screenshots/Scatter%20Plot.png)

- Select "Scatter Plot" as the chart type, then choose the X-axis and Y-axis.

![Screenshot](screenshots/Scatter%20Plot%20Result.png)

- Click the "Display" button to show the scatter plot. You can download the plot
  by clicking the "Download as PNG" button.

### Link Application Deployment

- [AI Data Visualizer](https://ai-data-visualizer.streamlit.app/)
