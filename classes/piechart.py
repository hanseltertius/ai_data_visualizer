from classes.charthandler import ChartHandler
import matplotlib.pyplot as plt

class PieChart(ChartHandler):
    def generate(self, df, column, selected_file_name, selected_sheet_name = ""):
        # region Generate Pie Chart
        chart_title = f"Pie Chart of {column} in {selected_file_name}" if not selected_sheet_name else f"Pie Chart of {column} in {selected_file_name} ({selected_sheet_name})"
        value_counts = df[column].value_counts(dropna=False)
        figure, axes = plt.subplots(figsize=(8, 8))
        axes.pie(value_counts, labels=value_counts.index.astype(str), autopct='%1.1f%%', startangle=90)
        axes.set_title(chart_title)
        plt.tight_layout()

        return figure
        # endregion