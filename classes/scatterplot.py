from classes.charthandler import ChartHandler
import matplotlib.pyplot as plt

class ScatterPlot(ChartHandler):
    def generate(self, df, x_axis, y_axis, selected_file_name, selected_sheet_name=""):
        # region Generate Scatter Plot
        chart_title = f"Scatter Plot of {y_axis} vs {x_axis} in {selected_file_name}" if not selected_sheet_name else f"Scatter Plot of {y_axis} vs {x_axis} in {selected_file_name} ({selected_sheet_name})"
        figure, axes = plt.subplots(figsize=(8, 6))
        axes.scatter(
            df[x_axis], df[y_axis], 
            c=df[y_axis], cmap="viridis", alpha=0.7)
        axes.set_xlabel(x_axis)
        axes.set_ylabel(y_axis)
        axes.set_title(chart_title)
        plt.xticks(rotation=45, ha="center")
        plt.yticks(rotation=315, va="center")
        plt.tight_layout()

        return figure
        # endregion