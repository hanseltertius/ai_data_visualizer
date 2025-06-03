from classes.charthandler import ChartHandler
import matplotlib.pyplot as plt

class BarChart(ChartHandler):
    def generate(self, df, x_axis, y_axis, selected_file_name, selected_sheet_name = ""):
        # region Generate Bar Chart
        chart_title = f"Summary of {selected_file_name} in {selected_sheet_name}" if selected_sheet_name else f"Summary of {selected_file_name}"
        num_x = len(df[x_axis].unique())
        fig_width = min(max(8, num_x * 0.5), 40) # Dynamically set width: 0.5 inch per x label, min 8, max 40
        
        # Generate color list using colormap
        colormap = plt.cm.get_cmap('tab10', num_x)
        colors = [colormap(i) for i in range(num_x)]

        figure, axes = plt.subplots(figsize=(fig_width, 6))
        axes.bar(
            df[x_axis].astype(str), 
            df[y_axis],
            color=colors
        )
        axes.set_xlabel(x_axis)
        axes.set_ylabel(y_axis)
        axes.set_title(chart_title)
        plt.xticks(rotation=45, ha="center")
        plt.yticks(rotation=315, va="center")
        plt.tight_layout()

        return figure
        # endregion