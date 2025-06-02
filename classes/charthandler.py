import matplotlib.pyplot as plt
import io
import base64

class ChartHandler:
    # TODO : generate_image dan juga get_buffer_image (base class) => pake barchart, dkk itu untuk derived class
    def generate_bar_chart(self, df, x_axis, y_axis, selected_file_name, selected_sheet_name = ""):
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

    def generate_pie_chart(self, df, column, selected_file_name, selected_sheet_name = ""):
        # region Generate Pie Chart
        chart_title = f"Pie Chart of {column} in {selected_file_name}" if not selected_sheet_name else f"Pie Chart of {column} in {selected_file_name} ({selected_sheet_name})"
        value_counts = df[column].value_counts(dropna=False)
        figure, axes = plt.subplots(figsize=(8, 8))
        axes.pie(value_counts, labels=value_counts.index.astype(str), autopct='%1.1f%%', startangle=90)
        axes.set_title(chart_title)
        plt.tight_layout()

        return figure
        # endregion

    def generate_scatter_plot(self, df, x_axis, y_axis, selected_file_name, selected_sheet_name=""):
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

    def generate_image(self, buffer, fig_width, fig_height):
        # region Generate Image

        # Display image in a scrollable div
        dpi = 100  # Default matplotlib DPI
        img_width_px = int(fig_width * dpi)
        img_height_px = int(fig_height * dpi)

        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"""
        <div style="overflow-x: auto; overflow-y: auto; width: 100%; max-height: 800px; padding-bottom: 8px; margin-bottom: 8px">
            <img 
                style="display: block; min-width: {img_width_px}px; min-height: {img_height_px}px; width: auto; height: auto;" 
                src="data:image/png;base64,{img_base64}" />
        </div>
        """
        # endregion

    def get_buffer_image(self, figure):
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)

        return buffer