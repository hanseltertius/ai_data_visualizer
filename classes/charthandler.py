import io
import base64

class ChartHandler:
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