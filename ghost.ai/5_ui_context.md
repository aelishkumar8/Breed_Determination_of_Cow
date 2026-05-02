# UI Context

Contains design tokens and component conventions. While this is primarily a backend/ML pipeline currently, these guidelines apply to any visualizations (like bounding box plots) or future dashboard interfaces (e.g., Streamlit/Gradio).

## Visualizations (Matplotlib/OpenCV)
- **Bounding Boxes:** Use distinct, high-contrast colors (e.g., Neon Green `#39FF14` or Bright Red `#FF0000`) so they stand out against natural backgrounds (grass/dirt).
- **Text Labels:** Must have a contrasting background block (e.g., white text on a black semi-transparent rectangle) to ensure readability over the image.
- **Cropped Previews:** When plotting pipeline results, show the original image alongside the isolated OBB crop for easy visual debugging.

## Future Dashboard (Streamlit/Gradio) - Placeholder
- **Primary Color:** TBD (Cow/Farm theme - e.g., deep green or earthy brown)
- **Layout:** Two-column layout preferred (Upload/Input on left, Results/Visualizations on right).
- **Feedback:** Always provide a loading spinner during model inference.
