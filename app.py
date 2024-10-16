import streamlit as st
import pandas as pd
import gc

VALID_CLASSES = [
    "Angioectasia", "Bleeding", "Erosion", "Erythema",
    "Foreign Body", "Lymphangiectasia", "Normal",
    "Polyp", "Ulcer", "Worms"
]

EXPECTED_COLUMNS = ["image_path"] + VALID_CLASSES + ["predicted_class"]
REFERENCE_FILE_PATH = "file_names_list.csv"

sample_data = {
    "image_path": ["00Z0Xo99wp.jpg", "02hvtCoV9C.jpg", "03pjR51twC.jpg", "03UqLvuk8v.jpg"],
    "Angioectasia": [0.004180671, 5.31846E-06, 0.01316455, 1.81824E-06],
    "Bleeding": [1.24772E-09, 2.6963E-05, 7.35843E-05, 1.31746E-06],
    "Erosion": [0.007089304, 0.000178704, 0.000328246, 0.045647398],
    "Erythema": [1.42296E-07, 0.05799336, 0.009602185, 0.001173414],
    "Foreign Body": [0.038560923, 1.55969E-08, 0.190064773, 0.001607239],
    "Lymphangiectasia": [2.79127E-05, 0.00050396, 0.721927106, 1.16342E-05],
    "Normal": [0.018649779, 0.941272676, 4.30372E-05, 0.204296276],
    "Polyp": [6.76447E-08, 1.89326E-05, 0.018246552, 0.747260571],
    "Ulcer": [0.931491256, 6.26834E-08, 0.046549879, 3.51713E-07],
    "Worms": [3.32206E-17, 3.29838E-10, 1.20817E-07, 3.123E-09],
    "predicted_class": ["Ulcer", "Normal", "Lymphangiectasia", "Polyp"]
}
sample_df = pd.DataFrame(sample_data)

def main():
    st.title("Capsule Vision Challenge 2024 Sanity Checker")

    st.write("### Guidelines for Uploading Excel file")
    st.markdown("""
    1. **image_path** column should contain only the image name (e.g., `image.jpg`).
    2. **Predicted probabilities** for each class should be present along with the `predicted_class` column.
    3. **Predictions for all images** must be complete, with no blanks or missing values.
    4. Correct file can be generated from the code provided [here](https://github.com/misahub2023/Capsule-Vision-2024-Challenge/blob/main/sample_codes_for_participants/Evaluate_model.py)
    5. Excel file name should be same as team name.
    """)

    st.write("### Example of the Expected CSV Format")
    st.dataframe(sample_df, hide_index=True)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["xlsx"])
    
    if uploaded_file is not None:
        passed_all_checks = True  

        with st.spinner("Checking..."):
            try:
                df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading the file: {e}")
                return

            if 'image_path' not in df.columns:
                st.error("The 'image_path' column is missing from the uploaded file.")
                passed_all_checks = False
                return
            try:
                reference_df = pd.read_csv(REFERENCE_FILE_PATH)
                reference_images = set(reference_df["file_name"])
            except Exception as e:
                st.error(f"Error reading the reference file: {e}")
                return

            predicted_images = set(df["image_path"])
            missing_images = reference_images - predicted_images
            extra_images = predicted_images - reference_images

            if len(missing_images) > 10:
                st.error("Image names do not match; too many missing images.")
                passed_all_checks = False
            elif missing_images:
                st.error(f"The following images are missing from the predictions: {', '.join(missing_images)}")
                passed_all_checks = False

            if len(extra_images) > 10:
                st.error("There are extra images in the predictions.")
                passed_all_checks = False
            elif extra_images:
                st.error(f"The following images are extra and not in the reference file: {', '.join(extra_images)}")
                passed_all_checks = False

            missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
            extra_columns = [col for col in df.columns if col not in EXPECTED_COLUMNS]
            
            if missing_columns:
                st.error(f"The following columns are missing: {', '.join(missing_columns)}")
                passed_all_checks = False

            if extra_columns:
                st.error(f"The following columns are not expected: {', '.join(extra_columns)}")
                passed_all_checks = False

            if df.isnull().values.any():
                st.error("The CSV contains missing values.")
                missing_rows = df[df.isnull().any(axis=1)]["image_path"].tolist()
                if len(missing_rows) > 10:
                    st.write("Too many rows with missing predictions.")
                else:
                    st.write(f"Missing values found in rows with image_path: {', '.join(missing_rows)}")
                passed_all_checks = False

            invalid_classes = df[~df['predicted_class'].isin(VALID_CLASSES)]
            if not invalid_classes.empty:
                st.error("Some `predicted_class` values are invalid.")
                invalid_rows = invalid_classes["image_path"].tolist()
                if len(invalid_rows) > 10:
                    st.write("Too many invalid predicted class values.")
                else:
                    st.write(f"Invalid `predicted_class` entries found for image_path: {', '.join(invalid_rows)}")
                passed_all_checks = False

            if df['image_path'].duplicated().any():
                st.error("Duplicate image paths found.")
                duplicated_rows = df[df['image_path'].duplicated()]["image_path"].tolist()
                if len(duplicated_rows) > 10:
                    st.write("Too many duplicate image paths found.")
                else:
                    st.write(f"Duplicate entries found for image_path: {', '.join(duplicated_rows)}")
                passed_all_checks = False

        if passed_all_checks:
            st.success("All checks passed.")
        else:
            st.error("File did not pass all checks. Please see the errors above.")

        del df, reference_df, missing_columns, extra_columns, missing_images, extra_images, invalid_classes, uploaded_file
        gc.collect()

if __name__ == "__main__":
    main()
