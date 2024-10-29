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
TRAINING_FILE_PATH = "training_data.xlsx"
VALIDATION_FILE_PATH = "validation_data.xlsx"

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

    # Dropdown to select between Test, Training, and Validation modes
    mode = st.selectbox("Select Mode", ["Test", "Training", "Validation"])

    if mode == "Test":
        test_mode()
    elif mode == "Training":
        training_mode()
    else:
        validation_mode()

def test_mode():
    """Sanity check logic for Test Mode."""
    st.write("### Guidelines for Uploading Excel file")
    st.markdown("""
    1. **image_path** column should contain only the image name (e.g., `image.jpg`).
    2. **Predicted probabilities** for each class should be present along with the `predicted_class` column.
    3. **Predictions for all images** must be complete, with no blanks or missing values.
    4. Correct file can be generated from the code provided [here](https://github.com/misahub2023/Capsule-Vision-2024-Challenge/blob/main/sample_codes_for_participants/Evaluate_model.py)
    5. Excel file name should be the same as the team name.
    """)

    st.write("### Example of the Expected Excel Format")
    st.dataframe(sample_df, hide_index=True)

    uploaded_file = st.file_uploader("Upload your Test CSV file", type=["xlsx"])

    if uploaded_file is not None:
        with st.spinner("Checking..."):
            try:
                df = pd.read_excel(uploaded_file)
                reference_df = pd.read_csv(REFERENCE_FILE_PATH)
                reference_images = set(reference_df["file_name"])
                check_file(df, reference_images, EXPECTED_COLUMNS)
            except Exception as e:
                st.error(f"Error during processing: {e}")

def training_mode():
    """Sanity check logic for Training Mode."""
    st.write("### Upload your Training Excel file")
    uploaded_file = st.file_uploader("Upload your Training CSV file", type=["xlsx"])

    if uploaded_file is not None:
        passed_all_checks = True
        with st.spinner("Checking..."):
            try:
                df = pd.read_excel(uploaded_file)
                training_df = pd.read_excel(TRAINING_FILE_PATH)

                # Perform the checks
                passed_all_checks = check_file_dimensions_and_columns(df, training_df, "training")
            except Exception as e:
                st.error(f"Error during validation: {e}")
                passed_all_checks = False

        if passed_all_checks:
            st.success("All checks passed.")
        else:
            st.error("File did not pass all checks. Please see the errors above.")

        del df, training_df, uploaded_file
        gc.collect()

    # Download button for the true training file
    st.write("If needed, you can download the true training file here:")
    with open(TRAINING_FILE_PATH, "rb") as file:
        btn = st.download_button(
            label="Download True Training File",
            data=file,
            file_name="training_data.xlsx"
        )

def validation_mode():
    """Sanity check logic for Validation Mode."""
    st.write("### Upload your Validation Excel file")
    uploaded_file = st.file_uploader("Upload your Validation CSV file", type=["xlsx"])

    if uploaded_file is not None:
        passed_all_checks = True
        with st.spinner("Checking..."):
            try:
                df = pd.read_excel(uploaded_file)
                validation_df = pd.read_excel(VALIDATION_FILE_PATH)

                # Perform the checks
                passed_all_checks = check_file_dimensions_and_columns(df, validation_df, "validation")
            except Exception as e:
                st.error(f"Error during validation: {e}")
                passed_all_checks = False

        if passed_all_checks:
            st.success("All checks passed.")
        else:
            st.error("File did not pass all checks. Please see the errors above.")

        del df, validation_df, uploaded_file
        gc.collect()

    # Download button for the true validation file
    st.write("If needed, you can download the true validation file here:")
    with open(VALIDATION_FILE_PATH, "rb") as file:
        btn = st.download_button(
            label="Download True Validation File",
            data=file,
            file_name="validation_data.xlsx"
        )

def check_file_dimensions_and_columns(df, reference_df, mode):
    """Checks dimensions and columns between uploaded file and the reference file."""
    passed_all_checks = True

    # Checking dimensions
    if df.shape != reference_df.shape:
        st.error(f"The number of rows or columns do not match with the {mode} file.")
        passed_all_checks = False

    # Checking column names
    if not (df.columns == reference_df.columns).all():
        st.error(f"The column names do not match with the {mode} file.")
        passed_all_checks = False

    # Checking image_path matches
    if not set(df["image_path"]) == set(reference_df["image_path"]):
        st.error(f"The image_path column entries do not match with the {mode} file.")
        passed_all_checks = False

    return passed_all_checks

def check_file(df, reference_images, expected_columns):
    """Performs sanity checks for the uploaded Test file."""
    passed_all_checks = True

    # Check for missing or extra images
    predicted_images = set(df["image_path"])
    missing_images = reference_images - predicted_images
    extra_images = predicted_images - reference_images

    if len(missing_images) > 10:
        st.error("Image names do not match; too many missing images.")
        passed_all_checks = False
    elif missing_images:
        st.error(f"The following images are missing: {', '.join(missing_images)}")
        passed_all_checks = False

    if len(extra_images) > 10:
        st.error("There are extra images in the predictions.")
        passed_all_checks = False
    elif extra_images:
        st.error(f"The following images are extra: {', '.join(extra_images)}")
        passed_all_checks = False

    # Check for missing or extra columns
    missing_columns = [col for col in expected_columns if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in expected_columns]

    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}")
        passed_all_checks = False

    if extra_columns:
        st.error(f"Extra columns: {', '.join(extra_columns)}")
        passed_all_checks = False

    # Check for missing values
    if df.isnull().values.any():
        st.error("The file contains missing values.")
        passed_all_checks = False

    # Check for invalid predicted_class values
    invalid_classes = df[~df["predicted_class"].isin(VALID_CLASSES)]
    if not invalid_classes.empty:
        st.error("Invalid predicted_class values found.")
        passed_all_checks = False

    if passed_all_checks:
        st.success("All checks passed.")
    else:
        st.error("File did not pass all checks.")

if __name__ == "__main__":
    main()
