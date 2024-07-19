import streamlit as st
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Constants
IMG_SIZE = 128  # Change this based on your model's input size

# Load your trained model
model = load_model("3D_MRI_Brain_tumor_segmentation.h5")

# Define segmentation classes
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
    2: 'EDEMA',
    3: 'ENHANCING'  # original 4 -> converted into 3 later
}

# Preprocess images
def preprocess_image(image_file):
    img = nib.load(image_file).get_fdata()
    img_resized = cv2.resize(img[:, :, 0], (IMG_SIZE, IMG_SIZE))  # Resize and take the first slice
    return img_resized / np.max(img_resized)  # Normalize

# Predict segmentation for dual images
def predict(flair_image, t1ce_image):
    X = np.zeros((1, IMG_SIZE, IMG_SIZE, 2))
    X[0, :, :, 0] = preprocess_image(flair_image)
    X[0, :, :, 1] = preprocess_image(t1ce_image)

    pred = model.predict(X)
    pred_img = np.argmax(pred, axis=-1)[0]
    return pred_img

# Predict segmentation for single image
def predict_single_image(image):
    input_data = np.expand_dims(image, axis=(0, -1))
    input_data = input_data / np.max(input_data)  # Normalize

    pred = model.predict(input_data)
    pred_img = np.argmax(pred, axis=-1)[0]
    return pred_img

# Streamlit app interface
st.title('MRI Brain Tumor Segmentation')

mode = st.radio("Choose Segmentation Mode", ("Single Image", "Dual Image"))

if mode == "Single Image":
    st.write("Upload a 3D MRI scan for segmentation")
    uploaded_file = st.file_uploader("Choose a NIFTI file", type="nii")

    if uploaded_file is not None:
        # Load and display the MRI scan
        mri_scan = nib.load(uploaded_file)
        mri_data = mri_scan.get_fdata()

        slice_num = st.slider("Choose slice number", 0, mri_data.shape[2] - 1, mri_data.shape[2] // 2)
        plt.imshow(mri_data[:, :, slice_num], cmap="gray")
        plt.title(f"Slice {slice_num}")
        st.pyplot(plt)

        if st.button("Run Segmentation"):
            # Predict segmentation
            prediction = predict_single_image(mri_data)

            # Display the segmented result
            plt.imshow(prediction[:, :, slice_num])
            plt.title(f"Segmented Slice {slice_num}")
            st.pyplot(plt)

elif mode == "Dual Image":
    st.write("Upload MRI images:")
    flair_image = st.file_uploader("Upload FLAIR Image (NIfTI)", type=["nii", "nii.gz"])
    t1ce_image = st.file_uploader("Upload T1CE Image (NIfTI)", type=["nii", "nii.gz"])

    if flair_image and t1ce_image:
        st.write("Processing...")
        pred_img = predict(flair_image, t1ce_image)

        fig, ax = plt.subplots()
        cax = ax.imshow(pred_img, cmap='viridis')
        cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['NOT tumor', 'NECROTIC/CORE', 'EDEMA', 'ENHANCING'])
        st.pyplot(fig)

st.write("""
**Note:**
- **Single Image Mode**: Upload one NIfTI file for segmentation.
- **Dual Image Mode**: Upload two NIfTI files - one for FLAIR and one for T1CE MRI images.
- **File Format**: Only .nii or .nii.gz files are accepted.
- **Output**: The model will return a segmented image where different tumor types are marked with different colors.
""")
