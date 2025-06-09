# Calcium-Imaging-Analysis
Takes Image Stacks of Calcium Imaging and Associated ROIs and Computes dF/F and Baseline for all pixels, then produces an average Df/F and Baseline trace for each ROI. This activity trace is then smoothed and is plotted in a ridge-like plot to compare all ROIs visually. Finally, the correlation matrix between each ROI (that is between each motor area) is computed and subsequently clustered to infer motor pattern and area symmetry and synchrony.

## Example Calcium Image of VNC with motor patterns and corresponding ROIs annotated
![download](https://github.com/user-attachments/assets/fdbd59d9-2076-4fac-bb2e-b0f4000b678b)


## Example Ridge-like plot of all ROIs
![download](https://github.com/user-attachments/assets/1883b468-0a31-4dee-bdbc-abc89caa9c9b)



## Example Correlation Matrix of all ROIs
![download](https://github.com/user-attachments/assets/963fc04e-21e6-4485-a0a3-c1c8d171ccf9)


### More explanations can be found in the [description of the main function](https://github.com/Gasparrrrrrrd/Motor-Patterns-Analysis-Calcium-Imaging/blob/main/explanation_main_function_calcium-imaging-analysis-md.md)


## Methodology


The computational framework developed for this project, implemented in the repository “Motor-Patterns-Analysis-Calcium-Imaging,” is designed to facilitate the quantitative analysis of calcium imaging data acquired from neural tissue, enabling the extraction of fluorescence dynamics for regions of interest (ROIs) during various experimental conditions. The repository is implemented predominantly in Python within interactive Jupyter Notebooks, allowing for both transparent documentation of analytical workflow and reproducibility. The core of the analysis pipeline is embodied in two principal notebooks: (1) the main function and script notebook, and (2) a dedicated batch processing notebook for high-throughput analysis.


### Data Structure and Pre-processing

The pipeline is architected to process image stacks (typically, multi-page TIFF files) corresponding to time-series data from calcium imaging experiments. Each image stack is associated with a set of user-defined ROIs, typically encoded in formats compatible with ImageJ (e.g., .roi or .zip files containing ROI definitions). Data and associated ROI sets are hierarchically organized within a folder structure, facilitating batch processing at the sub-experiment or trial level.

At the outset, the pipeline mounts a Google Drive directory for persistent storage and rapid access to large datasets. It auto-detects images and associated ROI sets, ensuring that each imaging dataset is processed in conjunction with the correct set of anatomical or functional ROIs.


### Main Function: ROI-Based dF/F Computation

The central computational element is an optimized function for calculating the change in fluorescence over baseline (dF/F) for each ROI, tailored for high fidelity and robustness to experimental artifacts (such as optogenetic stimulation). The main steps are as follows:

#### 1. ROI Mask Construction

For each ROI, a binary mask is created by computing the geometric center and radius (as the maximal distance from the center to the ROI boundary). Then, a pixel-wise mask is constructed such that only pixels within the defined ROI contribute to subsequent calculations. This mask is applied to each frame of the image stack, extracting the fluorescence time-series for each ROI.

#### 2. Baseline Estimation and Artifact Correction

The pipeline employs a sliding window approach to estimate the baseline fluorescence for each ROI and pixel. The window size is dynamically adjusted: at the beginning and end of the time series, window size increases or decreases to accommodate edge effects. For the central portion of the series, a symmetric window of fixed duration (typically two seconds) is used. The baseline is calculated as the mean fluorescence within the window for each pixel within the ROI. ΔF is computed as the difference between the instantaneous and baseline fluorescence, and ΔF/F is the normalized ratio.

Very importantly, it also applies optogenetic artifact correction, for experiments involving optogenetic stimulation, the pipeline implements a robust artifact correction strategy. An artificial “reference ROI” is algorithmically placed outside the anatomical region of interest but within the image boundaries. The reference ROI is matched in size and shape to the original ROI, and its placement is determined via a algorithm that prioritizes the ROI to be in phase with the reference ROI, that is, exactly 1 optogentic wavelength away, so as to get accurate signal correction. The fluorescence dynamics from the reference ROI are used to estimate and subtract potential optogenetic artifacts from the true ROI signal.

#### 3. Vectorization and Efficiency

The entire pipeline is highly vectorized, using NumPy arrays for efficient memory access and computation. Where possible, pixel-wise operations are performed in parallel, and smoothing operations leverage optimized libraries (e.g., Whittaker-Eilers smoothing for denoising, if enabled). 

#### 4. Artifact Correction Heuristics

Frames corresponding to known experimental artifacts (e.g., frame 759, which may coincide with a stimulus switch) are interpolated or replaced with adjacent data points to minimize the impact on downstream analyses.

---

### Batch Processing Pipeline

The Batch_Processing_Calcium_Imaging.ipynb notebook extends the main analytical function to enable automated, high-throughput processing of entire experimental campaigns. The batch processing workflow is characterized by the following features:

#### 1. User Configuration and Parameterization

The user specifies the root directory containing all experimental data. Then, the global parameters such as optogenetic stimulation status, stimulus onset frame, and stimulus duration are entered interactively or defaulted to standard values.

#### 2. Automated Traversal of Folder Structure

The notebook recursively traverses the directory tree, processing each sub-folder (corresponding to different experimental conditions or subjects). Within each sub-folder, all image folders are processed sequentially. Each image folder typically contains a single experiment (e.g., a single trial or replicate).

#### 3. Data Quality and Integrity Checks

For each image folder, the presence of both the TIFF stack and corresponding ROI set is verified. If either is missing or corrupted, the pipeline logs the error and skips the affected folder, ensuring that processing of subsequent datasets continues uninterrupted.

#### 4. Parallel and Sequential Processing

Each image is loaded, and the main function is invoked for each ROI set. Processing status, including successful completion or errors (with detailed traceback and error messages), is output in real time for user review. The pipeline prints the anatomical labels of each ROI as they are processed, providing transparency and traceability.

#### 5. Output and Post-processing

For each successfully processed dataset, the pipeline generates output files containing baseline, dF, and dF/F traces for each ROI. Diagnostic plots are generated using Matplotlib and Seaborn, including time-series traces and cluster heatmaps for exploratory data analysis. All outputs are stored in a structured format, compatible with downstream statistical and visualization workflows.

---

### Computational Subtleties and Robustness

The pipeline is distinguished by several computational subtleties that enhance its scientific rigor:

-**Placement Algorithms:** The placement of reference ROIs for artifact correction is non-trivial, involving a series of geometric checks to ensure that the reference ROI is both outside the region of interest and within image boundaries, while maintaining similar spatial properties for accurate artifact estimation.
- **Edge Effect Management:** The use of dynamically sized sliding windows at the beginning and end of time series ensures baseline estimation accuracy even in short recordings.
- **Automated Exception Handling:** The batch processing script is resilient to a variety of data integrity issues, logging errors and skipping problematic datasets without halting the entire batch process.


---
