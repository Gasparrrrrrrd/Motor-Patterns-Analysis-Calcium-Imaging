# Calcium Imaging Analysis with ROI Processing

This code is a function for processing calcium imaging data with region of interest (ROI) analysis.

## Core Function Purpose

The *DFOF_optimized_roi* function calculates ΔF/F (delta F over F) from calcium imaging data, which represents relative changes in fluorescence intensity, a proxy for neural activity.

## Key Components

### 1. ROI Mask Creation
Converts circular ROI coordinates into circular masks to define regions for analysis. (does this by computing center from ROI coordinates, and then from center + radius computes distance map to categorise all pixels that fall within this range to be part of the mask: equal to 1, the rest is equal to 0, this is the binary mask).

### 2. Optogenetic Artifact Correction

To correct for optogenetic artifacts, we create a control region with identical shape but positioned outside the area of interest (10 pixels beyond the maximum x-coordinate).

- Creates a reference ROI away from the stimulation site (this reference ROI is at exactly the same y-coordinate, to ensure similar optogenetics stimulus artifact [the illumination moves exactly vertically, hence we can ensure, if we set the same y-coordinate, that the stimulus will be corrected exactly. The x-coordinate is chosen so that it falls outside of the VNC zone: no activity signal)
  - Establishes a pre-stimulus baseline for the reference region
  - Measures baseline fluorescence before stimulation **FOR EACH PIXEL** (baseline is very low so is just the mean of the time window before stimulation)
  - During stimulation frames, calculates the deviation from baseline for each pixel
  - Calculates stimulation artifacts to subtract from the actual signal **FOR EACH PIXEL** (so for each pixel you get oscillatory signal, that is different from all other pixels, apart the ones on the same y-axis).

OPTO STIMULATION ARTIFACT = stimulation artifact[during stim] -- baseline

- Stores these deviations as the presumed artifact signal (this is done in a dictionary, with keys being ROI names, like "A5r", each entry contains a 2d array of shape (number_frames*number_pixels_in_ROI), in our case something like (1519,151))

**→ when calculating traces (the average for an ROI of all the pixel fluorescence time series), and especially average dff, we compute for each pixel the df_f from which we subtract the optogenetics stimulation artifact.**

Creates a reference region with identical dimensions to maintain pixel-to-pixel correspondence!

**for each pixel:**

**actual signal = df_f - opto stimulation artifact / baseline(calculated from ROI of interest)**

```python
avg_pixel_df_f = np.mean(dff - opto_df/baseline, axis = 1)  # Calculate average dff across all pixels for this ROI
```

This is done in vectorised form to optimise code. Arrays dff, opto_df and baseline are (1519,151) arrays taken from corresponding dictionaries.

### 3. Technical Implementation Details

1. **Pixel-wise Processing**: Artifacts are calculated individually for each pixel
2. **Temporal Specificity**: Only applied during the stimulation period (frames stimulus_start to stimulus_start+stimulus_duration)
3. **Spatial Matching**: Creates a reference region with identical dimensions to maintain pixel-to-pixel correspondence!!
4. **Baseline Reference**: Uses pre-stimulus frames to establish the normal fluorescence level

This approach assumes that:

1. Artifacts affect the control region similarly to the ROIs
2. The control region contains no relevant neural activity
3. The artifact magnitude is additive and spatially uniform

### 4. ΔF/F Calculation

- Uses a sliding window approach to calculate baseline fluorescence
- For each pixel: calculates ΔF (signal - baseline) and ΔF/F (ΔF ÷ baseline)
- Handles edge cases at the beginning and end of recordings

### 5. Optimization Techniques

- Uses vectorized operations instead of loops where possible
- Processes signals pixel-by-pixel within ROIs
- Pre-allocates arrays for efficiency

## Output Values

The function returns four dictionaries:

- ROI baselines
- Raw fluorescence changes (ΔF)
- Normalized fluorescence changes (ΔF/F)
- Optogenetic artifacts

The results are organized by ROI name for further analysis.
