# Experimental Data Summary

## Data Location
Path: `public/BrightnessFunctionMixAndPhaseData/`

## Data Overview
This directory contains CSV files from visual perception experiments focusing on brightness function mixing and phase manipulation. The data appears to be from vection (perception of self-motion) experiments with various brightness blend modes.

## Data Structure
Each CSV file contains the following columns:
- **FrondFrameNum**: Frame number for front stimulus
- **FrondFrameLuminance**: Luminance value of front frame
- **BackFrameNum**: Frame number for back stimulus  
- **BackFrameLuminance**: Luminance value of back frame
- **Time**: Time in milliseconds
- **Knob**: Control knob value (0-1 range)
- **ResponsePattern**: Pattern type (e.g., "Velocity")
- **StepNumber**: Step in the experiment sequence
- **Amplitude**: Stimulus amplitude
- **Velocity**: Stimulus velocity
- **FunctionRatio**: Ratio parameter for function mixing
- **CameraSpeed**: Camera movement speed

## Participants
The data includes experiments from 5 participants:
- **OMU** (6 FunctionMix trials + 3 Phase trials each for Dynamic/LinearOnly modes)
- **HOU** (6 FunctionMix trials + 3 Phase trials each for Dynamic/LinearOnly modes)
- **LL** (6 FunctionMix trials + 3 Phase trials each for Dynamic/LinearOnly modes)
- **ONO** (6 FunctionMix trials + 3 Phase trials each for Dynamic/LinearOnly modes)
- **YAMA** (6 FunctionMix trials + 3 Phase trials each for Dynamic/LinearOnly modes)

## Experiment Types
1. **FunctionMix**: Experiments focusing on mixing different brightness functions
2. **Phase**: Experiments with phase manipulation and two brightness blend modes:
   - **Dynamic**: Dynamic brightness blending
   - **LinearOnly**: Linear-only brightness blending

## File Naming Convention
Format: `YYYYMMDD_HHMMSS_Fps1_CameraSpeed1_ExperimentPattern_[TYPE]_ParticipantName_[NAME]_TrialNumber_[N]_[BrightnessBlendMode_MODE].csv`

Where:
- Date and time stamp
- Fixed parameters (Fps1, CameraSpeed1)
- Experiment pattern type (FunctionMix or Phase)
- Participant name
- Trial number (1-6 for FunctionMix, 1-3 for Phase)
- Brightness blend mode (Dynamic or LinearOnly, for Phase experiments only)

## Data Statistics
- **Total Files**: 60 CSV files
- **Date Range**: July 10-16, 2025
- **File Sizes**: Range from 80KB to 3.4MB
- **Data Points**: Varying from ~1,000 to ~15,000 rows per file

## Test Files
Some files are marked as "_Test" indicating preliminary or calibration runs.

## Data Quality
The data appears to be well-structured with consistent formatting across all files. Time series data shows smooth transitions in luminance values and other parameters, suggesting good data quality and proper experimental control.