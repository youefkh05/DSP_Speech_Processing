# DSP Speech Processing Assignment

Digital Signal Processing project implementing speech analysis, synthesis, and recognition using MATLAB.

## 📋 Overview

Solutions to 7 speech processing problems:
- **Problems 1-5**: Window analysis, LPC coefficient calculation, formant estimation
- **Problem 6**: Arabic speech vocoder (analysis & synthesis)
- **Problem 7**: Digit recognition system (0-9) using MFCC and DTW

## 🛠️ Requirements

- MATLAB R2018b or higher
- Signal Processing Toolbox
- Audio Toolbox (optional, for advanced features)

## 📁 Repository Structure

```
dsp-speech-processing/
├── src/
│   ├── problem1_windows.m
│   ├── problem2_lpc_autocorr.m
│   ├── problem3_lpc_frames.m
│   ├── problem4_pole_analysis.m
│   ├── problem5_lpc_spectrum.m
│   ├── problem6_vocoder/
│   │   ├── vocoder_main.m
│   │   ├── speech_analysis.m
│   │   ├── speech_synthesis.m
│   │   ├── pitch_detection.m
│   │   └── voice_unvoice_decision.m
│   └── problem7_recognition/
│       ├── extract_mfcc.m
│       ├── dtw_distance.m
│       └── recognize_digit.m
├── data/
│   ├── vocoder_input.wav
│   ├── vocoder_output.wav
│   └── digits/
├── results/
│   └── plots/
└── docs/
    ├── assignment_report.pdf
    └── improvements_presentation.pptx
```



## 📊 Key Parameters

- **Sampling Rate**: 16 kHz
- **Frame Size**: 20-30 ms (320-480 samples)
- **Frame Overlap**: 50%
- **Pre-emphasis**: α = 0.96
- **LPC Order**: 8-12

## 👨‍💻 Author

Youssef Khaled - DSP-1 Speech Processing Course

## 📚 References

- Rabiner & Schafer (2007). *Introduction to Digital Speech Processing*
- [Mean Opinion Score](https://en.wikipedia.org/wiki/Mean_opinion_score)
- [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)
