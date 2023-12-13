# Aptos2023 Big Data Competition Baseline Codes

This repository contains the baseline codes for the Aptos2023 Big Data Competition. Two methods are provided:

## Method 1: Multi-Label Classification
To train the multi-label classification model:
```bash
cd method1_classification/code
python main_multi_images_angi.py
```
To test the trained model:
```bash
python test_angi.py
```

## Method 2: Report Generation
For report generation, the code relies on the BLIP library. Please refer to the [BLIP repository](https://github.com/salesforce/BLIP) for library versions and installation instructions.

To train the report generation model:
```bash
cd method2_report_generation
python train_angi_multi.py
```

To test the trained model:
```bash
python test_angi.py
```

**Note:** Please download the dataset before running any code.
