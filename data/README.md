# Sample Data

This directory contains sample data for testing CampyML predictions and training.

## Files

- **sample_data.csv**: 10 diverse Campylobacter samples for prediction testing
  - Complete cgMLST profiles with 1171 features
  - No source labels (ready for prediction)

- **training_data.csv**: 10 samples with source labels for training testing
  - Same format as sample_data.csv PLUS 'source' column
  - 4 source classes: chicken, cattle, sheep, environmental waters

## Quick Test - Predictions

```bash
# From the project root directory:
docker run --rm -v $(pwd):/workspace ghcr.io/genpat-it/campy-ml:latest \
  --mode predict \
  --data /workspace/data/sample_data.csv \
  --model /app/models/modello_xgb_jejuni_coli_pubmlst_IZS_v2.pkl \
  --output /workspace/data/predictions.csv

# View results
cat data/predictions.csv
```

## Quick Test - Training

```bash
# From the project root directory:
docker run --rm -v $(pwd):/workspace ghcr.io/genpat-it/campy-ml:latest \
  --mode train \
  --data /workspace/data/training_data.csv \
  --model /workspace/data/my_new_model.pkl \
  --target source
```

## Data Format

Both files contain genomic features including:
- **Metadata**: ID, Location, Species
- **MLST alleles**: aspA, glnA, gltA, glyA, pgm, tkt, uncA (7 loci)
- **cgMLST alleles**: CAMP0001 to CAMP1164 (1164 loci)
- **Total features**: 1171 (excluding metadata)

For training data, an additional `source` column contains the target labels.

## Expected Output

- **Predictions**: CSV with prediction, confidence, and probabilities for each source class
- **Training**: New trained model saved as .pkl file with performance metrics

The sample data provides immediate plug-and-play functionality for testing both prediction and training modes.