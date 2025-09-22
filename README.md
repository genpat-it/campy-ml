# CampyML - Machine Learning for Campylobacter Source Attribution

A machine learning model for predicting the source (host species) of Campylobacter isolates based on genomic data using XGBoost.

## Quick Start - Try It Now!

```bash
# Clone and enter the repository
git clone https://github.com/genpat-it/campy-ml.git
cd campy-ml

# Run prediction on sample data (10 diverse Campylobacter samples)
docker run --rm -v $(pwd):/workspace ghcr.io/genpat-it/campy-ml:latest \
  --mode predict \
  --data /workspace/data/sample_data.csv \
  --model /app/models/modello_xgb_jejuni_coli_pubmlst_IZS_v2.pkl \
  --output /workspace/data/predictions.csv

# Check results
cat data/predictions.csv
```

## Overview

CampyML uses whole genome MLST (wgMLST) profiles to predict the likely source of Campylobacter jejuni and Campylobacter coli isolates. The model is trained on data from pubMLST combined with internal IZS sequences to predict sources such as:
- Chicken
- Cattle
- Sheep
- Turkey
- Environmental waters
- Human stool
- Other sources

## Usage

### Making Predictions

```bash
# Pull the latest image
docker pull ghcr.io/genpat-it/campy-ml:latest

# Run prediction on your data
docker run --rm -v $(pwd):/workspace ghcr.io/genpat-it/campy-ml:latest \
  --mode predict \
  --data /workspace/your_samples.csv \
  --model /app/models/modello_xgb_jejuni_coli_pubmlst_IZS_v2.pkl \
  --output /workspace/predictions.csv
```

### Training a New Model

```bash
# Prepare training data with 'source' column
# Then train a new model
docker run --rm -v $(pwd):/workspace ghcr.io/genpat-it/campy-ml:latest \
  --mode train \
  --data /workspace/training_data.csv \
  --model /workspace/my_new_model.pkl \
  --target source
```

### Local Development

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run predictions locally
python campyml_model.py --mode predict \
  --data data/sample_data.csv \
  --model models/modello_xgb_jejuni_coli_pubmlst_IZS_v2.pkl \
  --output data/predictions.csv

# Train locally
python campyml_model.py --mode train \
  --data data/training_data.csv \
  --model data/my_model.pkl \
  --target source
```

## Input Data Format

### For Predictions
- CSV file with genomic features (cgMLST allele profiles)
- Required columns: ID, Location, Species, aspA, glnA, gltA, glyA, pgm, tkt, uncA, CAMP0001-CAMP1164
- No source labels needed

### For Training
- Same format as predictions PLUS
- Additional column: `source` (target labels like "chicken", "cattle", etc.)

## Output

The prediction output includes:
- `prediction`: Predicted source/host species
- `confidence`: Confidence score for the prediction
- `prob_[source]`: Probability for each possible source class

## Model Details

The current model (`modello_xgb_jejuni_coli_pubmlst_IZS_v2.pkl`) features:
- 1171 cgMLST features for maximum accuracy
- RandomForest classifier optimized for Campylobacter data
- Trained on diverse samples including pubMLST and IZS data
- 9 source classes: chicken, cattle, sheep, environmental waters, human stool, etc.

## Requirements

- Docker (recommended) OR Python 3.8+
- At least 4GB RAM for model operations
- Input data in CSV format with cgMLST profiles

## Sample Data

The repository includes `data/sample_data.csv` with 10 test samples for immediate experimentation.

## Data Sources

The training approach is based on the pubMLST database methodology described in:

> Arning N, Sheppard SK, Bayliss S, Clifton DA, Wilson DJ (2021) **Machine learning to predict the source of campylobacteriosis using whole genome data.** *PLoS Genet* 17(10): e1009436. https://doi.org/10.1371/journal.pgen.1009436

**Dataset characteristics:**
- 5,799 C. jejuni and C. coli genomes from pubMLST
- Sources: chicken (4,147), cattle (716), sheep (584), wild bird (212), environment (140)
- cgMLST approach with 1,343 core genes for enhanced accuracy
- Public data available at: https://pubmlst.org/bigsdb?db=pubmlst_campylobacter_isolates&page=query&project_list=102&submit=1

## Credits

**Developer**: Laura Di Egidio (Master's thesis project)
**Organization**: IZS Teramo - GenPat Project
**Contact**: For questions and support, please open an issue on GitHub

## Citation

If you use CampyML in your research, please cite:
```
CampyML: Machine Learning for Campylobacter Source Attribution
Laura Di Egidio (Master's thesis), IZS Teramo - GenPat Project
https://github.com/genpat-it/campy-ml
```

## License

This project is part of the GenPat initiative at IZS Teramo.

## Contact

For questions and support, please open an issue on GitHub.