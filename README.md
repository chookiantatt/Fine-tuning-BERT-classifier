# Fine-tuning multi-label material classifier(BERT)


****A material classification task using BERT (Bidirectional Encoder Representations from Transformers) to predict multiple classes for chemical/biological substances.****

**Overview**

Targeted at chemical and biological substances, this classifier can predict the classes that a given substance belongs to. 
It leverages the powerful BERT language model, fine-tuned on labeled datasets, to enable accurate multi-label classification.

Features

* Accepts substance names in various formats: IUPAC, trade names, scientific names, compound names, SMILES, and InChIKey
* Utilizes a custom tokenizer augmented with filtered subunits extracted from PubChem (https://pubchem.ncbi.nlm.nih.gov/), an open-source substance database
* Leverages pre-trained BERT models (bert-base-cased or bert-base-uncased) from the Transformers library
* Fine-tuned on labeled datasets for optimal performance

Usage

Aluminium oxide --> Inorganic

Sapphire --> Mineral

PVC --> Polymer

Polyurethane --> Polymer

Fe<sub>2</sub>O<sub>3</sub>  --> Inorganic

Fe.Cu.Mn  --> Alloy

Al<sub>2</sub>O<sub>3</sub>  --> Mineral, Inorgic

Tungsten Carbide(WC)  --> Compound
