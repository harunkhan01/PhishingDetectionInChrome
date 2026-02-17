# PhishingDetectionInChrome

## ML Models

We implement several machine learning models to perform phishing detection including: an autoencoder for anomaly detection and a binary classifier.

### Model 1 : Char Encodings + AutoEncoder trained on negative samples

Model 1 performs highly accurate on discriminating between negative and positve samples of phishing links. This high accuracy (near 1) is due to the relative
easiness of distinguishing links from trancos top 1-million list and phishing links. This lack of FPR indicates that the model may not perform as well in practice. 

### Model 2: Char Encodings + AutoEncoder trained on negative samples from Common Crawl Dataset

Model 2 is trained on more common URLs rather than the top 1 million list provided by Tranco. Instead M2 is trained on...? 