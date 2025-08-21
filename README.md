This project applies federated learning to predict maternal health risk (Low, Medium, High) by collaboratively training a neural network model across multiple hospitals, using Flower and PyTorch. The system maintains patient data privacy—raw data never leaves any hospital.

Features:

Federated Learning (FL): Using Flower to enable distributed, privacy-preserving model training.

Maternal Health Risk Dataset: Predicts risk level (Low, Medium, High) using features like BP, age, blood sugar, etc.

Global Data Scaling: All data globally scaled before splitting for consistent feature distributions.

Customizable PyTorch Model: Modular MLP architecture, supports easy hyperparameter tuning and extension.

Fair Aggregation: Clients' updates combined with weighted averaging based on local data size.

Classification Report: Per-client classification breakdown for transparency and diagnostics.
