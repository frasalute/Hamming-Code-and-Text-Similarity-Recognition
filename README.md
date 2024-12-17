# Hamming Code and Text Similarity Recognition

This repository contains Python implementations for two key tasks: **Hamming Code for error detection/correction** and **text similarity recognition**. The project was developed as part of a Python Programming class in collaboration with three other students.

---

## **Project Overview**

The repository addresses two distinct computational problems:

### **1. Hamming Code: Error Detection and Correction**
The Hamming Code implementation focuses on encoding, detecting, and correcting errors in binary messages. This task is part of coding theory, which ensures reliable communication over noisy channels.

#### **Key Features**:
- **Encoding**: Implements the **(7,4) Hamming Code** algorithm to encode 4-bit data into 7-bit codewords using 3 parity bits.
- **Error Introduction**: Simulate single-bit and two-bit errors to test the algorithm’s robustness.
- **Error Detection**: Using a parity check matrix, the program identifies the position of single-bit errors.
- **Error Correction**: Corrects single-bit errors while highlighting limitations in handling two-bit errors.
- **Decoding**: Converts the corrected binary code back to the original message.

#### **Usage**:
You can initialize the `HammingCode` class, encode numbers (0-15), introduce errors, and simulate error detection and correction.

Example:
```python
message = HammingCode(10)
message.number_to_vector()
message.encode()
message.introduce_error(1, 3)  # Introduce 1-bit error at position 3
message.paritycheck(True)
message.correct_error()
message.decode(True)
```

---

### **2. Text Similarity Recognition**
The text similarity tool compares multiple documents and calculates similarity scores using vector-based approaches.

#### **Key Features**:
- **Document Loading**: Read `.txt` files from a specified folder or single document.
- **Word Vectorization**: Convert documents into binary and frequency-based vectors.
- **Similarity Methods**:
   - **Dot Product**  
   - **Euclidean Distance Norm**  
   - **Cosine Similarity**  
   - **Frequency-Based Cosine Similarity**  
- **Customizable**: Allows users to compare documents with specific methods or all methods simultaneously.
- **Output**: Provides a ranked DataFrame of similarity scores for a given document against all other documents in the corpus.

#### **Usage**:
Initialize the `DocumentSimilarity` class, add documents to the corpus, and compute similarity scores.

Example:
```python
document_similarity = DocumentSimilarity()
document_similarity.read_all_files("Documents_Q3")
print(document_similarity.add_doc_compute_similarity('Documents_Q3/DemocInn.txt', ['freq-cos','cos','dot', 'norm']))
```

---

## **How to Run the Project**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Hamming-Code-and-Text-Similarity-Recognition.git
   cd Hamming-Code-and-Text-Similarity-Recognition
   ```

2. **Setup Environment**:
   Ensure Python is installed (version 3.7+ recommended). Install required libraries:
   ```bash
   pip install numpy pandas
   ```

3. **Run Hamming Code Example**:
   Execute the provided Hamming Code examples by running the script:
   ```bash
   python hamming_code_example.py
   ```

4. **Run Text Similarity Example**:
   Place `.txt` documents into a folder (e.g., `Documents_Q3`) and execute:
   ```bash
   python text_similarity_example.py
   ```

---

## **Folder Structure**

```
Hamming-Code-and-Text-Similarity-Recognition/
│
├── hamming_code.py               # Hamming Code implementation
├── text_similarity.py            # Text Similarity implementation
├── Documents_Q3/                 # Folder containing sample text files
├── README.md                     # Project documentation
└── requirements.txt              # List of required libraries
```

---

## **References**

- Epp, S. (2011). *Discrete Mathematics with Applications*.
- Nuh, F. (2007). *Coding Theory for Reliable Communications*.

---

## **Future Improvements**
- Extend Hamming Code to support multi-bit error correction.
- Enhance text similarity by integrating advanced NLP techniques like TF-IDF and BERT embeddings.

---