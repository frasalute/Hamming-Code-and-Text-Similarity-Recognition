# Hamming-Code-and-Text-Similarity-Recognition

# 3 Questions to solve with the use of Python

# Question 2: Hamming Code 
The focus of the second question is the Hamming Code and its use in the process of coding and de- coding. Coding theory is often concerned with the reliability of communication over noisy channels where errors can occur and therefore error correcting codes are used in a wide range of commu- nication settings (Nuh, 2007). The Hamming codes are a family of linear error-correcting codes that can detect one-bit errors and two-bit errors. They can also be used to correct one-bit errors without detection of uncorrected errors because the program is not able to identify more than one error position, i.e., 2-bit errors, and is not able to give information about other bits in error. The most common use of the Hamming Code is for the (7,4) algorithm, which involves encoding four bits of data into seven bits by adding three parity bits (Epp, 2011).

# Question 3: Text Similarity Recognition 
Question 3 requires the development of a Python program to compute the similarity between a given set of documents. A class named DocumentSimilarity has been created for loading and comparing documents (.txt files). When presented with a text name, the class outputs a data frame containing similarity scores to the rest of the documents already present in the Corpus.
To calculate similarity, four distinct methods have been implemented: dot product, distance norm, and two variations of cosine similarity. Users are given the flexibility to select a specific method or compare all methods simultaneously. Regardless of the chosen method, the common approach involves comparing the words across documents by transforming them into vectors.
