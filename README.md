# Mamo-ForensicTransfer
Auto-encoder-based forgery detection tool for mammogram images.

Project resentation - https://www.youtube.com/watch?v=bOTE8LqXQCg&t=27s (Hebrew)

##
Due to computing processes in the medical system, hospitals and clinics around the world have become a target to many cyber-attacks. In 2019 a team of researchers from Ben-Gurion University and Soroka medical center demonstrated how an attacker can inject or remove lesions from chest computerized tomography (CT) scans potentially causing misdiagnosis.

This kind of attacks has destructive potential ranging from insurance scams and disruption of studies all the way to political assault and even committing murder. The risks are high and finding solution is necessary.

In this project I plan to develop a deep learning based system to detect malicious tempering in mammogram imagery. I hope that this system could be a proof of concept to a defensive solution against this kind of cyber-attacks, and in the future to be widely embedded in hospital’s Picture Archiving and Communication Systems (PACS).
##
This tool will consist of two neuronal networks:
1. Forensic transfer- an autoencoder NN that receives tiles from a mammogram image determines for each tile the degree of probability that it is forged
2. A Convolutional neural network that receives for each image the tile scores from the previous grid and returns whether the image is real or fake.
