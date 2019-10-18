# Introduction

Source code to reproduce the results in
 J. Yang, D. Ruan, J. Huang, X. Kang and Y. Shi, "An Embedding Cost Learning Framework Using GAN," in IEEE Transactions on Information Forensics and Security, vol. 15, pp. 839-851, 2020.

There are some files:

1. model/120000.ckpt  The trained model after 120,000 iterations (72 epochs) with target payload 0.4 bpp.
2. Main.py  The main framework used to train the model.
3. gen_prob.py  To generate the embedding probability map, then convert the embedding probability to embedding cost for practical steganographic coding scheme. 
4. gen_simu_stego.py  Using this file to generate the simulated stego image with target payload.

# Citation
Please cite the following paper if the code helps your research.

 J. Yang, D. Ruan, J. Huang, X. Kang and Y. Shi, "An Embedding Cost Learning Framework Using GAN," in IEEE Transactions on Information Forensics and Security, vol. 15, pp. 839-851, 2020.