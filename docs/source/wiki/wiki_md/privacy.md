# Privacy evaluation
## 1. Synthetic data as Privacy Enhancing Technology
In an era where data fuels innovation, the challenge of balancing data utility with privacy has become increasingly critical. Synthetic data has emerged as a transformative privacy-enhancing technology (PET) that offers a powerful solution to this challenge. By simulating artificial yet realistic datasets, synthetic data provides a way to anonymize sensitive information while preserving the statistical properties of the original data[^1].

Unlike traditional anonymization techniques, such as masking or redaction, which often compromise data utility, synthetic data ensures that privacy and usability go hand in hand. Here's how synthetic data stands out as a privacy-enhancing technology:

1. Anonymization through Simulation:
    Synthetic data is generated without directly copying or exposing real-world data points. Instead, it captures the statistical patterns, distributions, and relationships within the dataset, effectively "removing" sensitive personal information.

2. Preserving Utility:
    While anonymization techniques like differential privacy can sometimes degrade the quality of data, synthetic data retains the richness and structure necessary for downstream tasks like machine learning, analytics, and testing. This makes it an ideal choice for scenarios where data utility is paramount.

3. Regulatory Compliance:
    With growing privacy regulations like GDPR, CCPA, and HIPAA, synthetic data provides a compliant way to share and process data without risking exposure of personally identifiable information (PII). Organizations can innovate and collaborate without breaching data privacy laws.

4. Broad Applicability:
    Synthetic data has wide-ranging applications across industries. For example:

    - Healthcare: Sharing patient data for research while protecting sensitive medical information.
    - Finance: Simulating transaction data for fraud detection algorithms.
    - AI Development: Training and testing machine learning models in privacy-sensitive domains.

By decoupling data utility from privacy risks, synthetic data unlocks new opportunities for innovation, collaboration, and analysis. It represents a key step toward building a data-driven future where privacy is respected, and insights are not constrained by access restrictions.

As synthetic data continues to evolve, its role as a cornerstone of privacy-enhancing technologies becomes more evident—paving the way for responsible, ethical, and scalable data usage.

## 2. Privacy risks with synthetic data
> Data can be either useful or perfectly anonymous but never both.[^2]<br>
> -Paul Ohm

It is not possible to have a dataset that is 100% privacy reserving if we also want to obtain actionable information from it, and synthetic data is no exception.

If the utilty of a synthetic dataset increases, its privacy decreases, and vice versa. This trade-off is commonly referred to as the **utility-privacy conundrum**.<br>
While synthetic data is a powerful privacy-enhancing technology, it is not immune to privacy risks. Its effectiveness depends on how the data is generated and the methods used to assess its privacy guarantees. Understanding the potential risks is critical for ensuring synthetic data truly protects sensitive information. 

Even if a synthetic dataset is considered an anonymized version of its original counterpart, meaning that it does not contain direct identifiers like names or ID numbers, the risk of re-identification of its records remains.<br>
Synthetic data aims to anonymize sensitive data, but poorly designed generation methods can unintentionally recreate patterns or records too similar to the original data. If these generated records are close enough to identifiable individuals or entities in the source dataset, it opens the door to potential re-identification attacks.<br>
If a synthetic dataset closely mirrors rare or unique combinations of attributes in the real dataset (e.g., a person’s age, occupation, and location) it may inadvertently reveal sensitive information.

If synthetic data generation models (e.g., GANs, VAEs) overfit to the original dataset, adversaries may be able to determine whether a specific individual's data was used to train the model. This is known as a **membership inference attack**, which undermines the privacy of those included in the original dataset. Read the section below dedicated to membership inference attack to learn more about the topic. 

To minimize these risks, organizations can adopt best practices to balance utility and privacy to maintain data usefulness while protecting sensitive information.
It is essential to regularly assess privacy vulnerabilities using state-of-the-art adversarial attack simulations, such as the one integrated in the privacy evaluation module of Clearbox Synthetic Kit.

By addressing these privacy risks proactively, synthetic data can remain a reliable and effective tool for responsible data sharing and innovation.

## 3. Distance to Closest Record with Clearbox Synthetic Kit
Distance to Closest Record (DCR) is a widely used privacy metric to evaluate the vulnerability of individual records in a synthetic dataset. This metric measures the similarity between each synthetic record and its nearest real record in the original dataset. By analyzing these distances, organizations can identify synthetic records that are overly similar to real-world records, which might pose a privacy risk due to potential re-identification attacks, such as membership inference attack.

These are the steps that make up the DCR metric computation:

1. The **distance matrix** is calculated between two dataframes that contain mixed data types, including both numerical and categorical variables. This computation leverages a modified version of Gower's distance, that allows to consider both numerical and categorical features.

2. Following this, the **DCR** (**Distance to Closest Record**) **vector** is then derived by determining the shortest distance between each record in the synthetic dataset and all records in the original dataset.<br>
For each synthetic record, the smallest distance among all pairwise distances to records in the original dataset is identified.<br> 
This step effectively quantifies how closely each synthetic record resembles the original data, with smaller values indicating greater similarity.

3. The **distribution of DCR** values is then analyzed to assess the level of anonymization and safety of the synthetic dataset. Specifically:

- *Overlap of DCR Distributions*:
    
    If the distribution of DCR values between the synthetic and training dataset closely overlaps with the distribution of DCR values between the synthetic and holdout dataset, it indicates that the synthetic data does not reveal whether a record from the original dataset was used to train the synthetic data generator. In this case, the synthetic dataset can be considered sufficiently anonymized and safe from reidentification attacks, as there is no evidence that the generator is overfitting to specific records in the training data.

- *Lack of Overlap in DCR Distributions*:

    Conversely, if the DCR distribution between the synthetic and training dataset differs significantly from that of the synthetic and holdout dataset, it raises concerns about the privacy and anonymization of the synthetic dataset.
    
    Specifically, if the DCR distribution for the synthetic and training dataset contains a greater proportion of smaller distances (indicating higher similarity) compared to the holdout dataset, this suggests that many synthetic records are too similar to the training records.<br>
    This lack of diversity increases the risk of reidentification, even if the synthetic dataset has undergone anonymization procedures.<br>
    Such a scenario implies that the synthetic data generator has overfit to the training dataset, compromising its ability to generate sufficiently diverse and independent synthetic records.

3. Finally, the **share** of synthetic rows closer to the training dataset than to the validation dataset is computed as:

{math}`DCR\_share = \frac{number\space of\space DCR\_synth\_train \space smaller\space than\space DCR\_synth\_holdout}{total \space number \space of \space DCR\_synth\_train \space rows}   * 100`

If the percentage is close to or below 50%, it provides empirical evidence that the training and validation data are interchangeable with respect to the synthetic data. This indicates that the synthetic data does not disproportionately resemble the training dataset.<br> 
In such a case, it would not be possible to infer whether a specific individual was or was not included in the training dataset based on the synthetic data alone.<br>
Conversely, if the DCR share exceeds 50%, it suggests that the synthetic dataset contains a significant number of records that are easily reidentifiable. This is a strong indicator that the synthetic data overfits the training data, potentially compromising its anonymization and exposing individuals to reidentification risks.

Low DCR values indicate synthetic records that closely resemble specific real records. These records are at higher risk of enabling membership inference, where attackers may deduce sensitive details about individuals in the original dataset.\
By flagging overly similar records, DCR helps in refining synthetic data generation methods to balance privacy (ensuring records aren't too close to real data) with utility (maintaining useful patterns for analysis).\
DCR is also useful to evaluate the generative model performance. Infact, synthetic data generation methods that produce low DCR values across the dataset might indicate overfitting to the real data. This suggests a failure to adequately generalize, which compromises privacy.

## 4. Membership Inference Attack simulation with Clearbox Synthetic Kit
The **Membership Inference Test**[^3] [^4]: is a critical evaluation used to estimate the risk of revealing membership information in a dataset.<br> 

If a malicious attacker manages to put his hands on an anonymized synthetic dataset and also has some prior knowledge about one or more of the records of the original dataset he may be able to infer whether the records he has information about were part of the original dataset from which the synthetic dataset was generated from, possibily disclosing sensitive information about the records.

> **Example**\
> A hospital publicly releases an anonymized synthetic dataset about cancer patients, making it accessible to researchers in the medical field.\
> An attacker, who happens to be the neighbor of a frequent hospital visitor, stumbles upon this dataset. The attacker knows several personal details about their neighbor, such as height, native country, city of residence, smoking and drinking habits, and other lifestyle factors.\
> By examining the synthetic dataset, the attacker identifies an anonymous record that closely matches their neighbor’s profile across these attributes. Based on these similarities, the attacker deduces with a high degree of confidence that their neighbor’s data was part of the original dataset used to generate the synthetic version. Consequently, the attacker infers that their neighbor has cancer.

This simple example highlights how sensitive information can be inadvertently disclosed, even in anonymized synthetic datasets, through linkage attacks, where auxiliary knowledge combined with dataset patterns enables attackers to identify individuals and uncover private details. This breach of privacy illustrates the importance of robust privacy-preserving techniques in synthetic data generation to prevent such vulnerabilities.

The membership inference test assesses how well an adversary, armed with certain prior knowledge, could infer whether specific records were part of the original training dataset used to generate the synthetic data, highlighting the presence of vulnerable records.

**Why Membership Inference Poses a Privacy Risk**<br>

A malicious attacker who has prior knowledge of one or more records in the original dataset can use the membership inference test to determine if those records were part of the training dataset used to generate the synthetic data.<br> 
This process can compromise the privacy of the original dataset in several ways:

- *Revealing Sensitive Information*:

    If the synthetic dataset retains too much similarity to the original data, the attacker can infer sensitive details about individuals or entities, such as health records, financial transactions, or other private attributes.
- *Increased Risk of Reidentification*:

    The more closely the synthetic data resembles the training data, the higher the risk that an attacker can reidentify individuals within the synthetic dataset, thus violating their privacy.
- *Erosion of Anonymization*:

    The goal of synthetic data is to anonymize original records while maintaining data utility. However, if membership inference is successful, it undermines the very purpose of synthetic data generation by exposing private information.

Synthetic data generation techniques must ensure that records in the synthetic dataset cannot be linked back to the original dataset with high confidence.

**How the Membership Inference Test Works**

1. *Distance to the Closest Record (DCR)*:

    The test calculates the **DCR** (**Distance to the Closest Record**) for each record in the adversary dataset by measuring its distance to all records in the synthetic dataset.<br>
    The DCR quantifies how closely the adversary’s known records resemble records in the synthetic dataset.
2. *Threshold-Based Risk Evaluation*:

    A set of distance thresholds is applied to the DCR values. These thresholds determine whether a record is considered sufficiently "close" to indicate membership in the original dataset.<br>
    For each threshold, the test computes precision scores, which reflect how accurately the synthetic dataset can be linked back to specific records in the adversary dataset.
3. *Privacy Risk Assessment*:

    If many records from the adversary dataset are identified with high precision (low DCR values), this indicates that the synthetic data strongly resembles the original data, increasing the risk of membership inference.<br>
    Conversely, if the precision scores are low across thresholds, it suggests that the synthetic dataset is well-anonymized, reducing the likelihood of successful membership inference.

    The **MI Mean Risk score** is computed as $(precision - 0.5) * 2$.<br>
    MI Risk Score smaller than 0.2 (20%) are considered to be very LOW RISK of disclosure due to membership inference.

## 5. Appendix
### Distance to Closest Record with Gower's distance
The Gower's distance is a mathematical distance metric that can be computed between two records of two datasets with the same features.\
The version used in Clearbox Synthetic Kit to determine the Distance to Closest Record takes into account both numerical and categorical distances.

Suppose we have two observations $x_i=(x_{i1},...,x_{ip})$ and $x_j=(x_{j1},...,x_{jp})$.

For each feature $k=1,...,p$ we define the Gower's distance $d_{ij} \in [0,1].

The distance $d_{ij}$ is defined depending on the type of feature $k$:

Quantitative/numerical feature:

$$d_{ij}=\frac{|x_{ik}-x_{jk}|}{R_k}$$


Where the range $R_k$ is defined as:

$$
R_k=max(x_{ik},x_{jk})-min(x_{ik},x_{jk})
$$

Qualitative/categorical feature:

$$
d_{ijk} =
\begin{cases} 
0, & \text{if } x_{ik} = x_{jk}, \\
1, & \text{if } x_{ik} \neq x_{jk}.
\end{cases}
$$

The Gower's distance is:

$$
D_{ij} = \frac{1}{p}\sum_{k=1}^{p} d_{ijk}
$$

Finally the Distance to Closest Record is computed as:

$$
DCR=min(D_{i1},...,D_{iN})
$$

Where N is the length of the original dataset.
<br>
<br>
<br>
<br>
**References**
[^1]: Multipurpose synthetic population for policy applications [[Link](https://publications.jrc.ec.europa.eu/repository/handle/JRC128595)]
[^2]: Broken Promises of Privacy: Responding to the Surprising Failure of Anonymization [[Link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1450006)]
[^3]: Membership Inference on data: How To Break Anonymity of the Netflix Prize Dataset [[Link](https://arxiv.org/abs/cs/0610105)]
[^4]: Who's Watching? De-anonymization of Netflix Reviews using Amazon Reviews [[Link](https://courses.csail.mit.edu/6.857/2018/project/Archie-Gershon-Katchoff-Zeng-Netflix.pdf)]
