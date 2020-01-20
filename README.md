# USE-OF-HARALICK-TEXTURE-FEATURES-FOR-IMAGE-CLASSIFICATION
AUTOMATED FUZZY-C CLUSTERING BASED IMAGE SEGMENTATION AND USE OF HARALICK TEXTURE FEATURES FOR IMAGE CLASSIFICATION
-----------------------------

1.	USE CASE DESCRIPTION: 
Describe what you are doing and why it matters to patients using at least one citation.

Breast cancer is the most common type of cancer in women and one of the leading causes of premature deaths in patients (citation). Mammography is use of x-ray to detect and monitor any pathological changes in breast tissue that may be indicative of cancer.  Over 40  million mammograms are taken every year in the united states alone [1].

This massive amount of data has to be studied and examined by an expert and adept professional. This process is time consuming. For timely detection and diagnosis computer aided detection and diagnosis (CADe and CADx) systems are designed. These automatic assistants can help in discovering and pinpointing abnormalities and give a set of probable diagnoses [ 2] helping in timely intervention which may be life saving for the patient. 
We are trying to design a mammography lesion/mass detection system  that can help in computer aided detection For this we are using a reproducible and public repository of annotated mammogram[2]. 

We have chosen to implement Fuzzy C Means clustering [ 3 ] method for segmentation of full mammography images to create a mask of region of interest and compare with masks provided in the dataset. We think an auto segmentation technique would be useful in surfing through a database of images to pre-process and prepare the data for training ML models for classification and further analysis. We then extract Haralick texture features [ 4] using GLCM for extracting the image descriptors.  



2. Describe the relevant statistics of the data. How were the images taken? How were they labeled? What is the class balance and majority classifier accuracy? How will you divide the data into testing, training and validation sets?

In DSSM mamograms from following sources can be found:  
— Massachusetts General Hospital, 
— Wake Forest University School of Medicine, 
— Sacred Heart Hospital, and 
— Washington University of St Louis School of Medicine. 
These images were taken from several different scanners. They were then converted into raw-pixel data of 64 bit density. The optical density were then re-mapped to 16-bit gray scale and converted to TIFF. Noise reduction was performed by clipping the desity values between 0.05 to 3.0.

Each of these images and respective cases were annotated with region of interest. It also includes the information like geometric features of mass, mass shape, mass margin, calcification type, calcification distribution, and breast density. Some other semantic features like patient age, assessment and abnormality rating can also be found [2].


3. Describe your data pipeline (how is the data scrubbed, normalized, stored, and fed to the model for training?).

Over all steps taken:

•	Binarize Image
•	Crop Image
•	Find Roi
•	Perform Segmentation
•	Find Haralick Features On Roi And Use Them For Classification Of Malignant And Benign
•	Perform Feature Selection
•	Perform Model Selection
•	Perform Model Evaluation

For automatic Image Segmentation:
We downloaded the full mammography images from the CBIS archive and performed the following operations on it:
1.	Converted DICOM images to JPG format - code available in image_utility.py
2.	Preserved adequate quality of images in the JPG format to perform FCM 
3.	We identified the corresponding ROI and Mask images from the dataset to perform manual validation of our results

For GLCM based Haralick Feature extraction and Classification:
Due to slow output of FCM, we  were not able to utilize the ROI given by our previous algorithm. To test the next step we need cropped images. We used the cropped images given by  CBIS-DDSM to extract Haralick Texture features using GLCM. The training and test set selection was dependent on the data download that forced us to limit us ourselves to randomized  a small dataset. 

Mass Data ROI	Calcification Data ROI
MASS TRAINING 
BENIGN COUNT: 63 
MALIGNANT COUNT: 54 

MASS TEST 
BENIGN COUNT: 17 
MALIGNANT COUNT: 16 	CALCIFICATION TRAINING
BENIGN COUNT: 103 
MALIGNANT COUNT: 52 

CALCIFICATION TEST 
BENIGN COUNT: 38 
MALIGNANT COUNT: 7 


To compute texture in all direction:
Moving window of size 50 was applied on the images. Feature extraction was done patch-wise. 
We varied the angles [0,45,90,135,180]
We used distance [0,1,2]

We used given values for each feature. No normalization were performed. There were over 60000 unique features extracted. The large size feature space called in for dimensionality reduction. This was used using chi-sq statistic. 

Models were trained and validated using 90-10 hold out method on Linear SVM and Logistic Regression. We have used the default parameter values and have not performed any parameter tuning. We tested the model on an unseen test set. The results were given in Accuracy, Precision, Recall, AUC.

4. Explain how the model you chose works alongside the code for it. Add at least one
technical citation to give credit where credit is due.

AUTO SEGMENTATION USING FCM: [3]
FCM method is similar to a KNN clustering technique except for the one difference between the two which is - a given pixel in an image can belong to multiple clusters rather than just one cluster (as in a KNN). This ‘fuzziness’ is bounded by a parameter called membership value which provides a distance measure between the data point of interest and the centroid of clusters. Some nuances of the technique are as follows:
1.	The farther away a given data point is from a cluster centroid, the lower it’s membership value
2.	The number of clusters is user defined and often requires extensive trial error or expert inputs to ensure the right number of clusters are defined for a given set of images 
3.	There are various ways to determine the optimal set of parameters fed to an FCM such as : 
1.	Dunn’s index :
2.	DB Index : measure of cluster quality (smaller DB = better clusters)
3.	C-index

The code we used was an extension of the following project which seemed very relevant to our work: https://github.com/sunying1304/Breast-Cancer-Classification-Based-on-Full-Mammogram

The algorithm follows these steps:
1.	It uses the following inputs as initialization variables:
1.	Number of clusters : we have run the code with different values to determine the optimal number of clusters
2.	Max iterations : defines the number of times the algorithm iterates to compute new centers and updates the membership values
3.	Fix m (usually set to 2 based on prior research work)
4.	Define the smoothing kernel for noise removal : Gaussian or Uniform. We have run the code with both types of kernels to evaluate segmentation performance
5.	Randomly initialize cluster centers
2. For number of iterations defined, the following are performed:
1.	Computation of centroid of clusters
2.	Computation of intuitionistic U based on centroid distance from the data point
3.	Recalculation of weights based on membership value
4.	Computation of new neighborhood weights
3. Based on the maximum weights computed, the image is segmented 


Input selection and algorithm tuning:
1.	We had to run the algorithm over and over with multiple values of c,m,epsilon, kernel type and iterations to understand the significance and variation of results by varying each parameter. The following is our findings:
1.	Cluster selection should be done based on the number of regions we expect to see in the image. A higher value of c would not serve our purpose of segmenting ROI since we end up finding many more unnecessary regions of the breast which are not useful 
2.	In highly calcified images, the number of iterations should be more and the stopping factor should be lower to ensure we can extract the ROI effectively
3.	The DB score was analyzed across multiple cluster sections and it was found that more than 3 clusters degrades algorithm performance and produces poor segmentation results


GLCM and Haralick features for malignant and benign classification [4].

•	Moving window of size 50 was applied on the images. Feature extraction was done patch-wise. 
•	We varied the angles [0,45,90,135,180]
•	We used distance [0,1,2]

We used given values for each feature. No normalization were performed. There were over 60000 unique features extracted. The large size feature space called in for dimensionality reduction. This was used using chi-sq statistic. 

Models were trained and validated using 90-10 hold out method on Linear SVM and Logistic Regression. We have used the default parameter values and have not performed any parameter tuning. We tested the model on an unseen test set. The results were given in Accuracy, Precision, Recall, AUC.

  


Q. 5 There are many ways to do training. Take us through how you do it (e.g. “We used early stopping and stopped when validation loss increased twice in a row.”).

We used the Hold-Out method for training the classifier model. The hold out method split the training set into 90% training data and 10%test data. This test data can be considered validation set used for hyperparameter tuning. The model is then tested on unseen test set. This is sampled from the test-data set provided for each type of breast lesions (calcification and mass). 


Q6. Make a figure displaying your results.
--- several figures can seen in the notebook. 


The table below gives the results for model performance on the two classifiers before and after dimensionality reduction.
  

Data	Classifier	Accuracy on validation	On unseen test set
			Accyuracy	Precision	Recall	AUC before Dimensionality reduction	AUC after dimensionality reduction
MASS	Linear SVM	75%	52%	50%	Accuracy: 56%	0.83	0.86
CALC	Linear SVM	62.5%	53%	11%	29%	0.60	0.63
MASS	Logistic Regression	75%	49%	47%	56%	0.86	0.75
CALC	Logistic Regression	62.6%	55.6%	15.8%	42.9%	0.57	0.75

We found that all classifier models give at least 12% false negative where it classifies a malignant into benign. There is variation in number false positives, which is as high as 31%. The high false positive can be considered good for borderline detection cases and more investigation needs to be done into that. Even 12% False Negative is of concerns.


FCM segmentation:
 


7. Discuss pros and cons of your method and what you might have done differently now that you’ve tried or would try if you had more time.

•	Larger dataset
•	FCM ROI comparison with the given data
•	FCM parameter tuning
•	Use of Semantic features in classification
•	Model testing


References:
1. Kooi, T., Litjens, G., Van Ginneken, B., Gubern-Mérida, A., Sánchez, C.I., Mann, R., den Heeten, A. and Karssemeijer, N., 2017. Large scale deep learning for computer aided detection of mammographic lesions. Medical image analysis, 35, pp.303-312.

2. Lee, R.S., Gimenez, F., Hoogi, A., Miyake, K.K., Gorovoy, M. and Rubin, D.L., 2017. A curated mammography data set for use in computer-aided detection and diagnosis research. Scientific data, 4, p.170177.


3. Keller, B. M., Nathan, D. L., Wang, Y., Zheng, Y., Gee, J. C., Conant, E. F., & Kontos, D. (2012). Estimation of breast percent density in raw and processed full field digital mammography images via adaptive fuzzy c‐means clustering and support vector machine segmentation. Medical physics, 39(8), 4903-4917.


4. Biswas, R., Nath, A. and Roy, S., 2016, September. Mammogram classification using gray-level co-occurrence matrix for diagnosis of breast cancer. In 2016 International Conference on Micro-Electronics and Telecommunication Engineering (ICMETE) (pp. 161-166). IEEE.
