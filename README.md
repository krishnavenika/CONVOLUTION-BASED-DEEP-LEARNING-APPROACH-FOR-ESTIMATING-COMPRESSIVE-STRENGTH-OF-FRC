# CONVOLUTION-BASED-DEEP-LEARNING-APPROACH-FOR-ESTIMATING-COMPRESSIVE-STRENGTH-OF-FRC

1. **Understanding My Project Goal**:
   - I worked on a project aiming to predict the compressive strength of fiber reinforced concrete at elevated temperatures. This prediction is crucial for understanding how concrete behaves under different conditions, which is valuable in construction and engineering.

2. **Setting Up My Project**:
   - I've organized my project into different parts, each serving a specific purpose:
     - **Main Script (main.py)**:
       - This acts as the conductor of an orchestra, managing and coordinating all the important tasks in my project.
       - It oversees loading data, preprocessing it, creating the models, training them, and evaluating their performance.
     - **Model Files (cnn.py, lstm.py, svm.py)**:
       - These files contain the blueprints for my models, acting as recipe books for my cooking.
       - Each file has instructions on how to build a specific type of model: CNN, LSTM, or SVM.

3. **Training and Evaluating My Models**:
   - Here's how I train and evaluate my models:
     - **Loading Data**:
       - I start by bringing in my data, which includes information about fiber reinforced concrete and how it behaves under different conditions.
       - I use pandas to load this data from a file (data.csv) into my program.
     - **Data Preprocessing**:
       - Before I feed my data into the models, I need to prepare it, including scaling the data to ensure all features are in similar ranges.
       - I use MinMaxScaler for this job.
     - **Splitting Data**:
       - Next, I divide my data into two parts: one for training my models and the other for testing them to check their performance on unseen data.
     - **Building Models**:
       - I create instances of my models: CNN, LSTM, and SVM, each with its own set of instructions on how to process the data and make predictions.
     - **Training Models**:
       - This is where the magic happens! I give my models the training data and ask them to learn from it, adjusting their parameters to improve predictions over multiple epochs.
     - **Evaluating Models**:
       - Once trained, I test my models on the testing data to assess their performance, looking at a metric called loss to understand prediction accuracy.
     - **Visualizing Results**:
       - Finally, I visualize the performance of my models over time using graphs, comparing the loss across different models (CNN, LSTM, SVM).

4. **Understanding Model Performance**:
   - I observe that the CNN model's loss decreases steadily over epochs, indicating it's learning well.
   - The SVM model consistently decreases in loss and performs better than the other models.
   - The LSTM model shows fluctuations in loss, suggesting it might need further adjustments to improve stability.

5. **Next Steps**:
   - Based on these observations, I might want to further fine-tune my models, explore different architectures, or adjust training parameters to improve performance.
  
In a nutshell, my project revolves around building, training, and evaluating deep learning models to predict the compressive strength of fiber reinforced concrete at elevated temperatures. Each component plays a crucial role in understanding and improving the accuracy of my predictions.
