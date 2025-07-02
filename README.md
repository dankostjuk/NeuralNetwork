## Neural Network from Scratch  

This is a minimal neural network implementation written in pure C++ using the **Eigen** linear algebra library — no external machine learning frameworks.

### Features
Customizable number of hidden layers and activation functions
Implemented **Adam optimizer** for improved training convergence
Works well on small datasets (e.g. radiusClassification)


The next goal is to **optimize the network using parallel computation** to improve training speed on larger datasets.

How to Use

1. Install the Eigen library
2. Edit the `CMakeLists.txt` file to set the correct path to Eigen
3. Build and run the project as usual (with cmake)

