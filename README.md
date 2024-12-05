# ohpc sunspot model
1. Implement Simulated Annealing (SA) algorithm in python -  IMPORTANT: Project must be carried out using own SA function NOT a python lib
-  Optimise SA hyper params. Run SA for different values of the algorithm hyper paramsm, namely T0 and σ, and identify their best values.
-  Calibrate (=optimise) model params using your SA algorithm and the assigned SN dataset.

 ## Model Function
The shape of each solar cycle is described by a unique functional form:

![Model Function Formula](https://latex.codecogs.com/svg.latex?x_k(t)%20=%20\left(\frac{t-T_{0k}}{T_{Sk}}\right)^2%20e^{-\left(\frac{t-T_{0k}}{T_{dk}}\right)^2})

Where:

- **k** refers to the cycle number
- Parameters **T_{Sk}** and **T_{dk}** define its amplitude and rising time
- **T_{0k}** is the initial time (according to Hathaway, 2015)

### For 10 cycles:
1. ![x_1(t)](https://latex.codecogs.com/svg.latex?x_1(t)%20=%20\left(\frac{t-T_{01}}{T_{S1}}\right)^2%20e^{-\left(\frac{t-T_{01}}{T_{d1}}\right)^2}), with ![T_0 \leq t < T_2](https://latex.codecogs.com/svg.latex?T_0%20\leq%20t%20<%20T_2)
2. ![x_2(t)](https://latex.codecogs.com/svg.latex?x_2(t)%20=%20\left(\frac{t-T_{02}}{T_{S2}}\right)^2%20e^{-\left(\frac{t-T_{02}}{T_{d2}}\right)^2}), with ![T_2 \leq t < T_3](https://latex.codecogs.com/svg.latex?T_2%20\leq%20t%20<%20T_3)
3. ![x_3(t)](https://latex.codecogs.com/svg.latex?x_3(t)%20=%20\left(\frac{t-T_{03}}{T_{S3}}\right)^2%20e^{-\left(\frac{t-T_{03}}{T_{d3}}\right)^2}), with ![T_3 \leq t < T_4](https://latex.codecogs.com/svg.latex?T_3%20\leq%20t%20<%20T_4)
4. ...
10. ![x_{10}(t)](https://latex.codecogs.com/svg.latex?x_{10}(t)%20=%20\left(\frac{t-T_{010}}{T_{S10}}\right)^2%20e^{-\left(\frac{t-T_{010}}{T_{d10}}\right)^2}), with ![t \geq T_{010}](https://latex.codecogs.com/svg.latex?t%20\geq%20T_{010})

--> 30 free parameters to be optimised - T01, Ts1, Td1, T02, Ts2, Td2 ...

## Loss Function
The loss function to be minimized is **Mean Squared Error (MSE):**

![Loss Function Formula](https://latex.codecogs.com/svg.latex?\frac{1}{N}\sum_{i=1}^{N}e_i^2%20=%20\frac{1}{N}\sum_{i=1}^{N}(y_i-x(t_i))^2)

Where:
- **N** is the number of points in your dataset (approximately 40,000),
- **y_i** is a data point at time **t_i**, and
- **x(t_i)** is the model prediction.

## Hyper parameter optimisation
Optimise SA hyper parameters.
- Run SA for different values of the algorithm hyper params (**T0** and **σ**) and identify their best values
- This task must be parallelised on the cluster
- This is a very high-dimensional problem, and it is crucial to choose good starting values for the problem variables. As initial conditions for the starting times of the cycles, **T0k** , use the times reported in Hathaway 2015. As initial conditions for the other parameters, **T_{Sk}** and **T_{dk}**, you can use 0.3 and 5, respectively.
- For each run (=each set of hyper params) plot the evolution of the loss function and look at its final value. Compare the different results.

## Model calibration (optimisation)
Calibrate (=optimise) model parameters using your SA algorithm and the assigned SN dataset
- Using the best hyper params T0 and σ from Hyper parameter optimisation, run multiple independent SA optimisations (e.g. 10) using slightly different initial conditions.
- As initial condition, you can use the result of the best run of Hyper parameter optimisation and add a little bit of noise on top of it.
- This task must be parallelised on the cluster.
- At the end, collect the results of the independent runs into one numpy array and combine them calculating for each variable its Center of Mass. This will be your best estimation for the 30 model parameters!

## Final Fit and Linear Regression
Study the linear correlation between **T_{Sk}** and **T_{dk}**
- Equation (4) in Penza et al. 2024: **T_{dk}** = s1 * **T_{Sk}** + s2
                                    with s1 = 0.02 +- 0.01 year
- Carry out your own linear fit:
<slope, intercept = np.polyfiot(ts, td, 1) # 1 means linear fit>
<td_fit = slope * ts + intercept>

- What value do you obtain for the slope s1? Is it in agreement with the literature?
N.B.: the value 0.02 reported in the astrophysics literature is in ‘years’. Your dataset has a daily resolution, therefore your slope is likely to be in units of ‘days’. Remember to convert your result in the correct units!

## Practical Details:
**Deliverables:**
1. Presentation slides (Moodle)
2. Presentation video (TBD)
3. Code (Moodle)

**Presentation:**
1. 15-20 minutes
2. Every team member speaks

Outline:
1. Introduction
2. Describe the optimisation Problem
3. Parallel programming techniques
4. Results and performance metrics
5. Conclusions
6. Lessons learned and personal experience
7. Wrap-up

