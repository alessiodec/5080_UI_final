import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from pymoo.optimize import minimize as minimizepymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
import streamlit as st

# -------------------------------------------------------------------
# Load dataset and prepare data
# -------------------------------------------------------------------
csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
with st.spinner("Loading dataset..."):
    Data_ph5 = pd.read_csv(csv_url)

# Keep only the data of interest
XData = Data_ph5[["pH", "T", "PCO2", "v", "d"]]
YData = Data_ph5[["CR", "SR"]]

# Apply log transformation to PCO2, v, and d
XData["PCO2"] = np.log10(XData["PCO2"])
XData["v"] = np.log10(XData["v"])
XData["d"] = np.log10(XData["d"])
XData = XData.dropna()
YData = YData.dropna()

# -------------------------------------------------------------------
# Load pre-trained models
# -------------------------------------------------------------------
with st.spinner("Loading pre-trained models..."):
    CorrosionModel = load_model("models/CorrosionRateModel.keras")
    SaturationModel = load_model("models/SaturationRateModel.keras")

# -------------------------------------------------------------------
# Create and fit a scaler on the input data (all 5 features)
# -------------------------------------------------------------------
with st.spinner("Fitting scaler on input data..."):
    scaler = preprocessing.StandardScaler()
    XDataScaled = scaler.fit_transform(XData).astype("float32")

# -------------------------------------------------------------------
# Function to reverse scaling and log10 transformation.
# Expects a 1D array of length 5.
# -------------------------------------------------------------------
def ReverseScalingandLog10(optimisationResult):
    result_reshaped = optimisationResult.reshape(1, -1)
    real_values = scaler.inverse_transform(result_reshaped)
    # Reverse the log transformation for columns 2,3,4 (PCO2, v, d)
    real_values[:, 2:] = 10 ** real_values[:, 2:]
    return real_values

# -------------------------------------------------------------------
# Define the optimization problem using pymoo.
# The full design vector is: [pH, T, PCO2, v, d]
# We fix PCO2 and d (after applying log10 and scaling the user inputs)
# and optimize over pH, T, and v.
# -------------------------------------------------------------------
class MinimizeCR(ElementwiseProblem):
    def __init__(self, d, PCO2):
        """
        d: user-defined pipe diameter (original, real-world value)
        PCO2: user-defined CO₂ partial pressure (original, real-world value)
        """
        # Apply log transformation and scaling to the fixed values.
        d_log = np.log10(d)
        d_scaled = scaler.transform(np.array([0, 0, 0, 0, d_log]).reshape(1, -1))[0][4]

        PCO2_log = np.log10(PCO2)
        PCO2_scaled = scaler.transform(np.array([0, 0, PCO2_log, 0, 0]).reshape(1, -1))[0][2]

        self.fixed_d = d_scaled
        self.fixed_PCO2 = PCO2_scaled

        # Design variables: pH (index 0), T (index 1), and v (index 3)
        xl = np.array([XDataScaled[:, 0].min(), XDataScaled[:, 1].min(), XDataScaled[:, 3].min()])
        xu = np.array([XDataScaled[:, 0].max(), XDataScaled[:, 1].max(), XDataScaled[:, 3].max()])
        super().__init__(n_var=3, n_obj=1, n_ieq_constr=1, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        # Reconstruct full design vector: [pH, T, fixed_PCO2, v, fixed_d]
        full_design = np.zeros(5)
        full_design[0] = X[0]       # pH
        full_design[1] = X[1]       # T
        full_design[2] = self.fixed_PCO2  # fixed, scaled PCO2
        full_design[3] = X[2]       # v
        full_design[4] = self.fixed_d     # fixed, scaled d
        full_design = full_design.reshape(1, -1)

        corrosionResult = CorrosionModel.predict(full_design, verbose=False).flatten()
        saturationResult = SaturationModel.predict(full_design, verbose=False).flatten()

        out["F"] = corrosionResult
        out["G"] = -10 ** saturationResult + 1

# -------------------------------------------------------------------
# Define the minimise_cr function.
# User inputs: d and PCO2 in original units.
# -------------------------------------------------------------------
def minimise_cr(d, PCO2):
    """
    Minimises the corrosion rate (CR) for a given pipe diameter (d) and CO₂ partial pressure (PCO2).

    Args:
        d (float): Pipe diameter (original, real-world value).
        PCO2 (float): CO₂ partial pressure (original, real-world value).

    Returns:
        best_params (np.array): Full design vector (unscaled, real-world values).
        min_cr (float): Minimum corrosion rate.
    """
    # Transform inputs using log10.
    with st.spinner("Transforming inputs using log10..."):
        d_log = np.log10(d)
        PCO2_log = np.log10(PCO2)

    # Scale the fixed values.
    with st.spinner("Scaling fixed values..."):
        darray = scaler.transform(np.array([0, 0, 0, 0, d_log]).reshape(1, -1))
        d_scaled = darray[0][4]
        PCO2array = scaler.transform(np.array([0, 0, PCO2_log, 0, 0]).reshape(1, -1))
        PCO2_scaled = PCO2array[0][2]

    # Set up the optimization problem.
    with st.spinner("Setting up the optimization problem..."):
        problem = MinimizeCR(d, PCO2)

    # Run the optimization algorithm (Differential Evolution).
    with st.spinner("Running optimization algorithm (DE)..."):
        algorithmDE = DE(pop_size=30, sampling=LHS(), dither="vector")
        result = minimizepymoo(problem, algorithmDE, verbose=True, termination=("n_eval", 300))

    # Process optimization results.
    with st.spinner("Processing optimization results..."):
        optimized_vars = np.atleast_1d(result.X).flatten()
        if optimized_vars.size == 1:
            optimized_vars = np.array(result.X[0]).flatten()

        if optimized_vars.size != 3:
            raise ValueError(f"Expected optimized_vars to have 3 elements, got {optimized_vars.size}")

        full_design_scaled = np.zeros(5)
        full_design_scaled[0] = optimized_vars[0]   # pH
        full_design_scaled[1] = optimized_vars[1]   # T
        full_design_scaled[2] = PCO2_scaled         # fixed, scaled PCO2
        full_design_scaled[3] = optimized_vars[2]   # v
        full_design_scaled[4] = d_scaled            # fixed, scaled d

        best_params = ReverseScalingandLog10(full_design_scaled)
        min_cr = result.F[0]

    # Compute final predictions using the final design vector.
    with st.spinner("Computing final model predictions..."):
        # Scale best_params to obtain predictions from the models.
        scaled_final = scaler.transform(best_params)
        final_cr = CorrosionModel.predict(scaled_final, verbose=False).flatten()[0]
        final_sr = SaturationModel.predict(scaled_final, verbose=False).flatten()[0]

    # Concatenate the input design vector and the predicted outputs.
    # The final vector is: [pH, T, CO₂, v, d, CR, SR]
    final_vector = np.concatenate((best_params.flatten(), np.array([final_cr, final_sr])))

    st.write("Optimisation Summary:")
    st.write("Final Vector (Input variables and predicted outputs):", final_vector)

    return best_params, min_cr
