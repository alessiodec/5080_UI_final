import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from pymoo.optimize import minimize as minimizepymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
import streamlit as st
import matplotlib.pyplot as plt
from joblib import load
from pymoo.decomposition.weighted_sum import WeightedSum
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------------
# Load dataset and prepare data
# -------------------------------------------------------------------
csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
with st.spinner("Loading dataset..."):
    Data_ph5 = pd.read_csv(csv_url)

XData = Data_ph5[["pH", "T", "PCO2", "v", "d"]]
YData = Data_ph5[["CR", "SR"]]

# Use .loc to avoid SettingWithCopy warnings.
XData.loc[:, "PCO2"] = np.log10(XData["PCO2"])
XData.loc[:, "v"] = np.log10(XData["v"])
XData.loc[:, "d"] = np.log10(XData["d"])
XData = XData.dropna()
YData = YData.dropna()

# -------------------------------------------------------------------
# Load pre-trained models from the 'models' directory
# -------------------------------------------------------------------
with st.spinner("Loading pre-trained models..."):
    CorrosionModel = load_model("models/CorrosionRateModel.keras", compile=False)
    SaturationModel = load_model("models/SaturationRateModel.keras", compile=False)

# -------------------------------------------------------------------
# Create and fit a scaler on the input data (all 5 features)
# -------------------------------------------------------------------
with st.spinner("Fitting scaler on input data..."):
    scaler = preprocessing.StandardScaler()
    XDataScaled = scaler.fit_transform(XData).astype("float32")

# -------------------------------------------------------------------
# Load multiobjective optimisation results from the 'models' directory.
# Make sure these joblib files are present in your 'models' folder.
# -------------------------------------------------------------------
with st.spinner("Loading multiobjective optimisation results..."):
    advancedRobustProblemResultNSGAF = load("models/advancedRobustProblemResultNSGAF.joblib")
    advancedRobustProblemResultNSGAX = load("models/advancedRobustProblemResultNSGAX.joblib")

# -------------------------------------------------------------------
# Normalize the Pareto objectives.
# -------------------------------------------------------------------
minMaxScaler = MinMaxScaler()
normalisedadvancedRobustProblemparetoObjectivesNSGA = np.column_stack((
    minMaxScaler.fit_transform(advancedRobustProblemResultNSGAF[:, 0].reshape(-1, 1)),
    minMaxScaler.fit_transform(advancedRobustProblemResultNSGAF[:, 1].reshape(-1, 1))
))

# -------------------------------------------------------------------
# Function to reverse scaling and log10 transformation.
# This function now properly handles both 1D and 2D inputs.
# -------------------------------------------------------------------
def ReverseScalingandLog10(optimisationResult):
    # If the result is 1D and its length equals 5, reshape to (1,5)
    if optimisationResult.ndim == 1 and optimisationResult.shape[0] == 5:
        result_reshaped = optimisationResult.reshape(1, -1)
    else:
        result_reshaped = optimisationResult  # assume it is already 2D with 5 columns
    real_values = scaler.inverse_transform(result_reshaped)
    # Reverse the log transformation for columns 2, 3, and 4 (PCO2, v, d)
    real_values[:, 2:] = 10 ** real_values[:, 2:]
    return real_values

# -------------------------------------------------------------------
# Define the optimization problem using pymoo.
# The full design vector is: [pH, T, PCO2, v, d]
# We fix PCO2 and d (scaled) and optimize over pH, T, and v.
# -------------------------------------------------------------------
class MinimizeCR(ElementwiseProblem):
    def __init__(self, d, PCO2):
        d_log = np.log10(d)
        d_scaled = scaler.transform(np.array([0, 0, 0, 0, d_log]).reshape(1, -1))[0][4]

        PCO2_log = np.log10(PCO2)
        PCO2_scaled = scaler.transform(np.array([0, 0, PCO2_log, 0, 0]).reshape(1, -1))[0][2]

        self.fixed_d = d_scaled
        self.fixed_PCO2 = PCO2_scaled

        # Bounds for design variables: pH (index 0), T (index 1), and v (index 3)
        xl = np.array([XDataScaled[:, 0].min(), XDataScaled[:, 1].min(), XDataScaled[:, 3].min()])
        xu = np.array([XDataScaled[:, 0].max(), XDataScaled[:, 1].max(), XDataScaled[:, 3].max()])

        super().__init__(n_var=3, n_obj=1, n_ieq_constr=1, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        full_design = np.zeros(5)
        full_design[0] = X[0]            # pH
        full_design[1] = X[1]            # T
        full_design[2] = self.fixed_PCO2 # fixed, scaled PCO2
        full_design[3] = X[2]            # v
        full_design[4] = self.fixed_d    # fixed, scaled d
        full_design = full_design.reshape(1, -1)

        corrosionResult = CorrosionModel.predict(full_design, verbose=False).flatten()
        saturationResult = SaturationModel.predict(full_design, verbose=False).flatten()

        out["F"] = corrosionResult
        out["G"] = -10 ** saturationResult + 1

# -------------------------------------------------------------------
# Define the minimise_cr function.
# User inputs: d and PCO2 in original units.
# Displays one final table with all values.
# -------------------------------------------------------------------
def minimise_cr(d, PCO2):
    st.write("Please Allow up to one Minute for the Optimisation Process to Complete")

    with st.spinner("Transforming inputs using log10..."):
        d_log = np.log10(d)
        PCO2_log = np.log10(PCO2)

    with st.spinner("Scaling fixed values..."):
        darray = scaler.transform(np.array([0, 0, 0, 0, d_log]).reshape(1, -1))
        d_scaled = darray[0][4]
        PCO2array = scaler.transform(np.array([0, 0, PCO2_log, 0, 0]).reshape(1, -1))
        PCO2_scaled = PCO2array[0][2]

    with st.spinner("Setting up the optimization problem..."):
        problem = MinimizeCR(d, PCO2)

    with st.spinner("Running optimization algorithm (DE)..."):
        algorithmDE = DE(pop_size=30, sampling=LHS(), dither="vector")
        result = minimizepymoo(problem, algorithmDE, verbose=True, termination=("n_eval", 300))

    with st.spinner("Processing optimization results..."):
        optimized_vars = np.atleast_1d(result.X).flatten()
        if optimized_vars.size == 1:
            optimized_vars = np.array(result.X[0]).flatten()
        if optimized_vars.size != 3:
            raise ValueError(f"Expected 3 elements, got {optimized_vars.size}")
        full_design_scaled = np.zeros(5)
        full_design_scaled[0] = optimized_vars[0]   # pH
        full_design_scaled[1] = optimized_vars[1]   # T
        full_design_scaled[2] = PCO2_scaled         # fixed, scaled PCO2
        full_design_scaled[3] = optimized_vars[2]   # v
        full_design_scaled[4] = d_scaled            # fixed, scaled d
        best_params = ReverseScalingandLog10(full_design_scaled)

    with st.spinner("Computing final model predictions..."):
        best_params_log = best_params.copy()
        best_params_log[:, 2:] = np.log10(best_params_log[:, 2:])
        scaled_final = scaler.transform(best_params_log)
        final_cr = CorrosionModel.predict(scaled_final, verbose=False).flatten()[0]
        final_sr = SaturationModel.predict(scaled_final, verbose=False).flatten()[0]

    final_vector = np.concatenate((best_params.flatten(), [final_cr, final_sr]))
    st.write("Final Design Vector (Input variables and predicted outputs):")
    column_headers = [
        "pH (-)",
        "T (°C)",
        "PCO₂ (Pa) *fixed*",
        "v (m/s)",
        "d (m) *fixed*",
        "CR (mm/year)",
        "SR (-)"
    ]
    final_df = pd.DataFrame([final_vector], columns=column_headers)
    st.table(final_df)
    return best_params, final_cr

# -------------------------------------------------------------------
# Updated findWeightedSolution function to flatten weights to a 1D array.
# -------------------------------------------------------------------
def findWeightedSolution(weights):
    weights = np.array(weights).flatten()
    decomp = WeightedSum()
    robustI = decomp(normalisedadvancedRobustProblemparetoObjectivesNSGA, weights).argmin()
    return robustI

# -------------------------------------------------------------------
# New function to plot Pareto front with user-defined weights.
# -------------------------------------------------------------------
def plot_pareto_front_traverse(weight_sensitivity, weight_cr):
    weights = np.array([weight_sensitivity, weight_cr])
    robustI = findWeightedSolution(weights)
    
    plt.rcParams['font.size'] = 20
    fig = plt.figure(figsize=(20, 10), dpi=300)
    plt.scatter(advancedRobustProblemResultNSGAF[:, 0], advancedRobustProblemResultNSGAF[:, 1],
                facecolors='none', edgecolors="r", label="Pareto Front", marker="o")
    plt.scatter(0.0008762, 0.04218847, label="Utopia Point", color="limegreen", marker="o", s=250)
    plt.scatter(advancedRobustProblemResultNSGAF[robustI, 0], advancedRobustProblemResultNSGAF[robustI, 1],
                label="Optimum Design", color="limegreen", marker="x")
    plt.title("CR vs Sensitivity")
    plt.ylabel("CR")
    plt.xlabel(r"$\|\nabla CR\|$")
    plt.legend()
    plt.grid()
    return fig

# -------------------------------------------------------------------
# Display default robust solution information using st.write.
# -------------------------------------------------------------------
robustWeights = np.array([1, 1])
robustI = findWeightedSolution(robustWeights)
np.set_printoptions(suppress=True)
st.write(f"""
The Obj. Fun. Values (Sens, CR) are: {np.around(advancedRobustProblemResultNSGAF[robustI], 7)} 
and the Design Variables are: {np.around(ReverseScalingandLog10(advancedRobustProblemResultNSGAX)[robustI], 7)}
CR = {CorrosionModel.predict(advancedRobustProblemResultNSGAX[robustI].reshape(1, -1), verbose=False)[0]}
SR = {10**SaturationModel.predict(advancedRobustProblemResultNSGAX[robustI].reshape(1, -1), verbose=False)[0]}
""")
