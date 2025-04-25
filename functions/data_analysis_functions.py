import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA

# load models from models folder
def load_models():
    with st.spinner("Loading models..."):
        cr_model = load_model("models/CorrosionRateModel.keras")
        sr_model = load_model("models/SaturationRateModel.keras")
    return cr_model, sr_model

# summary statistics of dataset
def descriptive_analysis(X):
    with st.spinner("Computing descriptive statistics..."):
        st.write("Descriptive Statistics:")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if all(isinstance(col, int) for col in X.columns):
            col_names = ['pH', 'T (C)', 'log10 PCO2 (bar)', 'log10 v (ms-1)', 'log10 d']
            if X.shape[1] >= 5:
                X.columns = col_names + list(X.columns[5:])
            else:
                X.columns = col_names[:X.shape[1]]
        X_subset = X.iloc[:, :5]
        st.write(X_subset.describe())

# input histograms
def input_histogram():
    with st.spinner("Generating input histograms..."):
        csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
        df = pd.read_csv(csv_url)
        cols_to_keep = list(range(0, 5))
        df_subset = df.iloc[:, cols_to_keep].copy()
        df_subset.iloc[:, [2, 3, 4]] = np.log10(df_subset.iloc[:, [2, 3, 4]].replace(0, np.nan))
        plt.figure(figsize=(12, 8))
        df_subset.hist(bins=30)
        plt.suptitle("Histograms of Inputs", y=0.95)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

# read csv from google drive url, do necessary preprocessing
def load_preprocess_data():
    with st.spinner("Loading and preprocessing data..."):
        csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
        df = pd.read_csv(csv_url)
        cols_to_keep = list(range(0, 5)) + [7, 17]
        df_subset = df.iloc[:, cols_to_keep].copy()
        df_subset.iloc[:, [2, 3, 4]] = np.log10(df_subset.iloc[:, [2, 3, 4]])
        print(df_subset.head())  # For debugging
        X = df_subset.iloc[:, :5].values
        y = df_subset.iloc[:, 5:7].values
        scaler_X = StandardScaler()
        scaler_X.fit(X)
    return df_subset, X, scaler_X

# pca plot
def pca_plot():
    with st.spinner("Performing PCA and generating plots..."):
        st.markdown("""\
**Interpreting PCA:**

**Scree Plot:**  
A scree plot shows the variance explained by each principal component. Look for the "elbow"—the point where additional components offer diminishing returns—to decide how many components to retain.

**PCA Loadings Heatmap:**  
This heatmap displays how much each original feature contributes to each principal component; darker or more intense values indicate a stronger influence on that component.
        """)
        csv_url = "https://drive.google.com/uc?export=download&id=10GtBpEkWIp4J-miPzQrLIH6AWrMrLH-o"
        df = pd.read_csv(csv_url)
        pca_data = df.iloc[:, [0, 1, 2, 3, 4]]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data)
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        exp_var_ratio = pca.explained_variance_ratio_
        results = {
            'pca_result': pca_result,
            'loadings': pca.components_,
            'explained_variance_ratio': exp_var_ratio,
            'cumulative_variance_ratio': np.cumsum(exp_var_ratio)
        }
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].plot(range(1, len(exp_var_ratio) + 1), exp_var_ratio, 'bo-', label='Explained Variance')
        axes[0].plot(range(1, len(results['cumulative_variance_ratio']) + 1), results['cumulative_variance_ratio'], 'ro-', label='Cumulative Variance')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Variance Ratio')
        axes[0].set_title('Scree Plot')
        axes[0].legend()
        loadings = results['loadings']
        features = pca_data.columns
        loadings_df = pd.DataFrame(loadings.T,
                                   columns=[f'PC{i+1}' for i in range(loadings.shape[0])],
                                   index=features)
        sns.heatmap(loadings_df, cmap='RdBu', center=0, annot=True, fmt='.2f', ax=axes[1])
        axes[1].set_title('PCA Loadings Heatmap')
        st.pyplot(fig)
        explained_variance = {}
        for i, var in enumerate(exp_var_ratio):
            explained_variance[f"PC{i+1}"] = {
                'explained_variance_ratio': var,
                'cumulative_variance_ratio': results['cumulative_variance_ratio'][i]
            }
    # return explained_variance

# cr contour plot
def plot_5x5_cr(X, scaler_X, cr_model):
    with st.spinner("Generating corrosion rate contour plots..."):
        mid_points = np.median(X, axis=0)
        var_names = ['pH', 'T (C)', 'log10 PCO2 (bar)', 'log10 v (ms-1)', 'log10 d']
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        fig, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=False, sharey=False)
        for i in range(5):
            for j in range(5):
                ax = axes[i, j]
                if i == j:
                    ax.text(0.5, 0.5, var_names[i], fontsize=14, ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                x_vals = np.linspace(mins[j], maxs[j], 25)
                y_vals = np.linspace(mins[i], maxs[i], 25)
                grid_x, grid_y = np.meshgrid(x_vals, y_vals)
                grid_points = np.tile(mid_points, (grid_x.size, 1))
                grid_points[:, j] = grid_x.ravel()
                grid_points[:, i] = grid_y.ravel()
                grid_points_scaled = scaler_X.transform(grid_points)
                predictions_scaled = cr_model.predict(grid_points_scaled)
                # Handle potential 1D or 2D (N, 1) output from model predict
                if predictions_scaled.ndim == 1:
                    # Ensure reshape has correct dimensions
                    corrosion_rate = predictions_scaled.reshape(grid_x.shape)
                elif predictions_scaled.ndim == 2 and predictions_scaled.shape[1] >= 1:
                    # Assume first column is the target if multiple are returned (or N,1)
                    if predictions_scaled.shape[1] > 1:
                         st.warning(f"Model prediction shape is {predictions_scaled.shape}. Assuming first column is corrosion rate.")
                    # Ensure reshape has correct dimensions
                    corrosion_rate = predictions_scaled[:, 0].reshape(grid_x.shape)
                else:
                    # Handle unexpected shape, maybe raise error or log warning
                    st.error(f"Cannot interpret model prediction shape: {predictions_scaled.shape}")
                    return # Or handle error appropriately
                cont_fill = ax.contourf(grid_x, grid_y, corrosion_rate, levels=10, cmap='viridis')
                cont_line = ax.contour(grid_x, grid_y, corrosion_rate, levels=10, colors='black', linewidths=0.5)
                ax.clabel(cont_line, inline=True, fontsize=8, colors='white')
                ax.set_xlabel(var_names[j])
                ax.set_ylabel(var_names[i])
        fig.subplots_adjust(right=0.9, hspace=0.4, wspace=0.4)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        # Create colorbar once and store it in 'cbar'
        cbar = fig.colorbar(cont_fill, cax=cbar_ax)
        # Set the colorbar label with the specified fontsize
        cbar.set_label('Scaled Corrosion Rate', fontsize=40)
        # Set tick label size using the 'cbar' object
        cbar.ax.tick_params(labelsize=40)
        # Set the main title fontsize
        plt.suptitle('CR For Different Input Combinations', fontsize=40)
        st.pyplot(fig)
         
# sr contour plot
def plot_5x5_cr(X, scaler_X, cr_model):
     with st.spinner("Generating corrosion rate contour plots..."):
         mid_points = np.median(X, axis=0)
         var_names = ['pH', 'T (C)', 'log10 PCO2 (bar)', 'log10 v (ms-1)', 'log10 d']
         mins = X.min(axis=0)
         maxs = X.max(axis=0)
         fig, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=False, sharey=False)
         for i in range(5):
             for j in range(5):
                 ax = axes[i, j]
                 if i == j:
                     ax.text(0.5, 0.5, var_names[i], fontsize=14, ha='center', va='center')
                     ax.set_xticks([])
                     ax.set_yticks([])
                     continue
                 x_vals = np.linspace(mins[j], maxs[j], 25)
                 y_vals = np.linspace(mins[i], maxs[i], 25)
                 grid_x, grid_y = np.meshgrid(x_vals, y_vals)
                 grid_points = np.tile(mid_points, (grid_x.size, 1))
                 grid_points[:, j] = grid_x.ravel()
                 grid_points[:, i] = grid_y.ravel()
                 grid_points_scaled = scaler_X.transform(grid_points)
                 # Assuming cr_model.predict returns shape (n_samples, n_outputs)
                 # and CR is the first output
                 predictions_scaled = cr_model.predict(grid_points_scaled)
                 # Ensure predictions is 2D before indexing
                 if predictions_scaled.ndim == 1:
                      predictions_scaled = predictions_scaled.reshape(-1, 1 not already present globally

def plot_5x5_cr(X, scaler_X, cr_model):
    with st.spinner("Generating corrosion rate contour plots..."):
        mid_points = np.median(X, axis=0)
        var_names = ['pH', 'T (C)', 'log10 PCO2 (bar)', 'log10 v (ms-1)', 'log10 d']
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        fig, axes = plt.subplots(5, 5, figsize=(20, 20), sharex=False, sharey=False)
        for i in range(5):
            for j in range(5):
                ax = axes[i, j]
                if i == j:
                    ax.text(0.5, 0.5, var_names[i], fontsize=14, ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                x_vals = np.linspace(mins[j], maxs[j], 25)
                y_vals = np.linspace(mins[i], maxs[i], 25)
                grid_x, grid_y = np.meshgrid(x_vals, y_vals)
                grid_points = np.tile(mid_points, (grid_x.size, 1))
                grid_points[:, j] = grid_x.ravel()
                grid_points[:, i] = grid_y.ravel()
                grid_points_scaled = scaler_X.transform(grid_points)
                predictions_scaled = cr_model.predict(grid_points_scaled)
                # Handle potential 1D or 2D (N, 1) output from model predict
                if predictions_scaled.ndim == 1:
                    corrosion_rate = predictions_scaled.reshape(grid_x.shape)
                elif predictions_scaled.shape[1] >= 1:
                    # Assume first column is the target if multiple are returned (or N,1)
                    if predictions_scaled.shape[1] > 1:
                         st.warning(f"Model prediction shape is {predictions_scaled.shape}. Assuming first column is corrosion rate.")
                    corrosion_rate = predictions_scaled[:, 0].reshape(grid_x.shape)
                else:
                    # Handle unexpected shape, maybe raise error or log warning
                    st.error(f"Cannot interpret model prediction shape: {predictions_scaled.shape}")
                    return # Or handle error appropriately
                cont_fill = ax.contourf(grid_x, grid_y, corrosion_rate, levels=10, cmap='viridis')
                cont_line = ax.contour(grid_x, grid_y, corrosion_rate, levels=10, colors='black', linewidths=0.5)
                ax.clabel(cont_line, inline=True, fontsize=8, colors='white')
                ax.set_xlabel(var_names[j])
                ax.set_ylabel(var_names[i])
        fig.subplots_adjust(right=0.9, hspace=0.4, wspace=0.4)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        # Create colorbar once and store it in 'cbar'
        cbar = fig.colorbar(cont_fill, cax=cbar_ax, label='Scaled Corrosion Rate')
        # Set tick label size using the 'cbar' object
        cbar.ax.tick_params(labelsize=40)
        plt.suptitle('CR For Different Input Combinations', fontsize=40)
        st.pyplot(fig)
