import statsmodels.api as sm
import pandas as pd
import numpy as np
import random
import math

from statsmodels.stats.multitest import fdrcorrection
from sklearn.preprocessing import LabelEncoder


def random_sample(df, N):
    """
    Perform random sampling on a pandas DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame from which random sampling will be performed.

    N : int
        The number of random rows to select from the DataFrame.

    Returns
    -------
    sampled_df : pandas DataFrame
        A new DataFrame containing N randomly selected rows from the input DataFrame.
    """

    # Select N random rows from the DataFrame
    random_indices = random.sample(range(len(df)), N)
    sampled_df = df.iloc[random_indices]

    return sampled_df


def split_sampled_df(sampled_df):
    """
    Split a DataFrame into two equal-sized groups after shuffling its rows.

    Parameters
    ----------
    sampled_df : pandas DataFrame
        The input DataFrame to be split.

    Returns
    -------
    group1 : pandas DataFrame
        The first half of the shuffled input DataFrame.

    group2 : pandas DataFrame
        The second half of the shuffled input DataFrame.
    """

    # Use the sample method with frac=1 to shuffle all rows of the df
    sampled_df_shuffled = sampled_df.sample(frac=1).reset_index(drop=True)

    # Calculate the number of rows to split it in half
    half_rows = len(sampled_df_shuffled) // 2

    # Split the DataFrame into two halves
    group1 = sampled_df_shuffled.iloc[:half_rows]
    group2 = sampled_df_shuffled.iloc[half_rows:]

    return group1, group2


def fisher_transform(df):
    """
    Apply Fisher's transformation to a given DataFrame.

    Parameters
    ----------
    df : pandas DataFrame or numpy array
        The input data to which Fisher's transformation will be applied.

    Returns
    -------
    df_transformed : numpy array
        The data transformed using Fisher's transformation.
    """
    df_transformed = np.arctanh(df)

    return df_transformed


def extract_data(sampled_df, group1, group2):
    """
    Extract site and connectome data from two groups.

    Parameters
    ----------
    sampled_df : pandas DataFrame
        The original DataFrame containing subject data.

    group1 : pandas DataFrame
        The first group of subjects after splitting and shuffling.

    group2 : pandas DataFrame
        The second group of subjects after splitting and shuffling.

    Returns
    -------
    group1_site : pandas Series
        Encoded "Site" values for group 1.

    group2_site : pandas Series
        Encoded "Site" values for group 2.

    group1_conn : pandas DataFrame
        Connectome data for group 1 after removing "Subject" and "Site" columns.

    group2_conn : pandas DataFrame
    """

    # Extract the "site" columns
    group1_site = group1["Site"]
    group2_site = group2["Site"]

    # Convert site values to numeric using label encoding
    le = LabelEncoder()
    le.fit(sampled_df["Site"])
    group1_site = le.transform(group1_site)
    group2_site = le.transform(group2_site)

    # Convert the transformed site values to Pandas Series (for later concatenation)
    group1_site = pd.Series(group1_site)
    group2_site = pd.Series(group2_site)

    # Extract connectome values (excluding "Subject" and "site")
    group1_conn = group1.drop(columns=["Subject", "Site"])
    group2_conn = group2.drop(columns=["Subject", "Site"])

    return group1_site, group2_site, group1_conn, group2_conn


def apply_modification(value, d, std_value):
    """
    Apply a modification to a given value (connection) using an effect size.

    Parameters
    ----------
    value : float or numeric
        The original value to which the modification will be applied.

    d : float or numeric
        The effect size.

    std_value : float or numeric
        The standard deviation of the combined data.

    Returns
    -------
    modified_value : float or numeric
        The result of applying the modification to the original value.
    """

    return value + d * std_value


def modify_group2(group1_conn, group2_conn, pi, d):
    """
    Modify a subset of connections in the second group's connectome data.

    Parameters
    ----------
    group1_conn : pandas DataFrame
        Connectome data of the first group of subjects.

    group2_conn : pandas DataFrame
        Connectome data of the second group of subjects.

    pi : float
        The proportion of connections in group2_conn to modify.

    d : float
        The effect size applied to the selected connections.

    Returns
    -------
    connections_to_modify : pandas DataFrame
        A subset of connections randomly selected from group2_conn.

    group2_modified : pandas DataFrame
        The connectome data of the second group after modification.
    """

    # Calculate the total number of connections in the DataFrame
    total_conn = group2_conn.shape[1]

    # Calculate the number of connections to modify based on pi%
    num_to_modify = int(total_conn * pi)

    # Randomly select the connections (columns) to modify
    connections_to_modify = group2_conn.sample(n=num_to_modify, axis=1)

    # Stack both groups vertically for std claculation
    combined_data = pd.concat([group1_conn, group2_conn], axis=0)

    # Modify the selected columns in group2
    group2_modified = group2_conn.copy()
    for col in connections_to_modify.columns:
        std = combined_data[col].std()
        group2_modified.loc[:, col] = group2_modified.loc[:, col] + d * std

    return connections_to_modify, group2_modified


def standardize_data(group1_conn, group2_modified):
    """
    Standardize (z-score) the connectivity data of both groups based on the values of group1 (control group).

    Parameters
    ----------
    group1_conn : pandas DataFrame
        Connectome data of the first group of subjects (control group).

    group2_modified : pandas DataFrame
        Modified connectome data of the second group.

    Returns
    -------
    group1_conn_stand : pandas DataFrame
        Standardized connectome data of the first group.

    group2_modified_stand : pandas DataFrame
        Connectome data of the second group standardized according to the control group.
    """

    # Standardize the control group's data
    mean_control = group1_conn.mean()
    std_control = group1_conn.std()
    group1_conn_stand = (group1_conn - mean_control) / std_control

    # Standardize the "case" group's data
    group2_modified_stand = (group2_modified - mean_control) / std_control

    return group1_conn_stand, group2_modified_stand


def run_cwas(group1_conn_stand, group2_modified_stand, group1_site, group2_site):
    """
    Run a Connectome-Wide Association Study (CWAS) to identify significant connections.

    Parameters
    ----------
    group1_conn_stand : pandas DataFrame
        Standradised connectome data of the first group of subjects.

    group2_modified_stand : pandas DataFrame
        Modified connectome data of the second group of subjects, standardised based on the variance of the control group.

    group1_site : pandas Series
        Site values for the first group.

    group2_site : pandas Series
        Site values for the second group.

    Returns
    -------
    pval_list : list of floats
        List of p-values representing the significance of each connection.
    """

    connection_count = group1_conn_stand.shape[1]
    pval_list = []

    for connection_i in range(connection_count):
        # Extract the connectivity data for this connection
        connectivity_i_group1 = group1_conn_stand.iloc[:, connection_i]
        connectivity_i_group2 = group2_modified_stand.iloc[:, connection_i]

        # Stack the connectivity data
        connectivity_data = pd.concat(
            [connectivity_i_group1, connectivity_i_group2], axis=0
        )
        connectivity_data = connectivity_data.astype(float)

        # Create a design matrix with the group (0 or 1) and site information
        design_matrix = pd.DataFrame(
            {
                "Group": np.concatenate(
                    ([0] * len(connectivity_i_group1), [1] * len(connectivity_i_group2))
                ),
                "Site": pd.concat([group1_site, group2_site], axis=0),
                "Constant": 1,
            }
        )
        # Reset index so they match
        connectivity_data.index = design_matrix.index

        # Perform linear regression
        model = sm.OLS(connectivity_data, design_matrix)
        results = model.fit()

        # Save the p values for each connection
        pval = results.pvalues["Group"]
        pval_list.append(pval)

    return pval_list


def run_simulation(conn_df, N, pi, d):
    """
    Simulate a Connectome-Wide Association Study (CWAS) experiment.

    Parameters
    ----------
    conn_df : pandas DataFrame
        The original connectome data for a group of subjects.

    N : int
        The number of subjects to randomly select and simulate the experiment.

    pi : float
        The proportion of connections in group2 to modify.

    d : float
        The effect size used to modify the selected connections.

    Returns
    -------
    group1_conn : pandas DataFrame
        Connectome data of the first group of subjects.

    connections_to_modify : pandas DataFrame
        A subset of connections randomly selected for modification.

    pval_list : list of floats
        List of p-values representing the significance of each connection.
    """

    # Step 1: Randomly select N subjects
    sampled_df = random_sample(conn_df, N)

    # Step 2: Randomly split N selected subjects into 2 groups
    group1, group2 = split_sampled_df(sampled_df)

    group1_site, group2_site, group1_conn, group2_conn = extract_data(
        sampled_df, group1, group2
    )

    # Step 3: Pick pi% of connections at random and modify for group 2
    connections_to_modify, group2_modified = modify_group2(
        group1_conn, group2_conn, pi, d
    )

    # Step 4: Run CWAS
    group1_conn_stand, group2_modified_stand = standardize_data(
        group1_conn, group2_modified
    )
    pval_list = run_cwas(
        group1_conn_stand, group2_modified_stand, group1_site, group2_site
    )

    return group1_conn, connections_to_modify, pval_list


def calculate_sens_spef(group1_conn, connections_to_modify, rejected, q):
    """
    Calculate sensitivity and specificity based on the results of the CWAS.

    Parameters
    ----------
    group1_conn : pandas DataFrame
        Connectome data of the first group of subjects.

    connections_to_modify : pandas DataFrame
        A subset of connections that were modified during the simulation.

    rejected : list of bool
        A list of boolean values indicating whether the null hypothesis was rejected
        for each connection during CWAS.

    q : float
        The significance level or threshold for rejecting the null hypothesis.

    Returns
    -------
    sensitivity : float
        Sensitivity, a measure of the ability to correctly identify modified connections.

    specificity : float
        Specificity, a measure of the ability to correctly identify unmodified connections.
    """
    connection_count = group1_conn.shape[1]

    # Get a list of the modified connections
    modified_conn_list = connections_to_modify.columns.tolist()
    modified_conn_list = [int(conn) for conn in modified_conn_list]

    # Calculate the number of modified connections (condition positive), and non-modified connections (condition negative)
    condition_positive = len(modified_conn_list)
    condition_negative = connection_count - condition_positive
    true_positive_count = 0
    true_negative_count = 0

    for connection in range(connection_count):
        # Connection has been modified, the null hypothesis should be rejected
        if connection in modified_conn_list and rejected[connection]:
            true_positive_count += 1

        # Connection has not been modified, the null hypothesis should not be rejected
        elif connection not in modified_conn_list and not (rejected[connection]):
            true_negative_count += 1

    # Calculate sensitivity and specificity
    sensitivity = true_positive_count / condition_positive
    specificity = true_negative_count / condition_negative

    return sensitivity, specificity


def run_multiple_simulation(conn_df, N, pi, d, q, num_sample):
    sensitivity_list = []
    specificity_list = []
    correct_rejected_count = 0
    for sample in range(num_sample):
        # Perform steps 1-4 of simulation
        (
            group1_conn,
            connections_to_modify,
            pval_list,
        ) = run_simulation(conn_df, N, pi, d)

        # Step 5: Apply FDR correction
        rejected, corrected_pval_list = fdrcorrection(pval_list, alpha=q)

        # Calculate sensitivity and specificity
        sensitivity, specificity = calculate_sens_spef(
            group1_conn, connections_to_modify, rejected, q
        )

        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

        # If null hypothesis rejected, plus 1
        if np.any(corrected_pval_list < q):
            correct_rejected_count += 1

    result_summary = (
        f"Estimated mean sensitivity to detect d={d}, with pi={pi}%, q={q} and N={N}: "
        f"{round(np.mean(sensitivity_list), 2)}, with mean specificity of {round(np.mean(specificity_list), 2)}."
    )

    return result_summary
