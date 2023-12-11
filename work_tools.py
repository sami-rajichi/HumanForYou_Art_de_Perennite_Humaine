import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

def check_for_duplicates(data):
    """
    Check for duplicates in a pandas DataFrame.

    Parameters:
    - data (pd.DataFrame): The DataFrame to check for duplicates.

    Returns:
    - None

    Prints:
    - Rows that are duplicates (optional)
    - Total number of duplicates
    - Whether the DataFrame contains duplicates or not
    """
    # Check for duplicates
    duplicates = data.duplicated(keep=False)

    # Print the rows that are duplicates (optional)
    print(data[duplicates])

    # Count the total number of duplicates
    num_duplicates = duplicates.sum()
    print("Number of duplicates:", num_duplicates)

    # Check if there are any duplicates in the DataFrame
    if num_duplicates > 0:
        print("The DataFrame contains duplicates.")
    else:
        print("The DataFrame does not contain duplicates.")


def fill_missing_values(dataframe):
    """
    Fill missing values in a DataFrame.

    For numerical columns, missing values are replaced with the median.
    For categorical columns (if present), missing values are replaced with the mode.

    Args:
        dataframe: The input DataFrame.

    Returns:
        The DataFrame with missing values replaced.
    """
    # Fill missing values in numerical columns with the median
    numeric_columns = dataframe.select_dtypes(include=[np.number])
    dataframe[numeric_columns.columns] = dataframe[numeric_columns.columns].fillna(numeric_columns.median())

    # Fill missing values in categorical columns with the mode (if categorical columns exist)
    categorical_columns = dataframe.select_dtypes(include=['object'])
    if not categorical_columns.empty:
        dataframe[categorical_columns.columns] = dataframe[categorical_columns.columns].fillna(categorical_columns.mode().iloc[0])

    return dataframe


def visualize_outliers(dataframe):
    """
    Visualize outliers for all numerical variables in a DataFrame.

    Args:
        dataframe: The input DataFrame.

    Returns:
        None (displays subplots).
    """
    # Select numerical columns
    numeric_columns = dataframe.select_dtypes(include=[np.number])

    # Set up subplots
    num_cols = len(numeric_columns.columns)
    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(15, 5))

    # Plot boxplots for each numerical variable
    for i, col in enumerate(numeric_columns.columns):
        sns.boxplot(x=dataframe[col], ax=axes[i])
        axes[i].set_title(col)

    # Adjust layout
    plt.tight_layout()
    plt.show()


from scipy import stats

def z_scores(data,field):
    # Calculate z-scores for mont_enc
    z_scores = stats.zscore(data[field])

    # Set the threshold for outlier detection (e.g., 3 or -3)
    threshold = 3

    # Find the indices of potential outliers
    outlier_indices = [index for index, z_score in enumerate(z_scores) if abs(z_score) > threshold]

    # Get the values of mont_enc that are potential outliers
    potential_outliers = data[field].iloc[outlier_indices]

    # Print the potential outliers
    print(f"Potential Outliers in {field}: {len(potential_outliers)}")
    print(potential_outliers)



def visualize_frequency(dataframe, columns):
    """
    Visualize the frequency distribution of specified columns in a DataFrame.

    Args:
        dataframe: The input DataFrame.
        columns: List of column names to visualize.

    Returns:
        None (displays subplots).
    """
    # Set up subplots
    num_cols = len(columns)
    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(15, 5))

    # Plot countplots for each specified column
    for i, col in enumerate(columns):
        ax = sns.countplot(x=dataframe[col], ax=axes[i])
        ax.set_title(col)

        # Add annotations for each bar in the countplot
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def create_average_variable(dataframe, columns, new_variable_name):
    """
    Create a new variable in a DataFrame as the average of specified columns.

    Args:
        dataframe: The input DataFrame.
        columns: List of column names to calculate the average.
        new_variable_name: Name of the new variable to be created.

    Returns:
        The DataFrame with the new variable added.
    """
    # Check if the specified columns exist in the DataFrame
    if not all(col in dataframe.columns for col in columns):
        raise ValueError("Not all specified columns exist in the DataFrame.")

    # Calculate the average of specified columns
    dataframe[new_variable_name] = dataframe[columns].mean(axis=1).round(2)

    return dataframe


def create_annotated_histogram(dataframe, column, bins=10):
    """
    Create an annotated histogram for a given column in a DataFrame.

    Args:
        dataframe: The input DataFrame.
        column: Name of the column for which the histogram is to be created.
        bins: Number of bins in the histogram (default is 10).

    Returns:
        None (displays the annotated histogram).
    """
    # Set up the figure and axes
    plt.figure(figsize=(10, 6))

    # Create the histogram
    plt.hist(dataframe[column], bins=bins, edgecolor='black')

    # Add annotations and labels
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

    # Show mean and median as vertical lines
    mean_value = dataframe[column].mean()
    median_value = dataframe[column].median()
    
    plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_value:.2f}')

    # Add legend
    plt.legend()

    # Show the histogram
    plt.show()


def visualize_variable_pie(dataframe, variable_column, num_categories=5):
    """
    Visualize the distribution of a variable using a pie chart after discretization.

    Args:
        dataframe: The input DataFrame.
        variable_column: Name of the column to visualize.
        num_categories: Number of categories for discretization (default is 5).

    Returns:
        None (displays the pie chart).
    """
    # Discretize the variable into categories
    categories = pd.cut(dataframe[variable_column], bins=num_categories)

    # Count the occurrences of each category
    category_counts = categories.value_counts()

    # Set up the figure and axes
    plt.figure(figsize=(8, 8))

    # Plot the pie chart
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.0f%%', startangle=90, colors=sns.color_palette('pastel'))

    # Set plot title
    plt.title(f"Variable Distribution - {variable_column}")

    # Show the plot
    plt.show()


def fetch_date_time_nan_columns(dataframe, threshold = 0.8):
    """
    Fetches the null columns and Separates the most nullified columns for the others.

    Parameters:
        dataframe: The DataFrame containing the column with null values.
        threshold (float): A metric with which we separate the dataframe 
                           based upon holding missing values.

    Returns:
        Two lists of columns names.
    """
    # Store the number of null values within a dataframe df
    df = pd.DataFrame(dataframe.isna().sum(), columns=['Nb_null'])

    # Separate the number of nuls that surpass the length * threshold of the dataframe
    cols_to_drop = df[df['Nb_null'] > dataframe.shape[0] * threshold]

    # Separate the number of nuls that are less than the length of the dataframe except 0
    cols_to_manipulate_nan = df[(df['Nb_null'] <= dataframe.shape[0] * threshold) & (df['Nb_null'] > 0)]

    # Return just the name of columns
    return cols_to_drop.index, cols_to_manipulate_nan.index



def drop_columns(dataframe, columns_to_drop):
    """
    Drop one or multiple columns from a DataFrame.
    
    Parameters:
        dataframe: The input DataFrame.
        columns_to_drop (str or list of str): The column name(s) to drop.
        
    Returns:
        pd.DataFrame: The DataFrame with the specified columns dropped.
    """
    if isinstance(columns_to_drop, str):
        columns_to_drop = [columns_to_drop]

    return dataframe.drop(columns=columns_to_drop, axis=1)


def interpolate_datetime_nulls(dataframe, column_name):
    """
    Replace null datetime values in a DataFrame column using time-based interpolation.

    Parameters:
        dataframe: The DataFrame containing the column with null datetime values.
        column_name (str): The name of the column with datetime values to interpolate.

    Returns:
        The DataFrame with null datetime values replaced by interpolated values.
    """
    # Define a custom function for parsing datetime values
    def custom_parser(x):
        try:
            return pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S')
        except ValueError:
            return pd.NaT

    # Apply the custom parser to the specified column
    dataframe[column_name] = dataframe[column_name].apply(custom_parser)

    # Perform interpolation for missing values using 'bfill' method
    dataframe[column_name] = dataframe[column_name].fillna(method='bfill')

    # Perform interpolation for missing values using 'ffill' method
    dataframe[column_name] = dataframe[column_name].fillna(method='ffill')
    
    return dataframe


import pandas as pd

def average_time(dataframe, datetime_columns, column_name):
    """
    Calculate the average start working time across specified datetime columns in a DataFrame.

    Parameters:
        dataframe: The DataFrame containing datetime columns.
        datetime_columns (list): A list of column names with datetime values.
        column_name (str): The name of the new column to be added.

    Returns:
        The DataFrame with the new column containing the average start working time.
    """

    # Select the datetime columns
    datetime_data = dataframe[datetime_columns]

    # Extract the time component from each datetime column and convert to minutes since midnight
    time_data_minutes = datetime_data.apply(lambda x: x.dt.hour * 60 + x.dt.minute)

    # Calculate the average time in minutes across all columns
    average_start_time_minutes = time_data_minutes.mean(axis=1)

    # Convert the average time back to time format
    average_start_time = pd.to_datetime(average_start_time_minutes, unit='m').dt.time

    # Add the new column to the DataFrame
    dataframe[column_name] = average_start_time

    return dataframe


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_categorical_columns(dataframe):
    # Sélectionnez les colonnes catégorielles
    categorical_columns = dataframe.select_dtypes(include=['object', 'category'])

    # Définissez le nombre de sous-graphiques en fonction du nombre de colonnes catégorielles
    num_plots = len(categorical_columns.columns)
    num_cols = 2  # Vous pouvez ajuster le nombre de colonnes en fonction de vos besoins
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Créez une figure et des axes pour les sous-graphiques
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
    fig.subplots_adjust(hspace=0.5)

    for i, column in enumerate(categorical_columns.columns):
        # Calculez les indices de ligne et de colonne pour le sous-graphique actuel
        row_idx = i // num_cols
        col_idx = i % num_cols

        # Tracez le countplot
        sns.countplot(x=column, data=dataframe, ax=axes[row_idx, col_idx])
        axes[row_idx, col_idx].set_title(column)
        axes[row_idx, col_idx].set_xticklabels(axes[row_idx, col_idx].get_xticklabels(), rotation=45, ha='right')

        # Ajoutez des annotations au-dessus des barres
        for p in axes[row_idx, col_idx].patches:
            axes[row_idx, col_idx].annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                            ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # Affichez la figure
    plt.show()

# Exemple d'utilisation avec un dataframe pandas
# Remplacez df par votre propre dataframe
# plot_categorical_columns(df)



def visualize_object_feature_pie(data_frame, feature):
    """
    Visualizes the frequency of unique values for a given object feature in a DataFrame using a pie chart.

    Parameters:
    - data_frame: pandas DataFrame.
    - feature: Name of the object feature to visualize.

    Returns:
    - None (displays the pie chart).
    """
    
    # Check if the specified feature is of type 'object'
    if data_frame[feature].dtype == 'O':
        # Calculate value counts for the feature
        value_counts = data_frame[feature].value_counts()

        # Plot the pie chart
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        plt.title(f'Frequency of {feature}')
        plt.show()
    else:
        print(f"Warning: {feature} is not of type 'object'.")



def visualize_numeric_features_boxplot(data_frame):
    """
    Visualizes the distribution of numeric features in a DataFrame using box plots.

    Parameters:
    - data_frame: pandas DataFrame.

    Returns:
    - None (displays the plot).
    """

    # Select numeric features
    numeric_features = data_frame.select_dtypes(include=['int', 'float']).columns

    # Calculate the number of rows and columns for subplots
    num_features = len(numeric_features)
    num_rows = (num_features // 2) + (num_features % 2)
    num_cols = 2 if num_features > 1 else 1

    # Set up the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    # Loop through each numeric feature and plot the box plot
    for i, feature in enumerate(numeric_features):
        sns.boxplot(x=data_frame[feature], ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Values')

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming df is your DataFrame
# visualize_numeric_features_boxplot(df)




def visualize_numeric_by_binary_object(data_frame, binary_object_variable):
    """
    Visualizes the distribution of all numerical variables based on a binary object variable.

    Parameters:
    - data_frame: pandas DataFrame.
    - binary_object_variable: Name of the binary object variable ('Yes' or 'No').

    Returns:
    - None (displays the plot).
    """
    
    # Check if the specified variable exists in the DataFrame
    if binary_object_variable not in data_frame.columns:
        print("Error: Specified variable not found.")
        return

    # Select numerical variables
    numeric_variables = data_frame.select_dtypes(include=['number']).columns

    # Calculate the number of rows and columns for subplots
    num_variables = len(numeric_variables)
    num_rows = (num_variables // 2) + (num_variables % 2)
    num_cols = 2 if num_variables > 1 else 1

    # Set up the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    # Loop through each numeric variable and plot the distribution
    for i, variable in enumerate(numeric_variables):
        sns.histplot(data_frame, x=variable, hue=binary_object_variable, kde=True, ax=axes[i], palette='Set2')
        axes[i].set_title(f'Distribution of {variable}')
        axes[i].set_xlabel(variable)
        axes[i].set_ylabel('Frequency')

        # Annotate with median values
        for category in data_frame[binary_object_variable].unique():
            subset = data_frame[data_frame[binary_object_variable] == category]
            median_value = subset[variable].median()
            axes[i].annotate(f'{category} Median: {median_value:.2f}', 
                             xy=(median_value, 0), xytext=(20, 20), 
                             textcoords='offset points', ha='center', va='center',
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    # Adjust layout
    plt.tight_layout()
    plt.show()




def replace_outliers_with_median(data_frame, multiplier=1.5):
    """
    Replaces outliers in numerical columns with the median value.

    Parameters:
    - data_frame: pandas DataFrame.
    - multiplier: The multiplier for determining the outlier threshold.

    Returns:
    - df_outliers_removed: A new DataFrame with outliers replaced by the median.
    """

    # Select numerical features
    numerical_features = data_frame.select_dtypes(include=['int', 'float']).columns

    # Create a copy of the original DataFrame to avoid modifying the input DataFrame
    df_outliers_removed = data_frame.copy()

    # Replace outliers with the median for each numerical feature
    for feature in numerical_features:
        q1 = data_frame[feature].quantile(0.25)
        q3 = data_frame[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        # Replace outliers with the median
        df_outliers_removed[feature] = np.where(
            (data_frame[feature] < lower_bound) | (data_frame[feature] > upper_bound),
            data_frame[feature].median(),
            data_frame[feature]
        )

    return df_outliers_removed




def identify_columns_with_outliers(d, threshold=1.5):
    """
    Identify columns with outliers in a DataFrame and calculate the total number of outliers and rows with outliers.

    Parameters:
        df (DataFrame): The input DataFrame.
        threshold (float): The IQR multiplier to determine outliers. Default is 1.5.

    Returns:
        dict: A dictionary containing information about columns with outliers and the number of rows with outliers.
    """
    # Copy dataframe
    df = d.copy()
    
    # Create an empty dictionary and a list to store the results
    outliers_info = {}
    indexes = []
    
    # Variable to count rows with outliers
    rows_with_outliers = 0
    
    # Iterate through columns
    for col in df.columns:
        if df[col].dtype in [np.int64, np.float64]:  # Check if the column is numeric
            # Calculate the IQR for the column
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define the upper and lower bounds for outliers
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Identify outliers in the column
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            indexes.extend(list(outliers.index))

            # Count the number of outliers
            num_outliers = len(outliers)
            
            # Add the column and number of outliers to the dictionary
            if num_outliers > 0:
                outliers_info[col] = num_outliers
                
                # Count the number of rows with outliers
                rows_with_outliers += len(outliers)
    
    # Create a dictionary to return the results
    results = {
        "columns_with_outliers": outliers_info,
        "total_rows_with_outliers": f'{len(set(indexes))} / {df.shape[0]}',
        'total_outliers': len(set(indexes))
    }
    
    return results
