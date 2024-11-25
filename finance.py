# Secondary packages
import numpy as np
import pandas as pd
import scipy.stats as stats

# Plotting packages
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp

# Data packages
import yfinance as yf

# Package settings
pd.options.plotting.backend = "plotly" # Set pandas plotting package to plotly


class Finance_Tools: #pd.DataFrame
    @staticmethod
    def _help_frequency_int(frequency:str):
        """
        Helper function to convert a frequency string to an integer value.
        
        The function takes a frequency string as input and returns an integer value
        representing the frequency. The mapping between frequency strings and integer
        values is defined in the `frequencies` dictionary.
        
        If the input frequency string is not found in the `frequencies` dictionary, the
        function returns `np.nan`.
        
        Args:
            frequency (str): The frequency string to convert to an integer value.
        
        Returns:
            int or np.nan: The integer value representing the frequency, or `np.nan` if
            the input frequency string is not found in the `frequencies` dictionary.
        """

        frequencies = {'YE':11, 'QE':10, 'ME':9, 'W':8, 'D':7, 'h':6, 'min':5, 's':4, 'ms':3, 'us':2, 'ns':1}

        if frequency in frequencies.keys():
            return frequencies[frequency]
        else:
            return np.nan


    def _help_get_frequency(self, df, tolerance=0.05):
        """
        Helper function to determine the frequency of a given DataFrame.
        
        This function takes a DataFrame as input and analyzes the time differences between
        consecutive rows to determine the most common frequency of the data. It supports
        detecting yearly, quarterly, monthly, weekly, daily, hourly, minutely, secondly,
        millisecond, microsecond, and nanosecond frequencies.
        
        The function returns a tuple containing the detected frequency label (e.g. 'YE', 'QE',
        'ME', 'W', 'D', 'h', 'min', 's', 'ms', 'us', 'ns') and an integer value representing
        the frequency (using the `_help_frequency_int` helper function).
        
        If the frequency cannot be determined, the function returns a custom frequency label
        and `np.nan` for the integer value.
        
        Args:
            df (pandas.DataFrame): The input DataFrame to analyze.
            tolerance (float): The tolerance factor for detecting the frequency. Defaults to 0.05.
        
        Returns:
            tuple: A tuple containing the detected frequency label (str) and the corresponding
            integer value (int or np.nan).

        """

        frequencies = {'YE':11, 'QE':10, 'ME':9, 'W':8, 'D':7, 'h':6, 'min':5, 's':4, 'ms':3, 'us':2, 'ns':1}

        lower_tol = 1 - tolerance
        upper_tol = 1 + tolerance

        time_diffs = df.index.to_series().diff().dropna()
        most_common_diff = time_diffs.mode().iloc[0]

        # Determine the frequency label based on the most common difference
        if pd.Timedelta('252 days') * lower_tol <= most_common_diff <= pd.Timedelta('365 days') * upper_tol:
            frequency = 'YE'  # Year End
            freq_int = self._help_frequency_int(frequency)
        elif pd.Timedelta('60 days') * lower_tol <= most_common_diff <= pd.Timedelta('90 days') * upper_tol:
            frequency = 'QE'  # Quarterly End
            freq_int = self._help_frequency_int(frequency)
        elif pd.Timedelta('20 days') * lower_tol <= most_common_diff <= pd.Timedelta('30 days') * upper_tol:
            frequency = 'ME'  # Monthly End
            freq_int = self._help_frequency_int(frequency)
        elif pd.Timedelta('5 days') * lower_tol <= most_common_diff <= pd.Timedelta('7 days') * upper_tol:
            frequency = 'W'  # Weekly
            freq_int = self._help_frequency_int(frequency)
        elif pd.Timedelta('1 day') * lower_tol <= most_common_diff <= pd.Timedelta('1 day') * upper_tol:
            frequency = 'D'  # Daily
            freq_int = self._help_frequency_int(frequency)
        elif pd.Timedelta('1 hour') * lower_tol <= most_common_diff <= pd.Timedelta('1 hour') * upper_tol:
            frequency = 'h'  # Hourly
            freq_int = self._help_frequency_int(frequency)
        elif pd.Timedelta('1 minute') * lower_tol <= most_common_diff <= pd.Timedelta('1 minute') * upper_tol:
            frequency = 'min'  # Minutely
            freq_int = self._help_frequency_int(frequency)
        elif pd.Timedelta('1 second') * lower_tol <= most_common_diff <= pd.Timedelta('1 second') * upper_tol:
            frequency = 's'  # Secondly
            freq_int = self._help_frequency_int(frequency)
        elif pd.Timedelta('1 millisecond') * lower_tol <= most_common_diff <= pd.Timedelta('1 millisecond') * upper_tol:
            frequency = 'ms'  # Milliseconds
            freq_int = self._help_frequency_int(frequency)
        elif pd.Timedelta('1 microsecond') * lower_tol <= most_common_diff <= pd.Timedelta('1 microsecond') * upper_tol:
            frequency = 'us'  # Microseconds
            freq_int = self._help_frequency_int(frequency)
        elif pd.Timedelta('1 nanosecond') * lower_tol <= most_common_diff <= pd.Timedelta('1 nanosecond') * upper_tol:
            frequency = 'ns'  # Nanoseconds
            freq_int = self._help_frequency_int(frequency)
        else:
            frequency = f"Custom frequency based on {most_common_diff}"
            freq_int = np.nan
        
        return frequency, freq_int
    
    
    @staticmethod
    def _help_get_bins(ret, bin_size=0.005):
        """
        Helper function to determine the number of bins to use for a histogram based 
        on the range of the input data.
        
        Args:
            ret (pandas.Series): The input data to determine the number of bins for.
            bin_size (float, optional): The desired bin size. Defaults to 0.005.
        
        Returns:
            int: The number of bins to use for a histogram of the input data.
        """
        return int(round((ret.max() - ret.min()) / bin_size, 0))


    def get_data_interruptions(self, interrupt_limit=3):
        """
        Returns a Series of datetime differences that exceed the specified interrupt_limit.

        Args:
            interrupt_limit (int, optional): The minimum number of days between interruptions to include. Defaults to 3.

        Returns:
            pandas.Series: A Series of datetime differences that exceed the interrupt_limit.
        """
        interruptions = self.data.index.to_series().diff().dropna()
        return interruptions[interruptions>pd.Timedelta(days=interrupt_limit)]


    def __init__(self, data, name=None): # name is the asset name that will be used
        """
        Initializes a new instance of the class with the provided data.
        
        Args:
            data (pandas.DataFrame or pandas.Series): The input data, which can be a DataFrame or a Series.
            name (str, optional): The name of the asset, which will be used later. Defaults to None.
        
        Raises:
            ValueError: If the input data is not a valid DataFrame or Series, or if the index is not a valid datetime index.
        
        Attributes:
            data (pandas.DataFrame): The input data, with the index converted to a datetime index if necessary.
            data_type (str): The type of data, either "OHLC" or "close".
            main_cols (list or str): The main columns of the data, either the OHLC columns or just the "close" column.
            name (str): The name of the asset.
            start_date (pandas.Timestamp): The start date of the data.
            stop_date (pandas.Timestamp): The end date of the data.
            frequency (str): The frequency of the data, such as "D", "W", or "Custom".
            freq_int (int): An integer representation of the frequency, used for certain calculations.
            get_nas (pandas.DataFrame): A DataFrame containing the dates with missing values.
            get_zeros (pandas.DataFrame): A DataFrame containing the dates with zero values.
            weekday_value_counts (pandas.Series): The value counts per weekday.
            month_value_counts (pandas.Series): The value counts per month.
            year_value_counts (pandas.Series): The value counts per year.
        """
                
        index_is_dt = False # Bool to signify that the data is a series of df and has a datetime index

        # Even if just series is put in convert to df so can standardize on a df
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)

        # Check if index is valid datetime and if not try to pares or error
        if isinstance(data, pd.DataFrame):
            if isinstance(data.index, pd.DatetimeIndex):
                self.data = data
                index_is_dt = True
            else: # try to parse the index as datetime
                try:
                    self.data = data.set_index(pd.to_datetime(data.index))
                    index_is_dt = True
                except:
                    raise ValueError("Data must have a datetime index or datetime parsable string as index")
        else:
            raise ValueError("Input must be a pandas DataFrame or Series")
        

        # If the input is valid (i.e. Series or df with datetime index) check if its just closes or OHLC data
        # This sets the datatype variable for later use in methods to decide if it should be availabale or not
        if index_is_dt:
            # change col names to lower case
            data.columns = data.columns.str.lower()

            ohlc_req_cols = ["open", "high", "low", "close"] # , "volume"             

            if all(col in data.columns for col in ohlc_req_cols): # check if its an OHLC df
                self.data_type = "OHLC"
                self.main_cols = ohlc_req_cols

            elif "close" in data.columns:  # Check if at least close is present
                self.data_type = "close"
                self.main_cols = "close"

            else:
                raise ValueError(f"Invalid Input: Must contain either columns {ohlc_req_cols} or at least 'close'")


        # Data is valid, create all the attributes here
        if index_is_dt:
            data = self.data[self.main_cols] # localize data with only the main cols
            self.name = name

            # Show the date range of the data
            self.start_date = data.index.min()
            self.stop_date = data.index.max()

            self.frequency, self.freq_int = self._help_get_frequency(data) # Gets the data frequency as D, W, ...

            self.get_nas = data[data.isna()] # Gets a list of all the missing values in the data
            self.get_zeros = data[data==0] # Gets a list of all the zeros in the data

            self.weekday_value_counts = self.data.index.day_name().value_counts(sort=False)
            self.month_value_counts = self.data.index.month_name().value_counts(sort=False)
            self.year_value_counts = self.data.index.year.value_counts(sort=False)

        
    def __repr__(self):
        """
        Shows just the data df when printed out
        """
        return str(self.data)
    

    def data_analytics(self):
        """
        Generates a detailed data analytics report for the Finance_Tools object.
        
        The report includes the following information:
        - Data range: the start and end dates of the data
        - Frequency: the frequency of the data (e.g. daily, weekly, etc.)
        - Data type: the type of data (e.g. OHLC, close only)
        - Number of dates with NA values
        - Number of dates with 0 values
        - Number of data interruptions (larger than a normal weekend)
        - Value counts per weekday (for daily or higher frequency data)
        - Value counts per month (for monthly or higher frequency data)
        - Value counts per year (for yearly or higher frequency data)
        
        The report is printed to the console.
        """
                
        # Create repr string for the class
        representation = f"""
Data range: {self.start_date} - {self.stop_date}
Frequency: {self.frequency}
Data type: {self.data_type}

----------------------------------------------
{len(self.get_nas)} Dates with NA values
{len(self.get_zeros)} Dates with 0 values
{len(self.get_data_interruptions())} Interruptions (larger than a normal weekend)
See entire interruptions with get_data_interruptions()
----------------------------------------------
        """

        if self.freq_int <= 8:
            representation += f"""
Value counts per weekday
{self.weekday_value_counts}
----------------------------------------------
        """
        if self.freq_int <= 9:
            representation += f"""
Value counts per month
{self.month_value_counts}
----------------------------------------------
        """
        if self.freq_int <= 10:
            representation += f"""
Value counts per year
{self.year_value_counts}
----------------------------------------------
        """
            
        return representation
        
    # Returns
    def returns(self, return_pd = True, print_details=False): # Returns the arithmetic returns without nans of close prices
        """
        Returns the arithmetic returns without nans of close prices.
        
        This method calculates the arithmetic returns (percent change) of the close prices, excluding any NaN values.
        
        Args:
            return_pd (bool): If True, returns the returns as a pandas Series. If False, returns a new Finance_Tools instance with the returns data.
            print_details (bool): If True, prints the NaN values that were removed from the returns calculation.
        
        Returns:
            pandas.Series or Finance_Tools: The arithmetic returns without NaN values.
        """
        ret = self.data['close'].pct_change()
        ret.name = 'close'

        if print_details:
            print(f"Arithmetic returns deleted nans: {ret[ret.isna()]}")

        if return_pd:
            return ret.dropna()
        else:
            return Finance_Tools(ret.dropna(), name=self.name)
    
    # Returns
    def log_returns(self, return_pd = True, print_details=False): # Returns Log returns without nan values of close prices
        """
        Returns the log returns without NaN values of the close prices.
        
        This method calculates the log returns (percent change) of the close prices, excluding any NaN values.
        
        Args:
            return_pd (bool): If True, returns the log returns as a pandas Series. If False, returns a new Finance_Tools instance with the log returns data.
            print_details (bool): If True, prints the NaN values that were removed from the log returns calculation.
        
        Returns:
            pandas.Series or Finance_Tools: The log returns without NaN values.
        """
                
        ret = np.log(self.data['close'] / self.data['close'].shift(1))
        ret.name = 'close'

        if print_details:
            print(f"Log returns deleted nans: {ret[ret.isna()]}")

        if return_pd:
            return ret.dropna()
        else:
            return Finance_Tools(ret.dropna(), name=self.name)
    
    # Returns
    def intraday_returns(self, return_pd = True, print_details=False): #arithmetic ret of open to close
        """
        Returns the return of open to close relative to the open price.
        
        This method calculates the intraday returns, which are the returns from the open price to the close price, relative to the open price. This can be used on data of any frequency.
        
        Args:
            return_pd (bool): If True, returns the intraday returns as a pandas Series. If False, returns a new Finance_Tools instance with the intraday returns data.
            print_details (bool): If True, prints the NaN values that were removed from the intraday returns calculation.
        
        Returns:
            pandas.Series or Finance_Tools: The intraday returns without NaN values.


        can be used on all kinds of frequencies

        """
        # Check if data is OHLC
        if self.data_type != 'OHLC':
            raise ValueError(f"Invalid Finance_Tools data type: is: {self.data_type}, requires: OHLC")
        
        data = self.data # localize df

        ret = (data['close'] - data['open']) / data['open']
        ret.name = 'close'

        if print_details:
            print(f"Intraday returns deleted nans: {ret[ret.isna()]}")

        if return_pd:
            return ret.dropna()
        else:
            return Finance_Tools(ret.dropna(), name=self.name)

    # Returns
    def overnight_returns(self, return_pd = True, print_details=False): #arithmetic ret of last close to open
        """
        Returns the return from the last close to today's open.
        
        This method calculates the overnight returns, which are the returns from the last close price to the current open price. This can be used on data of any frequency.
        
        Args:
            return_pd (bool): If True, returns the overnight returns as a pandas Series. If False, returns a new Finance_Tools instance with the overnight returns data.
            print_details (bool): If True, prints the NaN values that were removed from the overnight returns calculation.
        
        Returns:
            pandas.Series or Finance_Tools: The overnight returns without NaN values.


        can be used in all kinds of frequencies
        """
        if self.data_type != 'OHLC':
            raise ValueError(f"Invalid Finance_Tools data type: is: {self.data_type}, requires: OHLC")
        
        data = self.data # localize df

        ret = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        ret.name = 'close'

        if print_details:
            print(f"Overnight returns deleted nans: {ret[ret.isna()]}")

        if return_pd:
            return ret.dropna()
        else:
            return Finance_Tools(ret.dropna(), name=self.name)
        
    # Returns
    def high_low_range(self, return_pd = True): #relative range of high to low in relation to ?open?
        """
        Returns the relative range of the high and low prices compared to the open price.
        
        This method calculates the relative range of the high and low prices compared to the open price. The relative range is calculated as (high - low) / open. This can be used on data of any frequency.
        
        Args:
            return_pd (bool): If True, returns the relative range as a pandas Series. If False, returns a new Finance_Tools instance with the relative range data.
        
        Returns:
            pandas.Series or Finance_Tools: The relative range without NaN values.
        """
        
        if self.data_type != 'OHLC':
            raise ValueError(f"Invalid Finance_Tools data type: is: {self.data_type}, requires: OHLC")
        
        data = self.data
        ret = (data['high'] - data['low']) / data['open']
        ret.name = 'close'
        
        print(f"High low relative range deleted nans: {ret[ret.isna()]}")

        if return_pd:
            return ret.dropna()
        else:
            return Finance_Tools(ret.dropna(), name=self.name)

    # Returns
    def weekend_return(self, return_pd=True, show_hist=False, bin_size=0.001):
        """
        Returns a DataFrame containing the weekend returns for the given Finance_Tools instance.
        
        The method first filters the data to only include Fridays, then iterates through each Friday and finds the next available trading day after it. It calculates the return from the Friday's close price to the next open price, and stores this information in a DataFrame.
        
        The method can optionally show a histogram of the weekend returns, and returns the DataFrame containing the weekend returns data.
        
        Args:
            return_pd (bool): If True, returns the weekend returns DataFrame. If False, returns a new Finance_Tools instance with the weekend returns data.
            show_hist (bool): If True, shows a histogram of the weekend returns.
            bin_size (float): The bin size for the histogram.
        
        Returns:
            pandas.DataFrame or Finance_Tools: The weekend returns data.

            
        returns df with friday, next open day, weekend ret, next open day weekday name
        prints errors if weekend returns cant be found
        prints significance test results later 

        ToDo:
        !!! maybe outsource hist, significance test to methods that take the weekend_returns_df as input?
        """

        # Requires OHLC data
        if self.data_type != 'OHLC':
            raise ValueError('Method requires OHLC data!')
        
        # Localize data & get weekday names
        data = self.data
        data['weekday'] = data.index.day_name()

        # Step 1: Filter out Fridays
        fridays = data[data['weekday'] == 'Friday']

        # Step 2: Iterate through each Friday and find the next available trading day
        weekend_returns = []

        # Loop through all the fridays
        for friday_index in fridays.index:
            # Get close price on Friday
            friday_close = data.loc[friday_index, 'close']
            
            # Get the next trading day after Friday
            next_trading_day = data.loc[friday_index:].iloc[1:].first_valid_index()
            
            if next_trading_day is not None:
                # Find the open price of the next trading day
                next_open = data.loc[next_trading_day, 'open']
                
                # Calculate the return from Friday's close to the next open
                weekend_return = (next_open - friday_close) / friday_close
                weekend_returns.append({
                    'friday': friday_index,
                    'next_open_day': next_trading_day,
                    'weekend_return': weekend_return
                })
            else:
                print(f'Warning: no next trading day found for {friday_index}')

        # Convert to a DataFrame for easier analysis
        weekend_returns_df = pd.DataFrame(weekend_returns)
        weekend_returns_df['next_open_weekday'] = weekend_returns_df['next_open_day'].dt.day_name()


        if show_hist:
            fig = px.histogram(weekend_returns_df['weekend_return'], nbins=self._help_get_bins(weekend_returns_df['weekend_return'], bin_size=bin_size))
            fig.show()


        if return_pd:
            return weekend_returns_df


    def simple_weekend_returns(self, return_pd=True):
        """
        Returns the weekend returns with a datetime index with the date of the weekends' Sunday.
        
        This method calculates the weekend returns by taking the difference between the open price on the next trading day after Friday and the close price on the previous Friday, divided by the previous Friday's close price. The returns are shifted to have the Sunday date as the index.
        
        If the data frequency is less than weekly, the data is increased to weekly frequency before calculating the returns. If the frequency is higher than weekly, an error is raised.
        
        Args:
            return_pd (bool, optional): If True, returns the weekend returns as a pandas DataFrame. If False, returns a Finance_Tools object with the weekend returns. Defaults to True.
        
        Returns:
            pandas.DataFrame or Finance_Tools: The weekend returns.

        """
        
        if self.freq_int < 8:
            data = self.increase_frequency(to_frequency='W')
        elif self.freq_int == 9:
            data = self.data
        else:
            raise ValueError("Frequency is higher than weekly, input lower frequent data")

        ret = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)).shift(-1)

        if return_pd:
            return ret.dropna()
        else:
            return Finance_Tools(ret.dropna(), name=self.name)
        

    def test_all_returns(self):
        """
        Test all kinds of returns for significance.
        
        This method tests the statistical significance of different types of returns, 
        including intraday returns, overnight returns, and weekend returns. 
        It prints the results of the normality tests for each type of return.
        
        Todo:
        - Adjust the method to work with all kinds of data frequencies?

        """

        print("Intraday returns")
        intra_sig, _ = self.returns(return_pd=False).ret_normal_test(self.intraday_returns())

        print("Overnight returns")
        overnight_sig, _ = self.returns(return_pd=False).ret_normal_test(self.overnight_returns())

        print("Weekend returns")
        weekend_sig, _ = self.returns(return_pd=False).ret_normal_test(self.simple_weekend_returns())

        return intra_sig, overnight_sig, weekend_sig

    # Returns
    def weekday_returns(self, hist_comp=False, alpha=0.05):
        """
        Calculates and analyzes the returns of a financial asset by weekday.
        
        This method calculates the returns of a financial asset by weekday, and performs statistical tests 
        to determine if the mean return for each weekday is significantly different from the overall mean return.
        
        The method returns a pandas DataFrame containing the following statistics for each weekday:
        - mean_return: The mean return for the weekday.
        - st_dev: The standard deviation of the returns for the weekday.
        - mean_diff: The difference between the weekday mean return and the overall mean return.
        - st_dev_diff: The difference between the weekday standard deviation and the overall standard deviation.
        - t_stat: The t-statistic for the one-sample t-test comparing the weekday mean return to the overall mean return.
        - p_value: The p-value for the one-sample t-test.
        
        The method also has an option to display a histogram of the returns by weekday.
        
        Args:
            hist_comp (bool, optional): If True, displays a histogram of the returns by weekday. Defaults to False.
            alpha (float, optional): The significance level for the statistical tests. Defaults to 0.05.
        
        Returns:
            pandas.DataFrame: A DataFrame containing the statistics for each weekday.


        !!! Warning: Monday includes the weekend ret !!!
        To-Do:
        - maybe fix the binsize in histogram to 0.1%?

        Questions: 
        ? add return type to have option to use intraday return 
        only to not have dependency on previous day? , return_type='close'
        """
        ret = pd.DataFrame(self.returns())
        ret['weekday'] = ret.index.day_name()

        if hist_comp:
            fig = px.histogram(ret['close'], color=ret['weekday'], opacity=0.6)
            fig.show()


        #   Test weekday returns for significance
        weekdays = ret['weekday'].unique() # Get list of weekdays

        # variable to store results
        weekday_stats = {'p_value':dict(),
                        't_stat':dict(),
                        'mean_return':dict(),
                        'st_dev':dict(),
                        'mean_diff':dict(),
                        'st_dev_diff':dict(),
                        }

        overall_mean = ret['close'].mean()
        overall_std = ret['close'].std()

        for day in weekdays:
            weekday_returns = ret[ret['weekday'] == day]['close']
            
            weekday_stats['mean_return'][day] = weekday_returns.mean()
            weekday_stats['st_dev'][day] = weekday_returns.std()
            weekday_stats['mean_diff'][day] = weekday_returns.mean() - overall_mean
            weekday_stats['st_dev_diff'][day] = weekday_returns.std() - overall_std

            # Perform one-sample t-test comparing the mean return of the weekday to the overall mean
            weekday_stats['t_stat'][day], weekday_stats['p_value'][day] = stats.ttest_1samp(weekday_returns, overall_mean)
            
            # Spell out results
            if weekday_stats['p_value'][day] < alpha:
                print(f"{day}: Mean return is significantly different from the overall mean (p-value = {weekday_stats['p_value'][day]:.4f})")
            else:
                pass
                #print(f"{day}: No significant difference in mean return compared to the overall mean (p-value = {weekday_stats['p_value'][day]:.4f})")

        return pd.DataFrame(weekday_stats).sort_values(by='p_value')

    # Graphing
    def ret_hist(self, bin_size=0.001, show=True):
        """
        Plots a histogram of the returns for the given data, with the mean and standard deviation marked.
        
        Args:
            bin_size (float): The size of the bins in the histogram.
            show (bool): Whether to display the plot.
        
        Returns:
            plotly.graph_objects.Figure: The generated histogram plot.
        
        Example:
            Finance_Tools(data, asset).returns(return_pd=False).ret_hist(show=False)

            
        Expects returns as financetools
        """
        #ret = self.returns()
        ret = self.data['close']
        ret_mean = ret.mean()
        ret_std = ret.std()


        fig = go.Figure()
        fig.add_trace(go.Histogram(x=ret, 
                                   name=self.name,
                                   opacity=0.5,
                                   xbins=dict(size=bin_size)
                                   ))
        
        # Manually compute the histogram to get y_max
        counts, bins = np.histogram(ret, bins=np.arange(ret.min(), ret.max(), bin_size))
        y_max = max(counts)
        
        # Add vertical line for mean
        fig.add_trace(go.Scatter(x=[ret_mean, ret_mean],
                                y=[0, y_max],
                                mode='lines',
                                name='Mean',
                                line=dict(color='red', width=2)))
        
        # Add vertical lines for mean ± std deviation
        fig.add_trace(go.Scatter(
            x=[ret_mean - ret_std, ret_mean - ret_std],
            y=[0, y_max],
            mode='lines',
            name='Mean - 1 Std Dev',
            line=dict(color='blue', width=1, dash='dot')
        ))

        fig.add_trace(go.Scatter(
            x=[ret_mean + ret_std, ret_mean + ret_std],
            y=[0, y_max],
            mode='lines',
            name='Mean + 1 Std Dev',
            line=dict(color='blue', width=1, dash='dot')
        ))
        
        # Combine the histograms into one figure
        fig.update_layout(
            barmode='overlay',  # 'overlay' will stack them transparently; 'group' will place them side by side
            title='Return distribution',
            xaxis_title='Return',
            yaxis_title='Occurence'
        )

        # Add normal distribution line for comparison
        x_values = np.linspace(ret.min(), ret.max(), 500)
        y_values = (1 / (ret_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - ret_mean) / ret_std) ** 2)
        y_values = y_values * len(ret) * bin_size  # Scale to histogram
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='green', width=1, dash='dash')
        ))

        # Add the means written on the chart 
        fig.add_annotation(
            x=0,  # Left edge of the plot
            y=1,  # Top of the plot
            xref='paper',
            yref='paper',
            xanchor='left',
            yanchor='top',
            text=f"{self.name} Mean: {ret_mean:.4f}",
            showarrow=False,
            font=dict(color='red')
        )


        if show:
            fig.show()

        return fig

    # Graphing
    def ret_hist_comparison(self, other_returns, other_name, show=True, bin_size=0.001):
        """
        Graphs the data series as histogram with another histogram of returns from the input.
        
        Expects returns as financetools objects.
        
        Args:
            other_return (pandas.Series): The return series to compare against.
            show (bool): Whether to display the plot. Default is True.
            bin_size (float): The size of the histogram bins. Default is 0.001.
        
        Returns:
            plotly.graph_objects.Figure: The generated plot.


        ToDo:
        Maybe separate the go.histogram creation into its own function?

        Example: 
        lol.returns(return_pd=False).ret_hist_comparison(gog.returns(), show=False)

        """
        # Get the returns for the current object and the benchmark
        ret = self.data['close']
        #bench_data = yf.download(benchmark, self.start_date)['Close']
        bench = Finance_Tools(other_returns).returns(return_pd=True)
        benchmark = other_name

        # Compute mean and standard deviation for both datasets
        ret_mean = ret.mean()
        ret_std = ret.std()
        bench_mean = bench.mean()
        bench_std = bench.std()

        # Create the histogram figure
        fig = go.Figure()

        # Add histogram for the benchmark
        fig.add_trace(go.Histogram(
            x=bench,
            name=benchmark,
            opacity=0.5,
            xbins=dict(size=bin_size)
        ))

        # Add histogram for the current object's returns
        fig.add_trace(go.Histogram(
            x=ret,
            name=self.name,
            opacity=0.5,
            xbins=dict(size=bin_size)
        ))

        # Manually compute the histograms to get y_max for plotting vertical lines
        combined_min = min(ret.min(), bench.min())
        combined_max = max(ret.max(), bench.max())
        bins = np.arange(combined_min, combined_max + bin_size, bin_size)
        counts_ret, _ = np.histogram(ret, bins=bins)
        counts_bench, _ = np.histogram(bench, bins=bins)
        y_max = max(max(counts_ret), max(counts_bench))

        # Add vertical lines for mean and std deviation of the current object's returns
        fig.add_trace(go.Scatter(
            x=[ret_mean, ret_mean],
            y=[0, y_max],
            mode='lines',
            name=f'{self.name} Mean',
            line=dict(color='red', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[ret_mean - ret_std, ret_mean - ret_std],
            y=[0, y_max],
            mode='lines',
            name=f'{self.name} Mean - 1 Std Dev',
            line=dict(color='red', width=1, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=[ret_mean + ret_std, ret_mean + ret_std],
            y=[0, y_max],
            mode='lines',
            name=f'{self.name} Mean + 1 Std Dev',
            line=dict(color='red', width=1, dash='dot')
        ))

        # Add vertical lines for mean and std deviation of the benchmark
        fig.add_trace(go.Scatter(
            x=[bench_mean, bench_mean],
            y=[0, y_max],
            mode='lines',
            name=f'{benchmark} Mean',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[bench_mean - bench_std, bench_mean - bench_std],
            y=[0, y_max],
            mode='lines',
            name=f'{benchmark} Mean - 1 Std Dev',
            line=dict(color='blue', width=1, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=[bench_mean + bench_std, bench_mean + bench_std],
            y=[0, y_max],
            mode='lines',
            name=f'{benchmark} Mean + 1 Std Dev',
            line=dict(color='blue', width=1, dash='dot')
        ))

        # Add normal distribution curves for the current object's returns
        x_values_ret = np.linspace(combined_min, combined_max, 500)
        y_values_ret = (
            (1 / (ret_std * np.sqrt(2 * np.pi))) *
            np.exp(-0.5 * ((x_values_ret - ret_mean) / ret_std) ** 2)
        )
        y_values_ret *= len(ret) * bin_size  # Scale to histogram
        fig.add_trace(go.Scatter(
            x=x_values_ret,
            y=y_values_ret,
            mode='lines',
            name=f'{self.name} Normal Distribution',
            line=dict(color='red', width=1, dash='dash')
        ))

        # Add normal distribution curves for the benchmark
        x_values_bench = np.linspace(combined_min, combined_max, 500)
        y_values_bench = (
            (1 / (bench_std * np.sqrt(2 * np.pi))) *
            np.exp(-0.5 * ((x_values_bench - bench_mean) / bench_std) ** 2)
        )
        y_values_bench *= len(bench) * bin_size  # Scale to histogram
        fig.add_trace(go.Scatter(
            x=x_values_bench,
            y=y_values_bench,
            mode='lines',
            name=f'{benchmark} Normal Distribution',
            line=dict(color='blue', width=1, dash='dash')
        ))

        # Add the means written on the chart 
        fig.add_annotation(
            x=0,  # Left edge of the plot
            y=1,  # Top of the plot
            xref='paper',
            yref='paper',
            xanchor='left',
            yanchor='top',
            text=f"{self.name} Mean: {ret_mean:.4f}",
            showarrow=False,
            font=dict(color='red')
        )

        # Annotation for the benchmark's mean
        fig.add_annotation(
            x=0,  # Left edge of the plot
            y=0.95,  # Slightly below the first annotation
            xref='paper',
            yref='paper',
            xanchor='left',
            yanchor='top',
            text=f"{benchmark} Mean: {bench_mean:.4f}",
            showarrow=False,
            font=dict(color='blue')
        )


        # Update the layout of the figure
        fig.update_layout(
            barmode='overlay',  # Overlay histograms
            title='Comparison of Return Distributions',
            xaxis_title='Return',
            yaxis_title='Occurrence',
            legend=dict(
                x=1.05,
                y=1,
                bgcolor='rgba(255,255,255,0)',
                bordercolor='rgba(255,255,255,0)'
            )
        )

        # Show the figure
        if show:
            fig.show()

        return fig

    # Graphing utilizes ret_hist_comparison
    def ret_hist_benchmark(self, show=True, bin_size=0.001, benchmark='^SPX'):
        """
        Graphs the return distribution of the current object and a benchmark,
        including mean and standard deviation lines, as well as normal distribution curves.

        Expects returns as Finance_Tools objects.

        Args:
            show (bool): Whether to display the plot. Default is True.
            bin_size (float): The size of the histogram bins. Default is 0.001.
            benchmark (str): The ticker symbol of the benchmark to compare against. Default is '^SPX'.

        Returns:
            plotly.graph_objects.Figure: The generated plot.

        Example:
            finance_tool.ret_hist_benchmark(show=True, bin_size=0.001, benchmark='^SPX')
        """

        # Get the benchmark
        bench_data = yf.download(benchmark, self.start_date)['Close']
        #bench = Finance_Tools(bench_data).returns(return_pd=True)
        #Applyt the ret hist function with SPX
        fig = self.returns(return_pd=False).ret_hist_comparison(bench_data, benchmark, show=False, bin_size=bin_size)

        if show:
            fig.show()

        return fig


    # Candlestick chart
    def candlestick(self, show=True, show_vol=False):
        """
        Generates a candlestick chart with optional volume bar chart for the given OHLC data.
        
        Parameters:
            show (bool): If True, the chart will be displayed. If False, the chart object will be returned.
        
        Returns:
            plotly.graph_objects.Figure: The generated candlestick chart figure.
        
        Raises:
            ValueError: If the data type is not 'OHLC'.

            
        Requires OHLC data
        """
        
        if self.data_type != 'OHLC':
            raise ValueError(f"Invalid Finance_Tools data type: is: {self.data_type}, requires: OHLC")
        
        df = self.data
        # Calc MA
        df['250MA'] = df['close'].rolling(250).mean()
        df['60MA'] = df['close'].rolling(60).mean()

        # Create a subplot with two rows, one for the candlestick and one for the volume bar chart
        if 'volume' in df.columns and show_vol:
            # Convert volume to money units if its of type integer
            if pd.api.types.is_integer_dtype(df['volume']):
                df['volume'] = df['volume'] * df['close']

            fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.02,
                                row_heights=[0.9, 0.1],  # Adjust the relative height of the plots
                                #subplot_titles=('OHLC Candlestick Chart', 'Volume')
                                )
        else:
            fig = sp.make_subplots(rows=1, cols=1, shared_xaxes=True,
                                vertical_spacing=0.02,
                                #subplot_titles=('OHLC Candlestick Chart',)
                                )

        # Add candlestick trace to the first row
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=self.name
        ), row=1, col=1)

        # Add MA 60 & 250 to the first row
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['250MA'],
            mode='lines',
            name='250-Day MA',
            line=dict(color='blue', width=2)  # Customize the color and width as needed
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['60MA'],
            mode='lines',
            name='60-Day MA',
            line=dict(color='red', width=1)  # Customize the color and width as needed
        ), row=1, col=1)

        # Add volume bar chart trace to the second row if volume data exists
        if 'volume' in df.columns and show_vol:
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker=dict(color='green')
            ), row=2, col=1)

        # Customize the layout for better appearance
        fig.update_layout(
            title='Historical OHLC Candlestick Chart' + (' with Volume' if 'volume' in df.columns else ''),
            #xaxis_title='Date',
            #yaxis_title='Price',
            #yaxis2_title='Volume' if 'volume' in df.columns else None,
            xaxis_rangeslider_visible=False,  # To hide the range slider under the chart, set to True to display it
            showlegend=False,  # You can enable this if you prefer having a legend
            height = 500
        )

        # Update x-axis properties to be shared across the subplots
        fig.update_xaxes(matches='x')

        # Show the figure
        if show:
            fig.show()

        return fig

    # Stats
    def ret_normal_test(self, test_returns, print_test=False, alpha=0.05):
        """
        Performs a statistical test to determine if the mean of a test set of returns is significantly 
        different from the mean of the overall returns.
        
        Args:
            test_returns (pandas.Series): The test set of returns to be compared to the overall returns.
            print_test (bool, optional): If True, prints the result of the statistical test. Defaults to False.
            alpha (float, optional): The significance level for the statistical test. Defaults to 0.05.
        
        Returns:
            bool: True if the test set of returns is significantly different from the overall returns, False otherwise.
            pandas.DataFrame: A DataFrame containing the mean and standard deviation of the overall returns, the test set of returns, and the difference between them.
        
        Example:
            Finance_Tools(data, asset).returns(return_pd=False).ret_normal_test(Finance_Tools(data, asset).weekend_return()['weekend_return'])
            
    
        Expects returns financetools object with name 'close'
            -> expected returns are used as population
            test_returns used as sample returns

        ToDo:
        - test for same timeframe of returns input
        - create comparative histogram of the test set and population to see shift visually
        """

        # Results storage
        result = {'returns':dict(),
                  'test_set':dict(),
                  'difference':dict()}

        # Test for significance of the weekend returns
        ret = self.data['close']
        overall_mean = ret.mean()
        overall_std = ret.std()
        # Store data 
        result['returns']['mean'], result['returns']['std'] = overall_mean, overall_std

        test_mean = test_returns.mean()
        test_std = test_returns.std()
        # Store data 
        result['test_set']['mean'], result['test_set']['std'] = test_mean, test_std

        # Calculate difference between the sample mean and the population mean
        mean_diff = test_mean - overall_mean
        std_diff = test_std - overall_std
        result['difference']['mean'], result['difference']['std'] = mean_diff, std_diff

        t_stat, p_value = stats.ttest_1samp(test_returns, overall_mean)
        significant = p_value < alpha

        if print_test:
            if significant:
                print(f"Weekend returns are significantly different from the overall returns (p-value = {p_value:.4f}, tstat = {t_stat:.4f})")
            else:
                print(f"Weekend returns are NOT significantly different from the overall returns (p-value = {p_value:.4f}, tstat = {t_stat:.4f})")

        return significant, pd.DataFrame(result)

    # Conversion
    def increase_frequency(self, to_frequency='W', return_pd=True):
        """
        Increases the frequency of the data in the Finance_Tools object to the specified frequency.
        
        Parameters:
            to_frequency (str): The frequency to increase the data to. Supported frequencies include:
                - 'YE': Yearly
                - 'QE': Quarterly
                - 'ME': Monthly
                - 'BME': Business monthly
                - 'BQE': Business quarterly
                - 'BYE': Business yearly
                - 'W': Weekly
                - 'D': Daily
                - 'h': Hourly
                - 'min': Minutes
                - 's': Seconds
                - 'ms': Milliseconds
                - 'us': Microseconds
                - 'ns': Nanoseconds
            return_pd (bool): If True, returns the resampled data as a pandas DataFrame. If False, returns a new Finance_Tools object with the resampled data.
        
        Raises:
            ValueError: If the target frequency is smaller or equal to the current frequency.
        
        Returns:
            pandas.DataFrame or Finance_Tools: The resampled data, either as a DataFrame or a new Finance_Tools object.


        Example: Finance_Tools(data, asset).increase_frequency(to_frequency='W')

        ToDo:
        - add Business month, quarter, year?: BME, BQE, BYE

        """
        df = self.data

        if self._help_frequency_int(to_frequency) > self.freq_int:
            df = df.resample(to_frequency).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum' #if 'volume' in data.columns else 'mean'
            })
        else:
            raise ValueError(f'Target frequency {to_frequency} is smaller or equal to the current frequency {self.frequency}')

        if return_pd:
            return df
        else:
            return Finance_Tools(df, self.name)
        

    def volatility(self, window=252, return_pd=True):
        """
        Calculates the volatility of the asset data using a rolling window.

        Parameters:
            window (int): The size of the rolling window to use for the volatility calculation. Defaults to 252 (the number of trading days in a year).

        Returns:
            pandas.Series: The volatility of the asset data, calculated using a rolling window.
        """

        vol = self.returns().rolling(window=window).std()

        if return_pd:
            return vol
        else:
            return Finance_Tools(vol, self.name)


    def correlation(self, other_series, window=252, convert_to_returns=False, return_pd=True):
        """
        Calculates the correlation between the asset data and another time series, using a rolling window.
        
        Parameters:
            other_series (pandas.Series): The other time series to calculate the correlation with.
            window (int): The size of the rolling window to use for the correlation calculation. Defaults to 252 (the number of trading days in a year).
            convert_to_returns (bool): If True, the function will convert both the asset data and the other time series to returns before calculating the correlation. Defaults to False.
            return_pd (bool): If True, the function will return the correlation as a pandas.Series. If False, the function will return a Finance_Tools object containing the correlation.
        
        Returns:
            pandas.Series or Finance_Tools: The correlation between the asset data and the other time series, calculated using a rolling window.

        Example: lol.correlation(goog, convert_to_returns=True)
        """

        if convert_to_returns:
            other_series = Finance_Tools(other_series).returns()
            corr = self.returns().rolling(window=window).corr(other_series)
        else:
            corr = self.data['close'].rolling(window=window).corr(other_series)

        if return_pd:
            return corr
        else:
            return Finance_Tools(corr, self.name)


    def cumulative_return(self, return_pd=True):
        """
        Calculates the cumulative return of the asset data.
        
        Returns:
            pandas.Series or Finance_Tools: The cumulative return of the asset data.

        ToDo:
        - add date restricter for considered date range? data.loc['01-01-2020':]
        """
        df = self.data
        cum_ret = df['close'] / df['close'].iloc[0]

        if return_pd:
            return cum_ret
        else:
            return Finance_Tools(cum_ret, self.name)


    def cumulative_comparison(self, other_series, plot=False):
        """
        Calculates and compares the cumulative returns of the asset data and another time series.

        Parameters:
            other_series (pandas.Series): The other time series to compare the cumulative return with.
            plot (bool): If True, the function will plot the cumulative returns of both the asset data and the other time series. Defaults to False.

        Returns:
            pandas.DataFrame: A DataFrame containing the cumulative returns of the asset data and the other time series, with the column names being the asset name and 'other'.
        
        ToDo:
        - maybe also add datetime restricter like in cum returns?
        
        """     
        og = self.cumulative_return()
        other_series = Finance_Tools(other_series).cumulative_return()

        both = pd.concat([og, other_series], axis=1)
        both.columns = [self.name, 'other']

        if plot:
            fig = both.plot()
            fig.show()
        
        return both
