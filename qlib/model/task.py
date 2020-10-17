'''
Please implement similar function here
    
# Rolling relealted

    def split_rolling_periods(
        self,
        train_start_date,
        train_end_date,
        validate_start_date,
        validate_end_date,
        test_start_date,
        test_end_date,
        rolling_period,
        calendar_freq="day",
    ):
        """
        Calculating the Rolling split periods, the period rolling on market calendar.
        :param train_start_date:
        :param train_end_date:
        :param validate_start_date:
        :param validate_end_date:
        :param test_start_date:
        :param test_end_date:
        :param rolling_period:  The market period of rolling
        :param calendar_freq: The frequence of the market calendar
        :yield: Rolling split periods
        """

        def get_start_index(calendar, start_date):
            start_index = bisect.bisect_left(calendar, start_date)
            return start_index

        def get_end_index(calendar, end_date):
            end_index = bisect.bisect_right(calendar, end_date)
            return end_index - 1

        calendar = self.raw_df.index.get_level_values("datetime").unique()

        train_start_index = get_start_index(calendar, pd.Timestamp(train_start_date))
        train_end_index = get_end_index(calendar, pd.Timestamp(train_end_date))
        valid_start_index = get_start_index(calendar, pd.Timestamp(validate_start_date))
        valid_end_index = get_end_index(calendar, pd.Timestamp(validate_end_date))
        test_start_index = get_start_index(calendar, pd.Timestamp(test_start_date))
        test_end_index = test_start_index + rolling_period - 1

        need_stop_split = False

        bound_test_end_index = get_end_index(calendar, pd.Timestamp(test_end_date))

        while not need_stop_split:

            if test_end_index > bound_test_end_index:
                test_end_index = bound_test_end_index
                need_stop_split = True

            yield (
                calendar[train_start_index],
                calendar[train_end_index],
                calendar[valid_start_index],
                calendar[valid_end_index],
                calendar[test_start_index],
                calendar[test_end_index],
            )

            train_start_index += rolling_period
            train_end_index += rolling_period
            valid_start_index += rolling_period
            valid_end_index += rolling_period
            test_start_index += rolling_period
            test_end_index += rolling_period

    def get_rolling_data(
        self,
        train_start_date,
        train_end_date,
        validate_start_date,
        validate_end_date,
        test_start_date,
        test_end_date,
        rolling_period,
        calendar_freq="day",
    ):
        # Set generator.
        for period in self.split_rolling_periods(
            train_start_date,
            train_end_date,
            validate_start_date,
            validate_end_date,
            test_start_date,
            test_end_date,
            rolling_period,
            calendar_freq,
        ):
            (
                x_train,
                y_train,
                x_validate,
                y_validate,
                x_test,
                y_test,
            ) = self.get_split_data(*period)
            yield x_train, y_train, x_validate, y_validate, x_test, y_test

    def get_split_data(
        self,
        train_start_date,
        train_end_date,
        validate_start_date,
        validate_end_date,
        test_start_date,
        test_end_date,
    ):
        """
        all return types are DataFrame
        """
        ## TODO: loc can be slow, expecially when we put it at the second level index.
        if self.raw_df.index.names[0] == "instrument":
            df_train = self.raw_df.loc(axis=0)[:, train_start_date:train_end_date]
            df_validate = self.raw_df.loc(axis=0)[:, validate_start_date:validate_end_date]
            df_test = self.raw_df.loc(axis=0)[:, test_start_date:test_end_date]
        else:
            df_train = self.raw_df.loc[train_start_date:train_end_date]
            df_validate = self.raw_df.loc[validate_start_date:validate_end_date]
            df_test = self.raw_df.loc[test_start_date:test_end_date]

        TimeInspector.set_time_mark()
        df_train, df_validate, df_test = self.process_data(df_train, df_validate, df_test)
        TimeInspector.log_cost_time("Finished setup processed data.")

        x_train = df_train[self.feature_names]
        y_train = df_train[self.label_names]

        x_validate = df_validate[self.feature_names]
        y_validate = df_validate[self.label_names]

        x_test = df_test[self.feature_names]
        y_test = df_test[self.label_names]

        return x_train, y_train, x_validate, y_validate, x_test, y_test

'''
