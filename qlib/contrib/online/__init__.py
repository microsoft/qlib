'''
TODO:

- Online needs that the model have such method
    def get_data_with_date(self, date, **kwargs):
        """
        Will be called in online module
        need to return the data that used to predict the label (score) of stocks at date.

        :param
            date: pd.Timestamp
                predict date
        :return:
            data: the input data that used to predict the label (score) of stocks at predict date.
        """
        raise NotImplementedError("get_data_with_date for this model is not implemented.")

'''
