from cvxopt.modeling import op

__author__ = 'Dworkin'

import Settings
import pandas as pd
from Settings import data_dir_path
import numpy as np
import datetime

from sklearn.feature_extraction import DictVectorizer


def dirty_trick():
    users = pd.DataFrame.from_csv(data_dir_path+Settings.user_list, index_col=False)
    null_users = users[(users.WITHDRAW_DATE.isnull()==False)&(users.WITHDRAW_DATE<'2012-06-24')]['USER_ID_hash']
    sample_submission_df = pd.DataFrame.from_csv(data_dir_path+Settings.sample_submission, index_col=False)

    sample_submission_df.ix[sample_submission_df.USER_ID_hash.isin(null_users)==False,
                            'PURCHASED_COUPONS'] = 'c9e1dcbd8c98f919bf85ab5f2ea30a9d 2fcca928b8b3e9ead0f2cecffeea50c1 0fd38be174187a3de72015ce9d5ca3a2'


    sample_submission_df.to_csv('Lucky_coupon_without_withdraw.csv', index=False)


def get_feature_vectorizer (df, cols):
    """

    :return: vectorizers 1-hot-encoding for feature
    """

    feature_vect = DictVectorizer(sparse=True)
    feature_vect.fit_transform(df[cols].to_dict(outtype='records'))
    return feature_vect



if __name__ == '__main__':

    #dirty_trick()

    #train data
    users = pd.DataFrame.from_csv(data_dir_path+Settings.user_list, index_col=False)
    coupons = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_list_train, index_col=False)
    view_log = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_visit_train, index_col=False)
    purchase_log = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_detail_train, index_col=False)
    coupon_area = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_area_train, index_col=False)

    #test data
    coupons_test = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_list_test, index_col=False)

    users_vect = get_feature_vectorizer(users, ['USER_ID_hash', 'SEX_ID'])





    withdraw_purchase = view_log[
        (view_log.USER_ID_hash.isin(users[users.WITHDRAW_DATE.isnull()==False].USER_ID_hash))
        & (view_log.PURCHASE_FLG==1)]
    last_purchase_withdraw = withdraw_purchase.groupby(['USER_ID_hash'],sort=False, as_index=False)['I_DATE'].max()

    updated_withdraw_users = pd.merge(users, last_purchase_withdraw, on='USER_ID_hash')

