from cvxopt.modeling import op

__author__ = 'Dworkin'

import Settings
import scipy.sparse as sp
import pandas as pd
from Settings import data_dir_path
import numpy as np
import datetime
import pickle
import io


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

def split_dataframe (df, split_step):
    """

    :param df:
    :param split_step:
    :return:
    """
    row_count = len(df.index)
    array_of_df = []

    if row_count <= split_step:
        array_of_df.append(df)
    else:
        start_indx = 0
        while start_indx < row_count:
            end_indx = start_indx + split_step if (start_indx+split_step) <= row_count else row_count
            array_of_df.append(df.iloc[start_indx:end_indx, :])
            start_indx += split_step

    return array_of_df

def prepare_train_matrix(users_df, pref_df, coupon_desc_df, view_log_df):
    """

    :param users_df:
    :param pref_df:
    :param coupon_desc_df:
    :param view_log_df:
    :return:
    """
    user_hash_vect = get_feature_vectorizer(users_df, ['USER_ID_hash'])
    sex_vect = get_feature_vectorizer(users_df, ['SEX_ID'])
    prefect_vect = get_feature_vectorizer(pref_df, ['PREF_NAME'])
    capsule_text_vect = get_feature_vectorizer(coupon_desc_df, ['CAPSULE_TEXT'])

    array_of_view_log_df = split_dataframe(view_log_df, 100000)
    rez_coo_matrix = None

    for part_df in array_of_view_log_df:
        merge_view_and_coupons = pd.merge(part_df, coupon_desc_df, how='left',
                                          left_on='VIEW_COUPON_ID_hash', right_on='COUPON_ID_hash')
        merge_total = pd.merge(merge_view_and_coupons, users, how='left', on='USER_ID_hash')

        users_hash = [{'USER_ID_hash': x} for x in merge_total['USER_ID_hash']]
        users_sex = [{'SEX_ID': x} for x in merge_total['SEX_ID']]
        users_age = [[x] for x in merge_total['AGE']]
        users_pref = [{'PREF_NAME': x} for x in merge_total['PREF_NAME']]

        coupon_capsule_text = [{'CAPSULE_TEXT': x} for x in merge_total['CAPSULE_TEXT']]
        coupon_price_rate = [[x] for x in merge_total['PRICE_RATE']]
        coupon_discount_price = [[x] for x in merge_total['DISCOUNT_PRICE']]
        coupon_pref = [{'PREF_NAME': x} for x in merge_total['ken_name']]

        coupon_purchase = [[x] for x in merge_total['PURCHASE_FLG']]

        temp_coo_matrix = sp.hstack([
                user_hash_vect.transform(users_hash),
                sex_vect.transform(users_sex),
                prefect_vect.transform(users_pref),
                users_age,
                capsule_text_vect.transform(coupon_capsule_text),
                coupon_price_rate,
                coupon_discount_price,
                prefect_vect.transform(coupon_pref)
            ])

        if rez_coo_matrix is None:
            rez_coo_matrix = temp_coo_matrix
        else:
            rez_coo_matrix = sp.vstack([rez_coo_matrix, temp_coo_matrix])

    return rez_coo_matrix

def prepare_view_log_df (view_log):

    view_log = pd.DataFrame(view_log)
    #TODO make through normal datetime.to_date()
    view_log['I_DATE'] = view_log['I_DATE'].map(lambda x: x[0:10])
    view_log = view_log.drop_duplicates(['PURCHASE_FLG', 'I_DATE', 'VIEW_COUPON_ID_hash', 'USER_ID_hash'])
    return view_log




if __name__ == '__main__':

    #dirty_trick()

    #train data
    users = pd.DataFrame.from_csv(data_dir_path+Settings.user_list, index_col=False)
    coupons = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_list_train, index_col=False)
    view_log = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_visit_train, index_col=False)
    purchase_log = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_detail_train, index_col=False)
    coupon_area = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_area_train, index_col=False)
    prefect = pd.DataFrame.from_csv(data_dir_path+Settings.prefecture_locations, index_col=False)
    prefect.columns = ['PREF_NAME', 'PREFECTUAL_OFFICE', 'LATITUDE', 'LONGITUDE']
    #test data
    coupons_test = pd.DataFrame.from_csv(data_dir_path+Settings.coupon_list_test, index_col=False)

    view_log = prepare_view_log_df(view_log)
    rez_coo_matrix = \
        prepare_train_matrix(users_df=users, pref_df=prefect, coupon_desc_df=coupons, view_log_df=view_log)

    print rez_coo_matrix.shape
    print 'Matrix DONE! Trying to save'

    shape_w = io.open('../total_matrix_shape.pickle', 'wb')
    pickle.dump(rez_coo_matrix.shape, shape_w)
    shape_w.close()
    rez_coo_matrix.data.tofile('../total_matrix_data')
    rez_coo_matrix.col.tofile('../total_matrix_col')
    rez_coo_matrix.row.tofile('../total_matrix_row')

    """
    withdraw_purchase = view_log[
        (view_log.USER_ID_hash.isin(users[users.WITHDRAW_DATE.isnull()==False].USER_ID_hash))
        & (view_log.PURCHASE_FLG==1)]
    last_purchase_withdraw = withdraw_purchase.groupby(['USER_ID_hash'],sort=False, as_index=False)['I_DATE'].max()

    updated_withdraw_users = pd.merge(users, last_purchase_withdraw, on='USER_ID_hash')
    """
    pass


