# -*- coding: utf-8 -*-
__author__ = 'Dworkin'

import pandas as pd

from Settings import data_dir_path, capsule_text_file, coupon_list_test_before_transl, coupon_list_train_before_transl, \
    coupon_list_test, coupon_list_train


if __name__ == '__main__':


    # read the file and parse the first and only sheet (need python xlrd module)
    f = pd.ExcelFile(capsule_text_file)
    all = f.parse(parse_cols=[2, 3, 6, 7], skiprows=4, header=1)

    # data comes in two columns, produce a single lookup table from that
    first_col = all[['CAPSULE_TEXT', 'English Translation']]
    second_col = all[['CAPSULE_TEXT.1', 'English Translation.1']].dropna()
    second_col.columns = ['CAPSULE_TEXT', 'English Translation']
    all = first_col.append(second_col).drop_duplicates('CAPSULE_TEXT')
    translation_map = dict(zip(all['CAPSULE_TEXT'], all['English Translation']))

    # write new files with substituted names
    for f in [data_dir_path+coupon_list_train_before_transl, data_dir_path+coupon_list_test_before_transl]:
        infile = pd.read_csv(f)
        infile['CAPSULE_TEXT'] = infile['CAPSULE_TEXT'].apply(lambda x: translation_map[unicode(x, "utf-8")])
        infile['GENRE_NAME'] = infile['GENRE_NAME'].apply(lambda x: translation_map[unicode(x, "utf-8")])
        infile.to_csv(f.replace(".csv", "_translated.csv"), index=False)