# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import time as tm
import csv
from utility import *
# 静态分类查询



def invest_pref(df, khh, real_time, label, label_num):
    # R_value = R_pref(df, khh, real_time, label, label_num)
    # if np.sum(R_value) > 0:
    #     value = 0.5 * Q_pref(df, khh, real_time, label, label_num) + 0.5 * R_value
    # else:
    #     value = Q_pref(df, khh, real_time, label, label_num)
    # return {khh: value}
    return Q_pref(df, khh, real_time, label, label_num)


def operate_pref(df, khh, label):
    df = df[df["custid"] == khh]
    stocks = np.unique(np.asarray(df["stkcode"]))
    pa = np.asarray([0, 0])
    pall = 0.0001
    for stock in stocks:
        temp = df[df["stkcode"] == stock]
        if label == 'MACD':
            pa += macd_op(temp, stock)[0]
            pall += macd_op(temp, stock)[1]
        elif label == 'KDJ':
            pa += kdj_op(temp, stock)[0]
            pall += kdj_op(temp, stock)[1]
    return pa / pall


def macd_op(df, stock):
    pa_op = [0, 0]
    pall_op = 0
    try:
        hist_k = pd.read_csv('./hist/' + check(stock) + '.csv', index_col=0, usecols=[0, 3])
        num = dict(zip(list(hist_k.index), [(len(hist_k) - i - 1) for i in range(len(hist_k))]))
        close = list(hist_k['close'])
        close.reverse()
        _, _, hist = talib.MACD(np.asarray(close, dtype=np.double))
        for _, row in df.iterrows():
            if row["matchqty"] > 0.0:
                time = int2date(row["busi_date"]).strftime('%Y-%m-%d')
                try:
                    no = num[time]
                    if hist[no - 2] < 0 and hist[no - 1] > 0 and row["stkeffect"] > 0:
                        pa_op[0] += 1
                    elif hist[no - 2] > 0 and hist[no - 1] < 0 and row["stkeffect"] < 0:
                        pa_op[1] += 1
                    pall_op += 1
                except KeyError:
                    continue
        return [pa_op, pall_op]
    except IOError:
        return [pa_op, pall_op]


def kdj_op(df, stock):
    pa_op = [0, 0]
    pall_op = 0
    try:
        hist_k = pd.read_csv('./hist/' + check(stock) + '.csv', index_col=0, usecols=[0, 2, 3, 4])
        num = dict(zip(list(hist_k.index), [(len(hist_k) - i - 1) for i in range(len(hist_k))]))
        high = list(hist_k['high'])
        high.reverse()
        low = list(hist_k['low'])
        low.reverse()
        close = list(hist_k['close'])
        close.reverse()
        K, D = talib.STOCH(np.asarray(high), np.asarray(low), np.asarray(close),
                           fastk_period=9)
        J = 3 * K - 2 * D
        for _, row in df.iterrows():
            if row["matchqty"] > 0.0:
                time = int2date(row["busi_date"]).strftime('%Y-%m-%d')
                try:
                    no = num[time]
                    if J[no - 2] < 0 and J[no - 1] > 0 and row["stkeffect"] > 0:
                        pa_op[0] += 1
                    elif J[no - 2] > 0 and J[no - 1] < 0 and row["stkeffect"] < 0:
                        pa_op[1] += 1
                    pall_op += 1
                except KeyError:
                    continue
        return [pa_op, pall_op]
    except IOError:
        return [pa_op, pall_op]


def pattern_op(df, stock, name, type):
    pa_op = 0
    pall_op = 0
    try:
        hist_k = pd.read_csv('./hist/' + check(stock) + '.csv',index_col=0, usecols=[0, 1, 2, 3, 4, 5])
        num = dict(zip(list(hist_k.index), [(len(hist_k) - i - 1) for i in range(len(hist_k))]))
        open = list(hist_k['open'])
        open.reverse()
        high = list(hist_k['high'])
        high.reverse()
        low = list(hist_k['low'])
        low.reverse()
        close = list(hist_k['close'])
        close.reverse()
        volume = list(hist_k['volume'])
        volume.reverse()
        chance = getattr(talib, name)(np.asarray(open), np.asarray(high), np.asarray(low), np.asarray(close))
        for index, row in df.iterrows():
            if row["matchqty"] > 0.0:
                time = int2date(row["busi_date"]).strftime('%Y-%m-%d')
                try:
                    no = num[time]
                    if type == 'other':
                        if (chance[no - 1] > 0 and row["stkeffect"] > 0) or (chance[no - 1] < 0 and row["stkeffect"] < 0):
                            pa_op += 1
                    elif type == 'up':
                        if chance[no - 1] > 0 and row["stkeffect"] > 0:
                            pa_op += 1
                    elif type == 'down':
                        if chance[no - 1] > 0 and row["stkeffect"] < 0:
                            pa_op += 1
                    pall_op += 1
                except KeyError:
                    continue
        return [pa_op, pall_op]
    except IOError:
        return [pa_op, pall_op]


def tor_pref(df, khh):
    return invest_pref(df, khh, 'daily', 'tor', 3)


def fund_pref(df, khh):
    return invest_pref(df, khh, 'static', 'fund', 2)


def otstd_pref(df, khh):
    return invest_pref(df, khh, 'static', 'outstanding', 3)


def op_pref(df, khh):
    return invest_pref(df, khh, 'daily', 'op', 3)


def pe_pref(df, khh):
    return invest_pref(df, khh, 'static', 'pe', 3)


def pb_pref(df, khh):
    return invest_pref(df, khh, 'static', 'pb', 3)


def npr_pref(df, khh):
    return invest_pref(df, khh, 'static', 'npr', 2)


def top_pref(df, khh):
    return invest_pref(df, khh, 'daily', 'top', 2)


def macd_pref(df, khh):
    return operate_pref(df, khh, 'MACD')


def kdj_pref(df, khh):
    return operate_pref(df, khh, 'KDJ')


def pattern_pref(df, khh):
    df = df[df["custid"] == khh]
    stocks = np.unique(np.asarray(df["stkcode"]))
    patterns = talib.get_function_groups()['Pattern Recognition']
    pref_result = []
    for pattern_index, pattern in enumerate(patterns):
        op = [0, 0.0001]
        pattern_type = pattern_dict[pattern_index]
        if not pattern_type == 'discard':
            for stock in stocks:
                temp = df[df["stkcode"] == stock]
                pat_op = pattern_op(temp, stock, pattern, pattern_type)
                op[0] += pat_op[0]
                op[1] += pat_op[1]
            pref_result.append(op[0] / op[1])
    return pref_result


def industry_pref(df, khh):
    return invest_pref(df, khh, 'static', 'industry', 75)


def concept_pref(df, khh):
    return invest_pref(df, khh, 'static', 'concept', 99)


def area_pref(df, khh):
    return invest_pref(df, khh, 'static', 'area', 32)


def company_pref(df, khh):
    return invest_pref(df, khh, 'static', 'company', 4)


def get_labels(users, custid):
    logasset = users.get_logdata(custid)
    user = logasset
    khh = custid
    dic = {}
    temp_list = []
    temp_list.append([khh])
    temp_list.append(pe_pref(user, khh))
    temp_list.append(otstd_pref(user, khh))
    temp_list.append(pb_pref(user, khh))
    temp_list.append(npr_pref(user, khh))
    temp_list.append(industry_pref(user, khh))
    temp_list.append(concept_pref(user, khh))
    temp_list.append(area_pref(user, khh))
    temp_list.append(company_pref(user, khh))
    temp_list.append(fund_pref(user, khh))
    temp_list.append(tor_pref(user, khh))
    temp_list.append(op_pref(user, khh))
    temp_list.append(top_pref(user, khh))
    #temp_list.append(macd_pref(user, khh))
    #temp_list.append(kdj_pref(user, khh))
    #temp_list.append(pattern_pref(user, khh))

    return temp_list

'''
user = pd.read_csv("logasset_all_good.csv", dtype={'stkcode': 'S6'})[["busi_date", "custid", "stkeffect", "matchqty", "matchprice", "stkcode", "matchamt"]].drop_duplicates()
user_set = np.unique(user["custid"])
lent = len(user_set)


start = tm.time()
result_list = []
for index, khh in enumerate(user_set):
    print(index, "/", lent)
          # 预先储存
    result_list.append(temp_list)
end = tm.time()

with open('all_good_users.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile)
    for result_user in result_list:
        row = [i for value in result_user for i in value]
        csvwriter.writerow(row)
print(end - start)
'''
