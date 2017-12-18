# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import talib
import time as tm
import csv

# 静态分类查询
def static_label_name(label):
    if label == "pe":
        return pe_class()
    elif label == "outstanding":
        return outstanding_class()
    elif label == "pb":
        return pb_class()
    elif label == "npr":
        return npr_class()
    elif label == "industry":
        return industry_class()
    elif label == "concept":
        return concept_class()
    elif label == "area":
        return area_class()
    elif label == "company":
        return company_class()
    elif label == 'fund':
        return fund_class()


# 动态分类查询
def dynamic_query(label, time, code):
    if label == 'tor':
        return tor_class(time, code)
    elif label == 'op':
        return op_class(time, code)
    elif label == 'top':
        return top_class(time, code)


# 市盈率
def pe_class():
    pe = pd.read_csv('basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'S6'})['pe']
    pe_arr = np.asarray(pe)
    result = {}
    for i in range(len(pe_arr)):
        if pe_arr[i] > 30:
            result[pe.index[i]] = 0
        elif pe_arr[i] < 15:
            result[pe.index[i]] = 2
        else:
            result[pe.index[i]] = 1
    return result


# 市净率
def pb_class():
    pb = pd.read_csv('basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'S6'})['pb']
    pb_arr = np.asarray(pb)
    result = {}
    for i in range(len(pb_arr)):
        if pb_arr[i] > 7.5:
            result[pb.index[i]] = 0
        elif pb_arr[i] < 2:
            result[pb.index[i]] = 2
        else:
            result[pb.index[i]] = 1
    return result


# 净利润
def npr_class():
    npr = pd.read_csv('basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'S6'})['npr']
    npr_arr = np.asarray(npr)
    result = {}
    for i in range(len(npr_arr)):
        if npr_arr[i] > 0:
            result[npr.index[i]] = 0
        else:
            result[npr.index[i]] = 1
    return result


# 行业
def industry_class():
    industry = pd.read_excel('new_industry.xls')
    industry_title = {}
    for no, name in enumerate(np.unique(industry[u"二级行业名称"])):
        industry_title[name] = no
    result = {}
    for _, row in industry.iterrows():
        result[check(row[u'代码'])] = industry_title[row[u"二级行业名称"]]
    return result


# 概念
def concept_class():
    concept = pd.read_csv('concept.csv', encoding='utf-8', dtype={'code': 'S6'})
    concept_title = {}
    for no, name in enumerate(np.unique(concept['c_name'])):
        concept_title[name] = no
    result = {}
    for _, row in concept.iterrows():
        result[row['code']] = concept_title[row['c_name']]
    return result


# 地区
def area_class():
    area = pd.read_csv('area.csv', encoding='utf-8', dtype={'code': 'S6'})
    area_title = {}
    for no, name in enumerate(np.unique(area['area'])):
        area_title[name] = no
    result = {}
    for _, row in area.iterrows():
        result[row['code']] = area_title[row['area']]
    return result


# 公司
def company_class():
    company = pd.read_csv('company.csv', encoding='utf-8', dtype={'code': 'S6'})
    result = {}
    # for i in range(len(company)):
    #     result[company.ix[i, 0]] = company.ix[i, 1]
    for _, row in company.iterrows():
        result[check(row['code'])] = row['label']
    return result


# 市值
def outstanding_class():
    otsd = pd.read_csv('basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'S6'})['outstanding']
    otsd_arr = np.asarray(otsd)
    result = {}
    for i in range(len(otsd_arr)):
        if otsd_arr[i] > 1:
            result[otsd.index[i]] = 0
        elif otsd_arr[i] < 0.5:
            result[otsd.index[i]] = 2
        else:
            result[otsd.index[i]] = 1
    return result


# 基金重仓股
def fund_class():
    fund = pd.read_csv('./fund/20173.csv', index_col=1, dtype={'code': 'S6'})['ratio']
    fund_arr = np.asarray(fund)
    result = {}
    for i in range(len(fund_arr)):
        if fund_arr[i] > 0.2:
            result[fund.index[i]] = 1
        else:
            result[fund.index[i]] = 0
    return result


# 换手率
def tor_class(time, code):  # 未考虑到三天均不在的情况
    try:
        # 若无turnover
        tor = pd.read_csv('./hist/' + code + '.csv', index_col=0)['turnover']

        past_three = [(time - 3 * datetime.timedelta(days=1)), time - 2 * datetime.timedelta(days=1),
                      time - datetime.timedelta(days=1)]
        past_three = [day.strftime('%Y-%m-%d') for day in past_three]
        clean_tor = [x for x in tor[past_three] if not str(x) == 'nan']
        tor_value = np.asarray(clean_tor).mean()
        if tor_value < 3:
            return 0
        elif tor_value > 7:
            return 2
        else:
            return 1
    except IOError:
        return 0
    except KeyError:
        return 0


# 股价
def op_class(time, code):
    time_str = time.strftime('%Y-%m-%d')
    try:
        op = pd.read_csv('./hist/' + code + '.csv', index_col=0)['open']
        if time_str in op.index:
            if op[time_str] < 10:
                return 0
            elif op[time_str] > 60:
                return 2
            else:
                return 1
        else:  # 向前递归未休市
            if time > datetime.date(2017, 7, 1):
                return op_class(time - datetime.timedelta(days=1), code)
            else:
                return 0
    except IOError:
        return 0


# 热点
def top_class(time, code):
    try:
        top = pd.read_csv('./top/' + time.strftime('%Y-%m-%d').replace('-', '') + '.csv', index_col=0)['code']
        top = [check(stock) for stock in top]
        if code in top:
            return 1
        else:
            return 0
    except IOError:
        if time > datetime.date(2017, 7, 1):
            return top_class(time - datetime.timedelta(days=1), code)
        else:
            return 0


def check(stock):
    v = str(stock)
    for i in range(len(v), 6):
        v = '0' + v
    return v


def int2date(time):
    time = str(time)
    time = datetime.date(int(time[:4]), int(time[4:6]), int(time[6:]))
    return time


# 考虑到一个股属于多个标签值
def Q_pref(df, khh, real_time, label, label_num):
    df = df[df["custid"] == khh]
    pa = [0.0 for i in range(label_num)]
    pall = 0.0001
    if real_time == 'static':
        targets = static_label_name(label)
    for _, row in df.iterrows():
        if row["matchqty"] > 0.0 and row["stkeffect"] > 0:
            if real_time == 'static':
                try:
                    pa[targets[row["stkcode"]]] += row["matchamt"]
                except KeyError:
                    continue
            else:
                targets = dynamic_query(label, int2date(row["busi_date"]), check(row["stkcode"]))
                if not type(targets) == list:  # 预留多标签值
                    pa[targets] += row["matchamt"]
            pall += row["matchamt"]
    return np.asarray(pa) / pall


def R_pref(df, khh, real_time, label, label_num):
    df = df[df["custid"] == khh]
    qa = [0.0 for i in range(label_num)]
    qall = 0.0001
    stocks = np.unique(np.asarray(df["stkcode"]))
    if real_time == 'static':
        targets = static_label_name(label)
    for stock in stocks:
        temp = df[df["stkcode"] == stock].sort_values(by="busi_date")
        times = np.unique(np.asarray(temp["busi_date"]))
        times_date = [int2date(time) for time in times]
        inv = 0
        if real_time == 'static':
            na = [0.0 for i in range(label_num)]
            nall = 0.0
            for no, time in enumerate(times):
                if inv > 0:
                    for i, target in enumerate(targets):
                        na[i] += (check(stock) in target) * (times_date[no] - times_date[no - 1]).days * inv
                    nall += (times_date[no] - times_date[no - 1]).days * inv
                tempp = temp[temp["busi_date"] == time]
                for index, row in tempp.iterrows():
                    if row["matchqty"] > 0.0:
                        if row["stkeffect"] > 0:
                            inv += row["matchqty"]
                        else:
                            inv = max(inv - row["matchqty"], 0)
            qa = np.asarray(na) + np.asarray(qa)
            qall += nall
        elif real_time == 'daily':  # 根据时间对每天库存求和
            # 可与后一项合并
            for no in range(len(times) - 1):
                tempp = temp[temp["busi_date"] == times[no]]
                for index, row in tempp.iterrows():
                    if row["matchqty"] > 0.0:
                        if row["stkeffect"] > 0:
                            inv += row["matchqty"]
                        else:
                            inv = max(inv - row["matchqty"], 0)

                temp_time = times_date[no]
                while not temp_time == times_date[no + 1]:
                    targets = dynamic_query(label, temp_time, check(stock))
                    if not type(targets) == list:  # 预留多标签值
                        qa[targets] += inv
                    qall += inv
                    temp_time += datetime.timedelta(days=1)
    return np.asarray(qa) / qall


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


user = pd.read_csv("logasset_all_good.csv", dtype={'stkcode': 'S6'})[["busi_date", "custid", "stkeffect", "matchqty", "matchprice", "stkcode", "matchamt"]].drop_duplicates()
user_set = np.unique(user["custid"])
lent = len(user_set)
classes = {}
classes['pe'] = pe_class()
classes['outstanding'] = outstanding_class()
classes['pb'] = pb_class()
classes['npr'] = npr_class()
classes['industry'] = industry_class()
classes['concept'] = concept_class()
classes['area'] = area_class()
classes['company'] = company_class()
classes['fund'] = fund_class()

k_down = [0, 1, 3, 8, 14, 19, 20, 22, 24, 31, 32, 44, 49, 52, 59]
k_up = [5, 6, 9, 17, 23, 33, 36, 40, 42, 43, 45, 58]
other = [2, 4, 10, 12, 13, 25, 26, 27, 30, 47, 55, 60]
discard = [7, 11, 15, 16, 18, 21, 28, 29, 34, 35, 37, 38, 39, 41, 46, 48, 50, 51, 53, 54, 56, 57]
pattern_dict = {}
for i in k_down:
    pattern_dict[i] = 'down'
for i in k_up:
    pattern_dict[i] = 'up'
for i in other:
    pattern_dict[i] = 'other'
for i in discard:
    pattern_dict[i] = 'discard'
start = tm.time()
result_list = []
for index, khh in enumerate(user_set):
    print index, "/", lent
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
    temp_list.append(macd_pref(user, khh))
    temp_list.append(kdj_pref(user, khh))
    temp_list.append(pattern_pref(user, khh))       # 预先储存
    result_list.append(temp_list)
end = tm.time()

with open('all_good_users.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile)
    for result_user in result_list:
        row = [i for value in result_user for i in value]
        csvwriter.writerow(row)
print end - start
