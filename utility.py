import numpy as np
import datetime
import time
import math
import pandas as pd
#import talib
#import pymssql
from functools import wraps

def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
               (function.__name__, str(t1-t0))
               )
        return result
    return function_timer

class MSSQL:
    def __init__(self, host, user, pwd, db):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.db = db

    def __GetConnect(self):
        if not self.db:
            raise (NameError, "没有设置数据库信息")
        self.conn = pymssql.connect(host=self.host, user=self.user, password=self.pwd, database=self.db)
        cur = self.conn.cursor()
        if not cur:
            raise (NameError, "连接数据库失败")
        else:
            return cur

    @fn_timer
    def ExecQuery(self, sql):
        cur = self.__GetConnect()
        cur.execute(sql)
        resList = cur.fetchall()
        # 查询完毕后必须关闭连接
        self.conn.close()
        return resList

    def ExecNonQuery(self, sql):
        cur = self.__GetConnect()
        cur.execute(sql)
        self.conn.commit()
        self.conn.close()


def get_concept(self):
    concept = pd.DataFrame(ts.get_concept_classified())
    print(concept.shape)
    concept.to_csv('concept.csv')

def get_industry(self):
    industry = pd.DataFrame(ts.get_industry_classified())
    print(industry.shape)
    industry.to_csv('industry.csv')


def fund_perference(lst):
    # 防止越界所以设成1
    fund_pefer = 1
    fund_total = 1
    for _, _, cjje, flag, bs in lst:      # cjje为成交金额 flag是判断是否属于该类 bs代表买入卖出方向
        if flag and bs:
            fund_pefer += math.log(cjje)
        fund_total += math.log(cjje)
    return fund_pefer/fund_total

def hold_perference(lst):
    l0 = lst[0]                 # 输入的数据是经过时间排序的，所以取第一个数据的时间作为起始时间
    start_time = l0[0]
    holdnum_perfer = 0
    holdnum_total = 0
    hold_perfer = 1
    hold_total = 1
    end_time = datetime.datetime(year=2017, month=10, day=26)
    for time, cjsl, _, flag, bs in lst:
        #计算两次操作相邻的时间
        delta = time-start_time
        hold_perfer += delta.days * holdnum_perfer
        hold_total += delta.days * holdnum_total
        if bs:
            holdnum_total += cjsl
            if flag:
                holdnum_perfer += cjsl
        else:
            holdnum_total = max(0, holdnum_total - cjsl)
            if flag:
                holdnum_perfer = max(0, holdnum_perfer - cjsl)
        start_time = time

    delta = end_time - start_time
    hold_perfer += delta.days * holdnum_perfer
    hold_total += delta.days * holdnum_total
    return math.log(hold_perfer/hold_total+1)


def calculate(tt, label):
    user_perfer = {}
    for _, row in tt.iterrows():
        khh = row['custid']
        if not khh in user_perfer.keys():
            user_perfer[khh] = []
        cjsl = row['matchqty']
        bs = row['stkeffect'] > 0
        if cjsl > 0:
            cjje = row['matchamt']
            sbrq = datetime.datetime.strptime(str(row["orderdate"]), '%Y%m%d')
            flag = row[label]
            user_perfer[khh].append([sbrq, cjsl, cjje, flag, bs])

    dic = {}
    dic['khh'] = []
    dic['Q'] = []
    dic['R'] = []
    f_p = []
    h_p = []
    for khh in user_perfer:
        lst = user_perfer[khh]
        dic['khh'].append(khh)
        if len(lst) > 0:
            fp = fund_perference(lst)
            hp = hold_perference(lst)
            dic['Q'].append(fp)
            dic['R'].append(hp)
        else:
            dic['Q'].append(0)
            dic['R'].append(0)

    f_perf = standardize(f_p)
    h_perf = standardize(h_p)
    return pd.DataFrame(dic)


def standardize(data):
    s = np.array(data)
    s = (s-s.mean())/s.std()
    return s

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
    pe = pd.read_csv('datas/basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'S6'})['pe']
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
    pb = pd.read_csv('datas/basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'S6'})['pb']
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
    npr = pd.read_csv('datas/basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'S6'})['npr']
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
    industry = pd.read_excel('datas/new_industry.xls')
    industry_title = {}
    for no, name in enumerate(np.unique(industry[u"二级行业名称"])):
        industry_title[name] = no
    result = {}
    for _, row in industry.iterrows():
        result[check(row[u'代码'])] = industry_title[row[u"二级行业名称"]]
    return result


# 概念
def concept_class():
    concept = pd.read_csv('datas/concept.csv', encoding='utf-8', dtype={'code': 'S6'})
    concept_title = {}
    for no, name in enumerate(np.unique(concept['c_name'])):
        concept_title[name] = no
    result = {}
    for _, row in concept.iterrows():
        result[row['code']] = concept_title[row['c_name']]
    return result


# 地区
def area_class():
    area = pd.read_csv('datas/area.csv', encoding='utf-8', dtype={'code': 'S6'})
    area_title = {}
    for no, name in enumerate(np.unique(area['area'])):
        area_title[name] = no
    result = {}
    for _, row in area.iterrows():
        result[row['code']] = area_title[row['area']]
    return result


# 公司
def company_class():
    company = pd.read_csv('datas/company.csv', encoding='utf-8', dtype={'code': 'S6'})
    result = {}
    # for i in range(len(company)):
    #     result[company.ix[i, 0]] = company.ix[i, 1]
    for _, row in company.iterrows():
        result[check(row['code'])] = row['label']
    return result


# 市值
def outstanding_class():
    otsd = pd.read_csv('datas/basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'S6'})['outstanding']
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
    fund = pd.read_csv('datas/fund/20173.csv', index_col=1, dtype={'code': 'S6'})['ratio']
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
        top = pd.read_csv('datas/top/' + time.strftime('%Y-%m-%d').replace('-', '') + '.csv', index_col=0)['code']
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