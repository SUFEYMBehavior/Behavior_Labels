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

# 静态分类查询
def static_label_name(label):
    return {
        'pe': pe_class(),
        'outstanding': outstanding_class(),
        'pb': pb_class(),
        'npr': npr_class(),
        'industry': industry_class(),
        'concept': concept_class(),
        'area': area_class(),
    }.get(label)


# 动态分类查询
def dynamic_query(label, time, code):
    return {
        'tor': tor_class(time, code),
        'op': op_class(time, code),
        'top': top_class(time, code),
        'fund': fund_class(time, code),
    }.get(label)


# 市盈率
def pe_class():
    pe = pd.read_csv('basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'S6'})['pe']
    pe_arr = np.asarray(pe)
    result = [[] for i in range(3)]
    for i in range(len(pe_arr)):
        if pe_arr[i] > 30:
            result[0].append(pe.index[i].decode())
        elif pe_arr[i] < 15:
            result[2].append(pe.index[i].decode())
        else:
            result[1].append(pe.index[i].decode())
    return result


# 市净率
def pb_class():
    pb = pd.read_csv('basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'S6'})['pb']
    pb_arr = np.asarray(pb)
    result = [[] for i in range(3)]
    for i in range(len(pb_arr)):
        if pb_arr[i] > 7.5:
            result[0].append(pb.index[i].decode())
        elif pb_arr[i] < 2:
            result[2].append(pb.index[i].decode())
        else:
            result[1].append(pb.index[i].decode())
    return result


# 净利润
def npr_class():
    npr = pd.read_csv('basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'S6'})['npr']
    npr_arr = np.asarray(npr)
    result = [[] for i in range(2)]
    for i in range(len(npr_arr)):
        if npr_arr[i] > 0:
            result[0].append(npr.index[i].decode())
        else:
            result[1].append(npr.index[i].decode())
    return result


# 行业
def industry_class():
    industry_title = pd.read_csv('industry_title.csv', index_col=0, encoding='utf-8')
    industry = pd.read_csv('industry.csv', encoding='utf-8', dtype={'code': 'S6'})
    result = [[] for i in range(len(industry_title))]
    for index, row in industry.iterrows():
        result[industry_title.loc[row['c_name']][0]].append(row['code'].decode())
    return result


# 概念
def concept_class():
    concept_title = pd.read_csv('concept_title.csv', index_col=0, encoding='utf-8')
    concept = pd.read_csv('concept.csv', encoding='utf-8', dtype={'code': 'S6'})
    result = [[] for i in range(len(concept_title))]
    for index, row in concept.iterrows():
        result[concept_title.loc[row['c_name']][0]].append(row['code'].decode())
    return result


# 地区
def area_class():
    area_title = pd.read_csv('area_title.csv', index_col=0, encoding='utf-8')
    area = pd.read_csv('area.csv', encoding='utf-8', dtype={'code': 'S6'})
    result = [[] for i in range(len(area_title))]
    for index, row in area.iterrows():
        result[area_title.loc[row['area']][0]].append(row['code'].decode())
    return result


# 市值
def outstanding_class():
    otsd = pd.read_csv('basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'S6'})['outstanding']
    otsd_arr = np.asarray(otsd)
    result = [[] for i in range(3)]
    for i in range(len(otsd_arr)):
        if otsd_arr[i] > 1:
            result[0].append(otsd.index[i].decode())
        elif otsd_arr[i] < 0.5:
            result[2].append(otsd.index[i].decode())
        else:
            result[1].append(otsd.index[i].decode())
    return result


# 换手率
def tor_class(time, code):  # 未考虑到三天均不在的情况
    try:
        tor = pd.read_csv('./hist/' + code + '.csv', index_col=0)['turnover']
        # time = int2date(time)
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
            return op_class(time - datetime.timedelta(days=1), code)
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
        return top_class(time - datetime.timedelta(days=1), code)


# 基金重仓股
def fund_class(time, code):
    year = int(time.year)
    quarter = int((time.month - 1) / 3) + 1
    try:
        fund = pd.read_csv('./fund/' + str(year) + str(quarter) + '.csv', index_col=1, dtype={'code': 'S6'})['ratio']
    except IOError:
        fund = pd.read_csv('./fund/' + str(year) + str(quarter - 1) + '.csv', index_col=1, dtype={'code': 'S6'})['ratio']
    try:
        ratio = fund[code]
        if ratio > 0.2:
            return 1
        else:
            return 0
    except KeyError:
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


def quarter_series(start, end):
    series = [start]
    if start.year == end.year:
        for i in range(start.month + 1, end.month + 1):
            if i % 3 == 1:
                series.append(datetime.date(start.year, i, 1))
    else:
        for i in range(start.month + 1, 13):
            if i % 3 == 1:
                series.append(datetime.date(start.year, i, 1))
        for i in range(start.year + 1, end.year):
            for j in range(1, 12, 3):
                series.append(datetime.date(i, j, 1))
        for i in range(1, end.month + 1):
            if i % 3 == 1:
                series.append(datetime.date(end.year, i, 1))
    series.append(end)
    return series


# 考虑到一个股属于多个标签值
def Q_pref(df, khh, real_time, label, label_num):

    pa = [0.0 for i in range(label_num)]
    pall = 0.0001
    if real_time == 'static':
        targets = static_label_name(label)
    for index, row in df.iterrows():
        if float(row["matchqty"]) > 0.0 and row["stkeffect"] > 0:
            if real_time == 'static':
                for i, target in enumerate(targets):
                    pa[i] += (check(row["stkcode"]) in target) * float(row["matchprice"])
            else:
                targets = dynamic_query(label, int2date(row["busi_date"]), check(row["stkcode"]))
                if not type(targets) == list:  # 预留多标签值
                    pa[targets] += float(row["matchprice"])
            pall += float(row["matchprice"])
    return np.asarray(pa) / pall


def R_pref(df, khh, real_time, label, label_num):

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
                        na[i] += float((check(stock) in target) * (times_date[no] - times_date[no - 1]).days * inv)
                    nall += float((times_date[no] - times_date[no - 1]).days * inv)
                tempp = temp[temp["busi_date"] == time]
                for index, row in tempp.iterrows():
                    if float(row["matchqty"]) > 0.0:
                        if row["stkeffect"] > 0:
                            inv += float(float(row["matchqty"]))
                        else:
                            inv = max(inv - float(float(row["matchqty"])), 0)
            qa = np.asarray(na) + np.asarray(qa)
            qall += nall
        elif real_time == 'daily':  # 根据时间对每天库存求和
            # 可与后一项合并
            for no in range(len(times) - 1):
                tempp = temp[temp["busi_date"] == times[no]]
                for index, row in tempp.iterrows():
                    if float(row["matchqty"]) > 0.0:
                        if row["stkeffect"] > 0:
                            inv += float(row["matchqty"])
                        else:
                            inv = max(inv - float(row["matchqty"]), 0)

                temp_time = times_date[no]
                while not temp_time == times_date[no + 1]:
                    targets = dynamic_query(label, temp_time, check(stock))
                    if not type(targets) == list:  # 预留多标签值
                        qa[targets] += inv
                    qall += inv
                    temp_time += datetime.timedelta(days=1)
        elif real_time == 'quarterly':  # 判断季度开始日
            for no in range(len(times) - 1):
                tempp = temp[temp["busi_date"] == times[no]]
                for index, row in tempp.iterrows():
                    if float(row["matchqty"]) > 0.0:
                        if row["stkeffect"] > 0:
                            inv += float(row["matchqty"])
                        else:
                            inv = max(inv - float(row["matchqty"]), 0)

                start = times_date[no]
                end = times_date[no + 1]
                time_series = quarter_series(start, end)
                for index in range(len(time_series) - 1):
                    targets = dynamic_query('fund', time_series[index], check(stock))
                    if not type(targets) == list:  # 预留多标签值
                        qa[targets] += inv * (time_series[index + 1] - time_series[index]).days
                    qall += inv * (time_series[index + 1] - time_series[index]).days
    return np.asarray(qa) / qall


def invest_pref(df, khh, real_time, label, label_num):
    if np.sum(R_pref(df, khh, real_time, label, label_num)) > 0:
        value = 0.5 * Q_pref(df, khh, real_time, label, label_num) + 0.5 * R_pref(df, khh, real_time, label, label_num)
    else:
        value = Q_pref(df, khh, real_time, label, label_num)
    return {khh: value}


def operate_pref(df, khh, label):

    stocks = np.unique(np.asarray(df["stkcode"]))
    pa = 0
    pall = 0.0001
    for stock in stocks:
        temp = df[df["stkcode"] == stock]
        if label == 'MACD':
            pa += macd_op(temp, stock)[0]
            pall += macd_op(temp, stock)[1]
        elif label == 'KDJ':
            pa += kdj_op(temp, stock)[0]
            pall += kdj_op(temp, stock)[1]
    return {khh: pa / pall}


def macd_op(df, stock):
    pa_op = 0
    pall_op = 0
    try:
        hist_k = pd.read_csv('./hist/' + check(stock) + '.csv',index_col=0, usecols=[0, 3])
        hist_k["num"] = [(len(hist_k)-i-1) for i in range(len(hist_k))]
        close = list(hist_k['close'])
        close.reverse()
        _, _, hist = talib.MACD(np.asarray(close, dtype=np.double))
        for index, row in df.iterrows():
            if row["matchqty"] > 0.0:
                time = int2date(row["busi_date"]).strftime('%Y-%m-%d')
                try:
                    no = int(hist_k.ix[time][1])
                    if (hist[no - 1] > 0 and row["stkeffect"] > 0) or (hist[no - 1] < 0 and row["stkeffect"] < 0):
                        pa_op += 1
                    pall_op += 1
                except KeyError:
                    continue
        return [pa_op, pall_op]
    except IOError:
        return [pa_op, pall_op]


def kdj_op(df, stock):
    pa_op = 0
    pall_op = 0
    try:
        hist_k = pd.read_csv('./hist/' + check(stock) + '.csv', index_col=0, usecols=[0, 2, 3, 4])
        hist_k["num"] = [(len(hist_k) - i - 1) for i in range(len(hist_k))]
        high = list(hist_k['high'])
        high.reverse()
        low = list(hist_k['low'])
        low.reverse()
        close = list(hist_k['close'])
        close.reverse()
        K, D = talib.STOCH(np.asarray(high), np.asarray(low), np.asarray(close),
                           fastk_period=9)
        J = 3 * K - 2 * D
        for index, row in df.iterrows():
            if row["matchqty"] > 0.0:
                time = int2date(row["busi_date"]).strftime('%Y-%m-%d')
                try:
                    no = int(hist_k.ix[time][3])
                    if (J[no - 1] > 0 and row["stkeffect"] > 0) or (J[no - 1] < 0 and row["stkeffect"] < 0):
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
    return invest_pref(df, khh, 'quarterly', 'fund', 2)


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


def industry_pref(df, khh):
    return invest_pref(df, khh, 'static', 'industry', 49)


def concept_pref(df, khh):
    return invest_pref(df, khh, 'static', 'concept', 99)


def area_pref(df, khh):
    return invest_pref(df, khh, 'static', 'area', 32)

def sql_num(khh, database):
    sql =  ('select custid, stkcode from tcl_logasset where custid = %s'%khh)
    ms = MSSQL(host = "localhost", user = "SA", pwd = "!@Cxy7300", db = database)
    df = ms.ExecQuery(sql)
    return len(df)