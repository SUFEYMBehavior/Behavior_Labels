
import numpy as np
import datetime
import math
import pandas as pd
import talib
buyset = set([4,9,10,14,23,24,29,41,42,44,50,51,49,1,91,89,115,124,57,59,55,78,26,28])
sellset = set([3,5,7,11,12,30,43,46,52,76,77,56,79,2,58,60,92,90,113,56,79,25,27])

def fund_perference(lst):
    #防止越界所以设成1
    fund_pefer = 1
    fund_total = 1
    for _, _, cjje, flag, bs in lst:      #cjje为成交金额 flag是判断是否属于该类 bs代表买入卖出方向
        if flag and bs:
            fund_pefer += math.log(cjje)
        fund_total += math.log(cjje)
    return fund_pefer/fund_total

def hold_perference(lst):
    l0 = lst[0]                 #输入的数据是经过时间排序的，所以取第一个数据的时间作为起始时间
    start_time = l0[0]
    holdnum_perfer = 0
    holdnum_total = 0
    hold_perfer = 1
    hold_total = 1
    end_time = datetime.datetime(year=2016, month=11, day=30)
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
            holdnum_total -= cjsl
            if flag:
                holdnum_perfer -= cjsl
        start_time = time

    delta = end_time - start_time
    hold_perfer += delta.days * holdnum_perfer
    hold_total += delta.days * holdnum_total
    if hold_total*hold_perfer > 0:
        return math.log(hold_perfer/hold_total+1)
    else:
        return math.log((hold_perfer - hold_total)/hold_perfer+1)

def calculate(tt, label):
    user_perfer = {}
    for _, row in tt.iterrows():
        khh = row['KHH']
        if not khh in user_perfer.keys():
            user_perfer[khh] = []
        cjsl = row['CJSL']
        wtlb = row['WTLB']
        if wtlb in buyset:
            bs = True
        elif wtlb in sellset:
            bs = False
        else:
            break
        if cjsl > 0:
            cjje = row['CJJE']
            sbrq = datetime.datetime.strptime(str(row["WTRQ"]), '%Y%m%d')
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

    f_perf = distribution(f_p)
    h_perf = distribution(h_p)
    return pd.DataFrame(dic)

def distribution(data):
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
    pe = pd.read_csv('basic.csv', index_col=0, dtype={'code':'S6'})['pe']
    pe_arr = np.asarray(pe)
    result = [[] for i in range(3)]
    for i in range(len(pe_arr)):
        if pe_arr[i] > 30:
            result[0].append(pe.index[i])
        elif pe_arr[i] < 15:
            result[2].append(pe.index[i])
        else:
            result[1].append(pe.index[i])
    return result
# 市净率
def pb_class():
    pb = pd.read_csv('basic.csv', index_col=0, dtype={'code': 'S6'})['pb']
    pb_arr = np.asarray(pb)
    result = [[] for i in range(3)]
    for i in range(len(pb_arr)):
        if pb_arr[i] > 7.5:
            result[0].append(pb.index[i])
        elif pb_arr[i] < 2:
            result[2].append(pb.index[i])
        else:
            result[1].append(pb.index[i])
    return result
# 净利润
def npr_class():
    npr = pd.read_csv('basic.csv', index_col=0, dtype={'code':'S6'})['npr']
    npr_arr = np.asarray(npr)
    result = [[] for i in range(2)]
    for i in range(len(npr_arr)):
        if npr_arr[i] > 0:
            result[0].append(npr.index[i])
        else:
            result[1].append(npr.index[i])
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
# 市值
def outstanding_class():
    otsd = pd.read_csv('basic.csv', index_col=0, dtype={'code':'S6'})['outstanding']
    otsd_arr = np.asarray(otsd)
    result = [[] for i in range(3)]
    for i in range(len(otsd_arr)):
        if otsd_arr[i] > 1:
            result[0].append(otsd.index[i])
        elif otsd_arr[i] < 0.5:
            result[2].append(otsd.index[i])
        else:
            result[1].append(otsd.index[i])
    return result
# 换手率
def tor_class(time, code):  # 未考虑到三天均不在的情况
    codes = np.loadtxt('./hist/codes.txt', dtype=str)
    if code in codes:
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
    else:
        return 0
# 股价
def op_class(time, code):
    time_str = time.strftime('%Y-%m-%d')
    codes = np.loadtxt('./hist/codes.txt', dtype=str)
    if code in codes:
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
    else:
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
    quarter = int((time.month - 1) / 3 + 1)
    fund = pd.read_csv('./fund/' + str(year) + str(quarter) + '.csv', index_col=1, dtype={'code': 'S6'})['ratio']
    try:
        ratio = fund[code]
        if ratio > 0.2:
            return 1
        else:
            return 0
    except KeyError:
        return 0

def is_buy(code):
    buy = [4, 9, 10, 14, 23, 24, 29, 41, 42, 44, 50, 51, 49, 1, 91, 89]
    if code in buy:
        return True
    else:
        return False

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
    df['buy'] = df['WTLB'].isin(buyset)
    df = df[df[u"KHH"] == khh]
    pa = [0.0 for i in range(label_num)]
    pall = 0.0001
    if real_time == 'static':
        targets = static_label_name(label)
    for index, row in df.iterrows():
        if row[u"CJSL"] > 0.0 and row['buy']:
            if real_time == 'static':
                for i, target in enumerate(targets):
                    pa[i] += (check(row[u"ZQDM"]) in target) * row[u"CJJE"]
            else:
                targets = dynamic_query(label, int2date(row[u"WTRQ"]), check(row[u"ZQDM"]))
                if not type(targets) == list:  # 预留多标签值
                    pa[targets] += row[u"CJJE"]
            pall += row[u"CJJE"]
    return np.asarray(pa) / pall

def R_pref(df, khh, real_time, label, label_num):
    df['buy'] = df['WTLB'].isin(buyset)
    df = df[df[u"KHH"] == khh]
    qa = [0.0 for i in range(label_num)]
    qall = 0.0001
    stocks = np.unique(np.asarray(df[u"ZQDM"]))
    if real_time == 'static':
        targets = static_label_name(label)
    for stock in stocks:
        temp = df[df[u"ZQDM"] == stock].sort_values(by=u"WTRQ")
        times = np.unique(np.asarray(temp[u"WTRQ"]))
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
                tempp = temp[temp[u"WTRQ"] == time]
                for index, row in tempp.iterrows():
                    if row[u"CJSL"] > 0.0:
                        if row['buy']:
                            inv += row[u"CJSL"]
                        else:
                            inv = max(inv - row[u"CJSL"], 0)
            qa = np.asarray(na) + np.asarray(qa)
            qall += nall
        elif real_time == 'daily':  # 根据时间对每天库存求和
            # 可与后一项合并
            for no in range(len(times) - 1):
                tempp = temp[temp[u"WTRQ"] == times[no]]
                for index, row in tempp.iterrows():
                    if row[u"CJSL"] > 0.0:
                        if row['buy']:
                            inv += row[u"CJSL"]
                        else:
                            inv = max(inv - row[u"CJSL"], 0)

                temp_time = times_date[no]
                while not temp_time == times_date[no + 1]:
                    targets = dynamic_query(label, temp_time, check(stock))
                    if not type(targets) == list:  # 预留多标签值
                        qa[targets] += inv
                    qall += inv
                    temp_time += datetime.timedelta(days=1)
        elif real_time == 'quarterly':  # 判断季度开始日
            for no in range(len(times) - 1):
                tempp = temp[temp[u"WTRQ"] == times[no]]
                for index, row in tempp.iterrows():
                    if row[u"CJSL"] > 0.0:
                        if row['buy']:
                            inv += row[u"CJSL"]
                        else:
                            inv = max(inv - row[u"CJSL"], 0)

                start = times_date[no]
                end = times_date[no + 1]
                time_series = quarter_series(start, end)
                for index in range(len(time_series) - 1):
                    targets = dynamic_query('fund', time_series[index], check(stock))
                    if not type(targets) == list:  # 预留多标签值
                        qa[targets] += inv * (time_series[index + 1] - time_series[index]).days
                    qall += inv * (time_series[index + 1] - time_series[index]).days
    return np.asarray(qa) / qall

def invest_pref(file, khh, real_time, label, label_num):
    df = pd.read_excel(file)
    value = Q_pref(df, khh, real_time, label, label_num) + R_pref(df, khh, real_time, label, label_num)
    return {khh: value}

def operate_pref(file, khh, label):
    df = pd.read_excel(file)
    df['buy'] = df['WTLB'].isin(buyset)
    #print(df['buy'])
    df = df[df[u"KHH"] == khh]
    stocks = np.unique(np.asarray(df[u"ZQDM"]))
    pa = 0
    pall = 0.0001
    for stock in stocks:
        temp = df[df[u"ZQDM"] == stock]
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
        hist_k = pd.read_csv('./k/' + check(stock) + '.csv', index_col=1, usecols=[0, 1, 3])
        _, _, hist = talib.MACD(np.asarray(hist_k['close'], dtype=np.double))
        for index, row in df.iterrows():
            if row[u"CJSL"] > 0.0:
                time = int2date(row[u"WTRQ"]).strftime('%Y-%m-%d')
                try:
                    no = int(hist_k.ix[time][0])
                    if (hist[no - 1] > 0 and row['buy']) or (hist[no - 1] < 0 and not row['buy']):
                        pa_op += 1
                    pall_op += 1
                except KeyError:
                    continue
        return [pa_op, pall_op]
    except IOError:
        return[pa_op, pall_op]

def kdj_op(df, stock):
    pa_op = 0
    pall_op = 0
    try:
        hist_k = pd.read_csv('./k/' + check(stock) + '.csv', index_col=1, usecols=[0, 1, 3, 4, 5])
        K, D = talib.STOCH(np.asarray(hist_k['high']), np.asarray(hist_k['low']), np.asarray(hist_k['close']),
                           fastk_period=9)
        J = 3 * K - 2 * D
        for index, row in df.iterrows():
            if row[u"CJSL"] > 0.0:
                time = int2date(row[u"WTRQ"]).strftime('%Y-%m-%d')
                try:
                    no = int(hist_k.ix[time][0])
                    if (J[no - 1] > 0 and row['buy']) or (J[no - 1] < 0 and not row['buy']):
                        pa_op += 1
                    pall_op += 1
                except KeyError:
                    continue
        return [pa_op, pall_op]
    except IOError:
        return[pa_op, pall_op]

dic = {'异常波动股偏好': abn['Q']*users.Q + abn['R']*users.R,
            '高转送股偏好': shr['Q']*users.Q + shr['R']*users.R,
               '持仓浮盈': users.holding_float(custid),
               '持仓集中度（个股）': users.hold_var(custid),
               '持仓集中度（概念）': users.hold_concept_var(custid),
               '涨跌停操作偏好': users.limit_perference(custid),
               '客户号': custid
               }

self.dic_keys = [
            '客户号',
            '行业偏好',
            '概念偏好',
            # '异常波动股偏好': abn['Q']*self.Q + abn['R']*self.R,
            # '高转送股偏好': shr['Q']*self.Q + shr['R']*self.R,
            # '活跃股偏好': hyg,
            '持仓概念偏好',
            '持仓浮盈浮亏',
            '持股周期',
            '持仓偏好',
            '涨跌停操作偏好',
            # '委托方式偏好': wtp['khh'],
            '止盈偏好', '止损偏好',
            'K线图',
            '日K买入', '周K买入', '月K买入',
            '日K卖出', '周K卖出', '月K卖出',
            '波段操作偏好',
            # '委托渠道偏好': [wtq[1], wtq[2], wtq[4], wtq[8], wtq[16], wtq[32], wtq[64], wtq[128]],
            # '交易时间偏好': jys.iloc[0],
            '''
            '打新股情况',
            # '动量策略偏好': dls.iloc[0],
            #'分时图',
            #'高点买入', '低点买入', '上行买入', '下跌买入',
            #'高点卖出', '低点卖出', '上行卖出', '下跌卖出',
            '新股偏好',
            '可转债偏好',
            '次新股偏好',
            'ST股偏好',
            # 'B股偏好': (bgp['FB'], bgp['B']),
            # '偏好市价委托': phw[0],
            # '偏好限价委托': phw[1],
            '偏好低换手率',
            '偏好中换手率',
            '偏好高换手率',
            '基金重仓股偏好',
            '偏好高市值',
            '偏好中市值',
            '偏好低市值',
            '偏好低价股',
            '偏好中价股',
            '偏好高价股',
            '偏好低市盈率',
            '偏好中市盈率',
            '偏好高市盈率',
            '偏好高市净率',
            '偏好中市净率',
            '偏好低市净率',
            '偏好正利润',
            '偏好负利润',
            '热点偏好',
            '非热点偏好',
            'MACD',
            'KDJ',
            # '行业偏好': [id_tlt.iloc[i, 0] for i in list(np.nonzero(idp[khh]))],
            # '概念偏好': [cp_tlt.iloc[i, 0] for i in list(np.nonzero(ccp[khh]))]
            '''
        ]