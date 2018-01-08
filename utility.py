import numpy as np
import datetime
import time
import math
import pandas as pd
#import talib
import pymssql
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
    s = (s-np.nanmean(s))/np.nanstd(s)
    return s

def static_label_name(label):
    return classes[label]


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
    pe = pd.read_csv('datas/basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'str'})['pe']
    pe_arr = np.asarray(pe)
    result = {}
    for i in range(len(pe_arr)):
        if pe_arr[i] > 30:
            result[check(pe.index[i])] = 0
        elif pe_arr[i] < 15:
            result[check(pe.index[i])] = 2
        else:
            result[check(pe.index[i])] = 1
    return result


# 市净率
def pb_class():
    pb = pd.read_csv('datas/basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'str'})['pb']
    pb_arr = np.asarray(pb)
    result = {}
    for i in range(len(pb_arr)):
        if pb_arr[i] > 7.5:
            result[check(pb.index[i])] = 0
        elif pb_arr[i] < 2:
            result[check(pb.index[i])] = 2
        else:
            result[check(pb.index[i])] = 1
    return result


# 净利润
def npr_class():
    npr = pd.read_csv('datas/basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'str'})['npr']
    npr_arr = np.asarray(npr)
    result = {}
    for i in range(len(npr_arr)):
        if npr_arr[i] > 0:
            result[check(npr.index[i])] = 0
        else:
            result[check(npr.index[i])] = 1
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
    concept = pd.read_csv('datas/concept.csv', encoding='utf-8', dtype={'code': 'str'})
    concept_title = {}
    for no, name in enumerate(np.unique(concept['c_name'])):
        concept_title[name] = no
    result = {}
    for _, row in concept.iterrows():
        result[check(row['code'])] = concept_title[row['c_name']]
    return result


# 地区
def area_class():
    area = pd.read_csv('datas/area.csv', encoding='utf-8', dtype={'code': 'str'})
    area_title = {}
    for no, name in enumerate(np.unique(area['area'])):
        area_title[name] = no
    result = {}
    for _, row in area.iterrows():
        result[check(row['code'])] = area_title[row['area']]
    return result


# 公司
def company_class():
    company = pd.read_csv('datas/company.csv', encoding='utf-8', dtype={'code': 'str'})
    result = {}
    # for i in range(len(company)):
    #     result[company.ix[i, 0]] = company.ix[i, 1]
    for _, row in company.iterrows():
        result[check(row['code'])] = row['label']
    return result


# 市值
def outstanding_class():
    otsd = pd.read_csv('datas/basic.csv', index_col=0, encoding='utf-8', dtype={'code': 'str'})['outstanding']
    otsd_arr = np.asarray(otsd)
    result = {}
    for i in range(len(otsd_arr)):
        if otsd_arr[i] > 1:
            result[check(otsd.index[i])] = 0
        elif otsd_arr[i] < 0.5:
            result[check(otsd.index[i])] = 2
        else:
            result[check(otsd.index[i])] = 1
    return result


# 基金重仓股
def fund_class():
    fund = pd.read_csv('datas/fund/20173.csv', index_col=1, dtype={'code': 'str'})['ratio']
    fund_arr = np.asarray(fund)
    result = {}
    for i in range(len(fund_arr)):
        if fund_arr[i] > 0.2:
            result[check(fund.index[i])] = 1
        else:
            result[check(fund.index[i])] = 0
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
                    pa[targets[check(str(row["stkcode"]))]] += row["matchamt"]
                except KeyError:
                    continue
            else:
                targets = dynamic_query(label, int2date(row["busi_date"]), check(str(row["stkcode"])))
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
labels_name_part = ["客户号", "偏好高市盈率", "偏好中市盈率", "偏好低市盈率", "偏好高市值", "偏好中市值", "偏好低市值", "偏好高市净率", "偏好中市净率", "偏好低市净率", "偏好盈利股", "偏好亏损股", "偏好IT设备", "偏好专用设备", "偏好中药", "偏好互联网", "偏好仓储物流", "偏好仪器仪表", "偏好保险", "偏好公共交通", "偏好公用事业", "偏好农林", "偏好农药", "偏好包装印刷", "偏好化工", "偏好化纤", "偏好化肥", "偏好医疗服务", "偏好医药制造", "偏好医药流通", "偏好园林工程", "偏好塑胶制品", "偏好复合材料", "偏好多元金融", "偏好家电", "偏好工程建筑", "偏好工程机械", "偏好建材", "偏好房地产", "偏好摩托自行车", "偏好文化传媒", "偏好旅游", "偏好日用化工", "偏好有色矿采", "偏好有色金属", "偏好服装", "偏好木材家具", "偏好机场", "偏好机床设备", "偏好机械配件", "偏好水上运输", "偏好水泥", "偏好汽车制造", "偏好汽车配件", "偏好港口", "偏好煤炭", "偏好牧渔", "偏好环境保护", "偏好生物制品", "偏好电力", "偏好电子元器件", "偏好电子设备", "偏好电气设备", "偏好石油石化", "偏好纺织", "偏好综合行业", "偏好航天国防", "偏好航空运输", "偏好船舶制造", "偏好证券期货", "偏好贸易", "偏好软件及系统", "偏好轻工产品", "偏好轻工机械", "偏好通信服务", "偏好通信设备", "偏好造纸", "偏好酒店餐饮", "偏好酿酒", "偏好金属制品", "偏好钢铁", "偏好铁路运输", "偏好银行", "偏好零售业", "偏好食品饮料", "偏好饲料加工", "偏好高速公路", "偏好3D打印","偏好4G概念","偏好5G概念","偏好IPV6概念","偏好IP变现","偏好O2O模式","偏好QFII重仓","偏好ST板块","偏好三沙概念","偏好三网融合","偏好上海本地","偏好上海自贸","偏好业绩预升","偏好业绩预降","偏好东亚自贸","偏好丝绸之路","偏好云计算","偏好互联金融","偏好京津冀","偏好低碳经济","偏好体育概念","偏好保险重仓","偏好保障房","偏好信息安全","偏好信托重仓","偏好充电桩","偏好免疫治疗","偏好养老概念","偏好内贸规划","偏好军工航天","偏好军民融合","偏好农村金融","偏好准ST股","偏好出口退税","偏好分拆上市","偏好创投概念","偏好券商重仓","偏好前海概念","偏好博彩概念","偏好卫星导航","偏好参股金融","偏好可燃冰","偏好含B股","偏好含H股","偏好含可转债","偏好固废处理","偏好国产软件","偏好国企改革","偏好图们江","偏好土地流转","偏好地热能","偏好基因概念","偏好基因测序","偏好基因芯片","偏好基金重仓","偏好外资背景","偏好多晶硅","偏好天津自贸","偏好太阳能","偏好央企50","偏好奢侈品","偏好婴童概念","偏好安防服务","偏好宽带提速","偏好广东自贸","偏好建筑节能","偏好循环经济","偏好成渝特区","偏好抗流感","偏好抗癌","偏好振兴沈阳","偏好摘帽概念","偏好整体上市","偏好文化振兴","偏好新能源","偏好新零售","偏好日韩贸易","偏好智能交通","偏好智能家居","偏好智能机器","偏好智能电网","偏好智能穿戴","偏好未股改","偏好本月解禁","偏好机器人概念","偏好核电核能","偏好次新股","偏好武汉规划","偏好民营医院","偏好民营银行","偏好氢燃料","偏好水利建设","偏好水域改革","偏好污水处理","偏好汽车电子","偏好油气改革","偏好沿海发展","偏好海上丝路","偏好海峡西岸","偏好海工装备","偏好海水淡化","偏好涉矿概念","偏好深圳本地","偏好燃料电池","偏好物联网","偏好特斯拉","偏好猪肉","偏好生态农业","偏好生物燃料","偏好生物疫苗","偏好生物育种","偏好生物质能","偏好甲型流感","偏好电商概念","偏好电子支付","偏好皖江区域","偏好石墨烯","偏好碳纤维","偏好社保重仓","偏好稀土永磁","偏好稀缺资源","偏好空气治理","偏好粤港澳","偏好维生素","偏好绿色照明","偏好网络游戏","偏好聚氨酯","偏好股期概念","偏好股权激励","偏好自贸区","偏好节能","偏好节能环保","偏好苹果概念","偏好草甘膦","偏好蓝宝石","偏好融资融券","偏好装饰园林","偏好触摸屏","偏好资产注入","偏好赛马概念","偏好超大盘","偏好超导概念","偏好超级细菌","偏好迪士尼","偏好送转潜力","偏好重组概念","偏好金融参股","偏好金融改革","偏好铁路基建","偏好锂电池","偏好长株潭","偏好阿里概念","偏好陕甘宁","偏好雄安新区","偏好页岩气","偏好风沙治理","偏好风能","偏好风能概念","偏好食品安全","偏好高校背景","偏好黄河三角","偏好黄金概念","偏好上海", "偏好云南", "偏好内蒙", "偏好北京", "偏好吉林", "偏好四川", "偏好天津", "偏好宁夏", "偏好安徽", "偏好山东", "偏好山西", "偏好广东", "偏好广西", "偏好新疆", "偏好江苏", "偏好江西", "偏好河北", "偏好河南", "偏好浙江", "偏好海南", "偏好深圳", "偏好湖北", "偏好湖南", "偏好甘肃", "偏好福建", "偏好西藏", "偏好贵州", "偏好辽宁", "偏好重庆", "偏好陕西", "偏好青海", "偏好黑龙江", "偏好类别一", "偏好类别二", "偏好类别三", "偏好类别四", "偏好非基金重仓股", "偏好基金重仓股", "偏好低换手率", "偏好中换手率", "偏好高换手率", "偏好低股价", "偏好中股价", "偏好高股价", "偏好非热点", "偏好热点", "偏好macd金叉买入", "偏好macd死叉卖出", "偏好kdj金叉买入", "偏好kdj死叉卖出", "两只乌鸦形态", "三只乌鸦形态", "三内部上涨和下跌形态", "三线打击形态", "三外部上涨和下跌形态", "南方三星形态", "三个白兵形态", "大敌当前形态", "捉腰带线形态", "脱离形态", "藏婴吞没形态", "反击线形态", "乌云压顶形态", "蜻蜓十字形态", "十字暮星形态", "暮星形态", "墓碑十字形态", "锤头形态", "上吊线形态", "母子线形态", "十字孕线形态", "风高浪大线形态", "家鸽形态", "三胞胎乌鸦形态", "颈内线形态", "倒锤头形态", "梯底形态", "相同低价形态", "十字晨星形态", "晨星形态", "颈上线形态", "刺透形态", "上升/下降三法形态", "射击之星形态", "停顿形态", "跳空并列阴阳线形态", "奇特三河床形态", "向上跳空的两只乌鸦形态", "上升/下降跳空三法形态", "操作频率（买）", "操作频率（卖）", "调仓情况", "持仓集中度趋势", "趋势显著性"]