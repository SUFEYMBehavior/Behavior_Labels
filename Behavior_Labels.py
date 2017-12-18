# -*- coding=utf-8 -*-
import tushare as ts
import pandas as pd
import datetime
import pickle
from utility import *
import numpy as np
import time

class Users():
    def __init__(self, q=0.5, r=0.5, δ=1e-4,database='db_0901', endtime='20170915', threshold=20, fromcsv=False):
        self.hist_data = 'hist/'
        self.ed = endtime
        self.Q = q
        self.R = r
        self.δ = δ
        '''self.X_flag = 0
        self.X_sta = 0.00
        self.S_flag = 0
        self.S_sta = 0.00
        self.C_flag = 0
        self.C_sta = 0.00
        self.K_flag = 0
        self.K_sta = 0.00'''
        self.fromcsv = fromcsv
        self.threshold = threshold
        self.ms = MSSQL(host="localhost", user="SA", pwd="!@Cxy7300", db=database)
        if self.fromcsv == True:
            custids_df = pd.DataFrame(self.ms.ExecQuery('select distinct custid from tcl_logasset'), columns=['custid'])
            self.custids = set(custids_df['custid'])
        self.blacklist = set('204001')
        '''self.dict_keys = ['客户号',
            '持仓概念偏好',
            '持仓浮盈浮亏',
            '持股周期',
            '持仓偏好',
            '涨跌停操作偏好',
            #'委托方式偏好': self.get_WTFS_l(custid),
            '止盈偏好',            '止损偏好',
            'K线图',
            '日K买入',  '周K买入', '月K买入',
            '日K卖出',  '周K卖出', '月K卖出',
            '波段操作偏好',
            'bs操作偏好',
            '过度自信',
            '偏好市价委托',
            '偏好限价委托']'''
        self.logcustid = ''
        self.fundcustid = ''
        self.stkcustid = ''
        self.codelist = pickle.load(open('bin/codeset.bin', 'rb'))
        self.labels = ['客户号','异常波动', '持仓集中度（概念）', '持仓集中度（个股）', '持仓盈亏', '持股周期', '涨跌停操作偏好',
                       '高转送股偏好', '止盈偏好', '止损偏好',
                       '追高买入', '追高卖出', '看涨买入', '看涨卖出', '看跌买入', '看跌卖出',
                       '日线买入', '日线卖出', '周线买入', '周线卖出', '月线买入', '月线卖出',
                       '早盘集合竞价买入', '早盘集合竞价卖出', '早盘开盘后买入',
                       '早盘开盘后卖出', '早盘开盘中买入', '早盘开盘中卖出', '早盘收盘前买入',
                       '早盘收盘前卖出', '下午盘开盘前买入', '下午盘开盘前卖出', '下午盘开盘后买入',
                       '下午盘开盘后卖出', '下午盘开盘中买入', '下午盘开盘中卖出', '下午盘收盘前买入',
                       '下午盘收盘前卖出', '收盘结合竞价买入', '收盘集合竞价卖出',
                       '波段操作偏好', 'bs操作偏好', '打新股情况',
                       '高点买入', '高点卖出', '上涨买入', '上涨卖出', '下跌买入', '下跌卖出', '低点买入', '低点卖出',
                       'ST股偏好', '次新股偏好', '可转债偏好', '中小板偏好', '永远满仓', '注意力驱动偏好买', '注意力驱动偏好卖', '过度自信']

    def decay_divide(self, cal_num, cal_all):
        if cal_all == 0:
            return -1
        else:
            return cal_num/cal_all
        '''thre = self.threshold
        p = cal_num/cal_all
        if cal_all >= thre:
            return p
        else:
            return p * math.sqrt(cal_all/thre)'''
    #
    def get_funddata(self, custid):

        if custid == self.fundcustid:
            return self.fundasset
        if self.fromcsv:
            df = pd.read_csv('test/fundasset.csv')
            fundasset = df[df['custid'] == int(custid)]
        else:
            log = self.ms.ExecQuery("select distinct custid, marketvalue, busi_date, fundbal, fundlastbal from tcl_fundasset "
                                   "where custid = %s"
                                   "order by busi_date ASC "%str(custid))
            fundasset = pd.DataFrame(log, columns=['custid', 'marketvalue', 'busi_date', 'fundbal', 'fundlastbal'])
        self.fundcustid = custid
        self.fundasset = fundasset
        return fundasset

    #fn_timer

    def get_logdata(self, custid):

        if custid == self.logcustid:
            return self.logasset
        if self.fromcsv:
            df = pd.read_csv('test/logasset.csv')
            logasset = df[df['custid'] == int(custid)]
        else:
            log = self.ms.ExecQuery("select distinct custid, stkcode, busi_date, matchqty, bsflag, stkeffect, stkname,"
                                    "matchprice, matchamt, orderdate, ordertime, cleardate, market from tcl_logasset "
                                    "where matchamt > 0 and matchqty > 0 and busi_date != 0 and custid = %s "
                                    "order by busi_date ASC " % str(custid))
            logasset = pd.DataFrame(log, columns=['custid', 'stkcode', 'busi_date', 'matchqty', 'bsflag', 'stkeffect', 'stkname',
                                                  'matchprice', 'matchamt', 'orderdate', 'ordertime', 'cleardate',
                                                  'market'])
            #logasset = logasset[(logasset['matchamt'] > 0) & (logasset['matchqty'] > 0) & (logasset['busi_date'] != 0)]
        self.logcustid = custid
        self.logasset = logasset
        return logasset

    #
    def get_stkdata(self, custid):
        if custid == self.stkcustid:
            return self.stkasset
        if self.fromcsv:
            df = pd.read_csv('test/stkasset.csv')
            stkasset = df[df['custid'] == int(custid)]
        else:
            log = self.ms.ExecQuery(
                "select distinct custid, stkcode, busi_date, stkbal, stklastbal from tcl_stkasset "
                "where custid = %s "
                "order by busi_date ASC " % str(custid))
            stkasset = pd.DataFrame(log, columns=['custid', 'stkcode', 'busi_date', 'stkbal', 'stklastbal'])
        self.stkcustid = custid
        self.stkasset = stkasset
        return stkasset

    # 获得需要统计所有人数据的字典
    def get_dicts(self):
        '''


        try:
            self.hyg_dict = pickle.load(open('dict/hyg_dict', 'rb'))
        except:
            self.hyg_dict = self.get_HYG_l(self.custids)
            pickle.dump(self.hyg_dict, open('dict/hyg_dict', 'wb'))
        '''
        try:
            self.phwt_dict = pickle.load(open('dict/phwt_dict', 'rb'))
        except:
            self.phwt_dict = self.get_PHWT_l()
            pickle.dump(self.phwt_dict, open('dict/phwt_dict', 'wb'))
        '''
        try:
            self.abn_dict = pickle.load(open('dict/abn_dict', 'rb'))
        except:
            self.abn_dict = self.abnormals_l()
            pickle.dump(self.abn_dict, open('dict/abn_dict', 'wb'))

        try:
            self.hsr_dict = pickle.load(open('dict/hsr_dict', 'rb'))
        except:
            self.hsr_dict = self.high_shares_l()
            pickle.dump(self.hsr_dict, open('dict/hsr_dict', 'wb'))
        try:
            self.xg_dict = pickle.load(open('dict/xg_dict', 'rb'))
        except:
            self.xg_dict = self.get_XG_dict()
            pickle.dump(self.xg_dict, open('dict/xg_dict', 'wb'))
        '''

    # 获取所有标签

    @fn_timer
    def get_labels(self, custid):

        zyl, zsl = self.get_ZYZS_l(custid)
        kli = self.get_Kline_l(custid)
        dl = self.get_DLCL2(custid)
        jt = self.get_JYSJ(custid)
        fst = self.get_FSTPH(custid)
        att = self.get_ZJLQD(custid)
        hsr = self.high_shares_l(custid)
        values = [custid, self.abnormals_l(custid),  self.hold_concept_var(custid), self.hold_var(custid), self.holding_float(custid),
                  self.holdings(custid, self.ed), self.limit_perference(custid), hsr, zyl, zsl,
                  kli[0], kli[1], kli[2], kli[3], kli[4], kli[5],
                  dl['DB'], dl['DS'], dl['WB'], dl['WS'], dl['MB'], dl['MS'],
                  jt['1B'], jt['1S'], jt['2B'], jt['2S'], jt['3B'], jt['3S'], jt['4B'], jt['4S'], jt['5B'], jt['5S'],
                  jt['6B'], jt['6S'], jt['7B'], jt['7S'], jt['8B'], jt['8S'], jt['9B'], jt['9S'],
                  self.get_BDCZ_l(custid), self.bs_operate_l(custid), self.get_XGQK(custid),
                  fst['highB'], fst['highS'], fst['upB'], fst['upS'], fst['downB'], fst['downS'], fst['lowB'], fst['lowS'],
                  self.get_STPH(custid), self.get_CXPH(custid), self.get_KZPH(custid), self.get_ZXBPH(custid), self.get_YYMCPH(custid),
                  att[0], att[1], self.get_GDZX_l(custid)
                  ]
        return dict(zip(self.labels, values))


    #异常波动
    def abnormals_l(self, custid, spl=1 ):
        m = []
        user = self.get_logdata(custid)
        for i, row in user.iterrows():
            t2 = str(row["busi_date"])
            if len(t2) != 8:
                continue
            time = datetime.datetime.strptime(t2, '%Y%m%d')
            st_time = time - datetime.timedelta(days=30)
            t1 = datetime.datetime.strftime(st_time, '%Y-%m-%d')
            t2 = datetime.datetime.strftime(time, '%Y-%m-%d')
            zqdm = check(row['stkcode'])
            # 取前一个月的历史行情 
            if not zqdm in self.codelist:
                m.append(0)
            else:
                data = pd.read_csv(self.hist_data + zqdm + '.csv')
                data = data[(data['date']>=t1) & (data['date']<=t2)]
                if data is not None and len(data['p_change']) > 0:
                    p = data['p_change'].abs().mean()
                    m.append(p)
                else:
                    m.append(0)
        m = standardize(m)
        flag = [True if x > spl else False for x in m]
        user['flag'] = flag
        dic = calculate(user, label='flag')
        if dic.shape[0] == 0:
            return -1
        else:
            abl = dic.iloc[0]
            return self.Q * abl['Q'] + self.R * abl['R']

    #持仓概念集中度
    def hold_concept_var(self, custid, path='datas/concept.csv'):
        user = self.get_stkdata(custid)
        file = path
        concept = pd.read_csv(file)
        user['code'] = user['stkcode'].astype('int64')
        df = pd.merge(user, concept, on='code', how='left').dropna(axis=0)
        if df.shape[0] == 0:
            return -1
        keys = list(set(df['c_name']))
        res = pd.DataFrame(columns=keys)
        for tim in set(df['busi_date']):
            rec = df[df['busi_date']==tim]
            dic = dict.fromkeys(keys, 0)
            for _, row in rec.iterrows():
                dic[row['c_name']] += row['stkbal']
            res = res.append(dic, ignore_index=True)
        a = np.array(res).astype('float')
        su = np.maximum(np.sum(a, axis=1), self.δ)
        a = np.divide(a, su[:, None])
        return np.mean(np.std(a, axis=1))

    #持仓地区集中度
    def hold_area_var(self, custid, path='datas/area.csv'):
        user = self.get_stkdata(custid)
        file = path
        concept = pd.read_csv(file)
        user['code'] = user['stkcode'].astype('int64')
        df = pd.merge(user, concept, on='code', how='left').dropna(axis=0)
        if df.shape[0] == 0:
            return -1
        keys = list(set(df['area']))
        res = pd.DataFrame(columns=keys)
        for tim in set(df['busi_date']):
            rec = df[df['busi_date']==tim]
            dic = dict.fromkeys(keys, 0)
            for _, row in rec.iterrows():
                dic[row['area']] += row['stkbal']
            res = res.append(dic, ignore_index=True)
        a = np.array(res).astype('float')
        sum = np.maximum(np.sum(a, axis=1), self.δ)
        a = np.divide(a, sum[:, None])
        return np.mean(np.std(a, axis=1))

    #持仓个股集中度
    def hold_var(self, custid):
        df = self.get_stkdata(custid)
        keys = list(set(df['stkcode']))
        res = pd.DataFrame(columns=keys)
        for tim in set(df['busi_date']):
            rec = df[df['busi_date'] == tim]
            dic = dict.fromkeys(keys, 0)
            for _, row in rec.iterrows():
                dic[row['stkcode']] += row['stkbal']
            res = res.append(dic, ignore_index=True)
        a = np.array(res).astype('float')
        sum = np.maximum(np.sum(a, axis=1), self.δ)
        a = np.divide(a, sum[:, None])
        return np.mean(np.std(a, axis=1))

    #持仓盈亏
    def holding_float(self, custid):
        df = self.get_funddata(custid)
        df = df[df['marketvalue'] > 0]
        if df.shape[0] == 0:
            return -1
        last = float(max(df['marketvalue'].iloc[0] + df['fundbal'].iloc[0], 0.01))
        holding_float = 0
        for _, row in df.iterrows():
            now = max(float(row['marketvalue'] + row['fundlastbal']), 0.0001)
            holding_float += (now-last)/float(last)
            last = float(row['marketvalue'] + row['fundbal'])
        return holding_float/df.shape[0]

    #持股周期 不加decay
    def holdings(self, custid, end_time):
        user = self.get_logdata(custid)
        avg_time = []
        holdstock = {}
        for i, row in user.iterrows():
            cjsl = int(row["matchqty"])
            flag = row['stkeffect']>0
            row['matchqty'] = abs(row['matchqty'])
            zqdm = check(row["stkcode"])
            time = datetime.datetime.strptime(str(int(row["orderdate"])), '%Y%m%d')
            if zqdm not in holdstock.keys():
                holdstock[zqdm] = []
            holdqueue = holdstock[zqdm]
            if flag:  # 如果是买入则在队尾添加
                holdqueue.append((cjsl, time))
            else:
                for index, (num, t) in enumerate(holdqueue):
                    delta = time - t
                    if num is 0:
                        continue
                    if num >= cjsl:  # 当前这笔交易被“填平”了
                        num -= cjsl
                        avg_time.append((cjsl, delta.days))
                        holdqueue[index] = (num, t)
                        break
                    else:
                        avg_time.append((num, delta.days))
                        cjsl -= num
                        num = 0
                        holdqueue[index] = (num, t)
        ed = datetime.datetime.strptime(str(int(end_time)), '%Y%m%d')
        for code in holdstock:
            holdqueue = holdstock[code]
            for index, (num, t) in enumerate(holdqueue):
                if num > 0:
                    delta = ed - t
                    avg_time.append((num, delta.days))
                    num = 0

        times = nums = 0
        for num, time in avg_time:
            nums += num
            times += num * time
        if nums == 0:
            return -1
        else:
            return times/nums
    
    def limit_perference(self, custid, time_range=1):
        custids = [0, 0]
        user = self.get_logdata(custid)
        for i, row in user.iterrows():
            cal_num = custids[0]
            cal_all = custids[1]
            t2 = str(int(row["orderdate"]))
            time = datetime.datetime.strptime(t2, '%Y%m%d')
            st_time = time - datetime.timedelta(days=time_range)
            t1 = datetime.datetime.strftime(st_time, '%Y-%m-%d')
            t2 = datetime.datetime.strftime(time, '%Y-%m-%d')
            zqdm = check(row['stkcode'])
            if zqdm in self.codelist:
                data = pd.read_csv(self.hist_data+zqdm+'.csv')
                data = data[(data['date']>=t1) & (data['date']<=t2)]
                
            # 取前一个月的历史行情 如果文件中有则取 否则查询
                if data is None or data.shape[0] == 0:
                    continue
                p_change = data['p_change'].max()
                q_change = data['p_change'].min()
                cal_all += 1
                if p_change >= 9.95 and row['stkeffect'] > 0:
                    cal_num += 1
                elif q_change <= -9.95 and row['stkeffect'] < 0:
                    cal_num += 1
                custids = [cal_num, cal_all]
        return self.decay_divide(custids[0], custids[1])
    
    def high_shares_l(self, custid, path='datas/shares.csv', share_limit=5):
        shr = pd.read_csv(path)
        user = self.get_logdata(custid)
        flag = []

        for _, row in user.iterrows():
            sh = shr[shr['code'] == int(row["stkcode"])]
            sh = sh[sh['report_date']<=str(int(row['orderdate']))]['shares']
            if sh.shape[0] == 0:
                flag.append(False)
                continue
            tm = sh.iloc[-1]
            if tm > share_limit:
                flag.append(True)
            else:
                flag.append(False)

        user['flag'] = flag
        dic = calculate(user, 'flag')
        if dic.shape[0] == 0:
            return -1
        else:
            hsr = dic.iloc[0]
            return self.Q * hsr['Q'] + self.R * hsr['R']

    def calculate_the_means_and_var(self, dict_name, col_number):
        nlist = []
        for d in dict_name.keys():
            if col_number == 'none':
                nlist.append(dict_name[d])
            elif dict_name[d][0] == -1:
                continue
            else:
                nlist.append(dict_name[d][col_number])
        narray = np.array(nlist)
        mean = narray.mean()
        var = narray.var()
        return mean, var

    def get_n_days_date(self, today_string, n):
        today_date = time.strptime(today_string, "%Y%m%d")  # 字符串转换成time类型
        # print type(date)  # 查看date的类型<type 'time.struct_time'>
        date = datetime.datetime(today_date[0], today_date[1], today_date[2])  # time类型转换成datetime类型
        day = date.strftime('%w')
        ntime = date - datetime.timedelta(days=n)
        time_string = ntime.strftime("%Y-%m-%d")
        return time_string

    def Checktime(self, starttime, endtime, given_time):
        Flag = 0
        starttime = time.strptime(starttime, '%H:%M:%S')
        endtime = time.strptime(endtime, '%H:%M:%S')
        if (starttime < given_time) and (endtime > given_time):
            Flag = 1
        else:
            Flag = 0
        return Flag

    
    # 止盈止损(输入一个客户号即可得到标签，无需用到所有人的数据）
    def get_ZYZS_l(self, custid):
        user = self.get_logdata(custid)
        zy_dict = {}
        zs_dict = {}
        dict = {}
        total_ylcjsl = 0
        total_kscjsl = 0
        total_ylv = 0
        total_ksl = 0
        for _, line in user.iterrows():
            zqfss = int(line['stkeffect'])
            cjsl = str(line['matchqty'])
            zqdm = check(line['stkcode'])
            cjjg = float(line['matchprice'])
            if cjsl == "0":
                continue
            elif zqfss >0:                # 委托类别为买入
                cjsl = float(cjsl)
                if zqdm in dict.keys():  # 如果已存在该证券，则证券均值改变，股数改变

                    dict[zqdm][0] = (dict[zqdm][0] * dict[zqdm][1] + cjsl * cjjg) / (dict[zqdm][1] + cjsl)
                    dict[zqdm][1] = dict[zqdm][1] + cjsl
                else:  # 如果没有该证券，则新建一个字典
                    jy_list = []
                    jy_list.append(cjjg)
                    jy_list.append(cjsl)
                    dict[zqdm] = jy_list
            elif zqfss <0:  # 委托类别为卖出
                cjsl = float(cjsl)
                if zqdm in dict.keys():  # 已经有买入交易
                    jc = cjjg - dict[zqdm][0]
                    dict[zqdm][1] = max((dict[zqdm][1] - cjsl), 0)
                    if jc > 0:
                        ylv = jc / dict[zqdm][0]  # 单笔交易的盈利率
                        total_ylcjsl = total_ylcjsl + cjsl
                        total_ylv = ylv * cjsl + total_ylv
                    else:
                        ksl = -jc / dict[zqdm][0]  # 单笔交易的亏损率
                        total_kscjsl = total_kscjsl + cjsl
                        total_ksl = ksl * cjsl + total_ksl
                else:
                    continue  # 如果之前没有该股票的买入记录（数据时间问题）则忽视该笔交易
        zyl = self.decay_divide(total_ylv, total_ylcjsl)
        zsl = self.decay_divide(total_ksl, total_kscjsl)
        return zyl, zsl

    # 活跃股（返回的是集体的偏好委托，需要用到均值和方差,应该套用q&r公式) 和换手率重复
    '''def get_HYG_l(self, custids):
        hyg_dict = {}
        for custid in custids:
            total_cjje = 0.01
            hyg_jyzj = 0
            user = self.get_logdata(custid)
            for _, line in user.iterrows():
                zqfss = int(line['stkeffect'])
                cjsl = str(line['matchqty'])
                zqdm = line['stkcode']
                cjje = float(line['matchamt'])
                if (zqfss > 0) and (cjsl != "0"):
                    total_cjje = total_cjje + cjje
                    wtrq = str(line['orderdate'])
                    wtrq = self.get_n_days_date(wtrq, 0)
                    try:
                        gphq = ts.get_hist_data(zqdm, start=wtrq, end=wtrq)
                        hsl = float(gphq['turnover'][0])
                    except:
                        continue
                    if hsl > 0.02: hyg_jyzj = hyg_jyzj + cjje
                else:
                    continue
            hyg_ph = hyg_jyzj / total_cjje
            hyg_dict[custid] = hyg_ph

        # 标准化：
        mean, var = self.calculate_the_means_and_var(hyg_dict, 'none')
        for custid in hyg_dict.keys():
            hyg_dict[custid] = (hyg_dict[custid] - mean) / var
            if hyg_dict[custid] > 0.85:
                hyg_dict[custid] = 1  # 80%分位数
            else:
                hyg_dict[custid] = 0
        return hyg_dict'''

    
    # k_line(输入客户号即可返回，不用考虑其他用户)
    def get_Kline_l(self, custid, path='datas/data_Kline'):
        kline_dict = {}
        zgmr_times = 0
        zgmc_times = 0
        kdmr_times = 0
        kdmc_times = 0
        kzmr_times = 0
        kzmc_times = 0
        total_times = 0
        user = self.get_logdata(custid)
        for _, line in user.iterrows():
            zqdm = check(line['stkcode'])
            zqfss = int(line['stkeffect'])
            wtrq = str(line['orderdate'])
            yesterday = self.get_n_days_date(wtrq, 10)
            tomorrow = self.get_n_days_date(wtrq, -10)
            wtrq = self.get_n_days_date(wtrq, 0)
            if zqdm not in self.codelist:
                continue
            else:
                data = pd.read_csv(self.hist_data+ zqdm + '.csv')
                df1 = data[(data['date']>=yesterday) & (data['date']<=wtrq)]
                df2 = data[(data['date']>=wtrq) & (data['date']<=tomorrow)]
            if df1 is None or df2 is None or len(df1) < 2 or len(df2) < 2:
                continue

            yesterday_5ma = float(df1.iloc[1]['ma5'])
            today_5ma = float(df1.iloc[0]['ma5'])
            tomarrow_5ma = float(df2.iloc[-2]['ma5'])
            if zqfss >0 and yesterday_5ma < today_5ma and tomarrow_5ma > today_5ma:
                zgmr_times = zgmr_times + 1
            elif zqfss >0 and yesterday_5ma > today_5ma:
                kdmr_times = kdmr_times + 1
            elif zqfss >0 and yesterday_5ma <= today_5ma:
                kzmr_times = kzmr_times + 1
            elif zqfss <0 and yesterday_5ma < today_5ma and tomarrow_5ma > today_5ma:
                zgmc_times = zgmc_times + 1
            elif zqfss <0 and yesterday_5ma > today_5ma:
                kdmc_times = kdmc_times + 1
            elif zqfss <0 and yesterday_5ma <= today_5ma:
                kzmc_times = kzmc_times + 1
            total_times = total_times + 1
        zgmr_fre = self.decay_divide(zgmr_times, total_times)
        zgmc_fre = self.decay_divide(zgmc_times, total_times)
        kzmr_fre = self.decay_divide(kzmr_times, total_times)
        kzmc_fre = self.decay_divide(kzmc_times, total_times)
        kdmr_fre = self.decay_divide(kdmr_times, total_times)
        kdmc_fre = self.decay_divide(kdmc_times, total_times)

        kline_fre_list= [zgmr_fre, zgmc_fre, kzmr_fre, kzmc_fre, kdmr_fre, kdmc_fre]

        return kline_fre_list


    def get_DLCL2(self, custid, path='datas/industry.csv'):
        user = self.get_logdata(custid)
        d_kind = pd.read_csv(path)
        # d_kind["code"] = d_kind["code"].astype(int)
        d_kind["code"] = d_kind["code"]
        d_kind = d_kind.sort_values('code')
        # user['stkcode'] = user['stkcode'].astype(int)
        user['stkcode'] = user['stkcode'].astype('int')
        datas = pd.merge(user, d_kind, how="left", left_on="stkcode", right_on="code", )
        datas = pd.DataFrame(datas)
        datas = datas.dropna(how='any', axis=0)
        '''if len(datas) == 0:
            return -1'''
        res_l = []
        dataB = datas[datas["stkeffect"] > 0.00]
        dataS = datas[datas["stkeffect"] < 0.00]
        alW = [dataB, dataS]
        for i, data in enumerate(alW):
            number = data.shape[0]
            kh_res = {}
            kh_res["custid"] = int(custid)
            if i == 0:
                way = "B"
            else:
                way = "S"
            if number == 0:
                kh_res["D" + way] = -1
                kh_res["W" + way] = -1
                kh_res["M" + way] = -1
            else:
                data_DK = data.loc[:, ["orderdate", "c_name"]].drop_duplicates()
                data_D = data_DK["orderdate"].drop_duplicates()
                D_n = data_D.shape[0]  # 全部的日期

                for n in range(D_n):
                    end_date = datetime.datetime.strptime(str(data_D.iloc[n]), '%Y%m%d')
                    end_date_S = end_date.strftime('%Y-%m-%d')
                    # delta_D = datetime.timedelta(days=2)
                    # delta_W = datetime.timedelta(days=14)
                    # start_date_D = (end_date - delta_D).strftime('%Y-%m-%d')
                    # start_date_W = (end_date - delta_W).strftime('%Y-%m-%d')

                    M = end_date.month
                    Y = end_date.year
                    end_date_M = str(Y) + "-" + str(M) + "-01"
                    '''if M > 2:
                        start_date_M = str(Y) + "-" + str(M - 2) + "-01"
                    elif M == 2:
                        start_date_M = str(Y - 1) + "-" + str(12) + "-01"
                    else:
                        start_date_M = str(Y - 1) + "-" + str(11) + "-01"'''

                    Each_D_K = data_DK[data_DK["orderdate"] == data_D.iloc[n]]  # 找出当日所有申报的股票列表

                    if Each_D_K.shape[0] != 0:
                        Each_D_K_l = list(set(np.array(Each_D_K["c_name"]).tolist()))
                    else:
                        continue
                    for k in Each_D_K_l:
                        if str(k) == 'nan':
                            Each_D_K_l.remove(k)

                    f_d = open("datas/list/" + way + "_D_" + end_date_S + ".txt", "r")
                    f_w = open("datas/list/" + way + "_W_" + end_date_S + ".txt", "r")
                    f_m = open("datas/list/" + way + "_M_" + end_date_S + ".txt", "r")
                    buy_D = f_d.read().split()
                    buy_W = f_w.read().split()
                    buy_M = f_m.read().split()
                    Buy_D = []
                    Buy_W = []
                    Buy_M = []
                    for bd in buy_D:
                        Buy_D.append(int(bd))
                    for bw in buy_W:
                        Buy_W.append(int(bw))
                    for bm in buy_M:
                        Buy_M.append(int(bm))

                    Each_kh = data[data["orderdate"] == data_D.iloc[n]]
                    bdn = Each_kh[Each_kh['stkcode'].isin(Buy_D)].shape[0]
                    bwn = Each_kh[Each_kh["stkcode"].isin(Buy_W)].shape[0]
                    bmn = Each_kh[Each_kh["stkcode"].isin(Buy_M)].shape[0]

                    if "D" + way not in kh_res.keys():
                        kh_res["D" + way] = bdn
                        kh_res["W" + way] = bwn
                        kh_res["M" + way] = bmn
                    else:
                        kh_res["D" + way] += bdn
                        kh_res["W" + way] += bwn
                        kh_res["M" + way] += bmn

                kh_res["D" + way] = self.decay_divide(float(kh_res["D" + way]), float(number))
                kh_res["W" + way] = self.decay_divide(float(kh_res["W" + way]), float(number))
                kh_res["M" + way] = self.decay_divide(float(kh_res["M" + way]), float(number))
            res = pd.DataFrame(columns=list(kh_res.keys()))
            res = res.append(kh_res, ignore_index=True)
            res_l.append(res)
        res = pd.merge(res_l[0], res_l[1], how="left", on="custid")
        return res.iloc[0]
    # 波段操作_CCI(输入客户号即可返回波段操作偏好）
    def get_CCI(self, df, N):
        df['typ'] = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = ((df['typ'] - df['typ'].rolling(N).mean()) /
                     (0.015 * abs(df['typ'] - df['typ'].rolling(N).mean()).rolling(N).mean()))
        return df

    def get_BDCZ_l(self, custid):
        bdcz_dict = {}
        cci_times = 0
        wt_times = 0
        user = self.get_logdata(custid)
        for _, line in user.iterrows():
            cjsl = str(line['matchqty'])
            zqfss = int(line['stkeffect'])
            if cjsl == "0":
                continue
            else:
                wt_times = wt_times + 1
                zqdm = check(line['stkcode'])
                wtrq = str(line['orderdate'])
                start_time = self.get_n_days_date(wtrq, 0)
                end_time = self.get_n_days_date(wtrq, 50)
                if zqdm in self.codelist:
                    gphq = pd.read_csv(self.hist_data + zqdm + '.csv')
                    gphq = gphq[gphq['date'] >= start_time][gphq['date'] <= end_time]
                else:
                    continue
                if gphq is None or len(gphq) == 0:
                    continue
                df = gphq[['open', 'high', 'low', 'close', 'volume']].sort_index(ascending=True)
                cci_df = self.get_CCI(df, 14)
                cci = cci_df.iloc[-1]['cci']
                if zqfss>0 and cci < -100: cci_times = cci_times + 1
                elif zqfss<0 and cci > 100: cci_times = cci_times + 1


        return self.decay_divide(cci_times, wt_times)

    #波段操作_bs操作偏好
    def bs_operate_l(self,custid, path='datas/bsday.csv'):

        bs_operate_dict = {}
        bs_times = 0
        wt_times = 0
        user = self.get_logdata(custid)
        bs_data = pd.DataFrame(pd.read_csv(path))
        for _, line in user.iterrows():
            cjsl = str(line['matchqty'])
            zqfss = int(line['stkeffect'])
            zqdm = check(line['stkcode'])
            wtrq = str(line['orderdate'])
            month = wtrq[4:6]
            if month[0] == '0':
                month = month[1]
            day = wtrq[6:8]
            if day[0] == '0':
                day = day[1]
            wtrq = wtrq[0:4]+'/'+month+'/'+day
            bs = bs_data[bs_data['Code'] == int(zqdm)]
            if cjsl == "0":
                continue

            elif zqfss > 0:# 委托类别为买入
                wt_times = wt_times + 1
                bs_times += bs[bs['BuyPointDate']==wtrq].shape[0]
            elif zqfss < 0:  # 委托类别为卖出
                wt_times = wt_times + 1
                bs_times += bs[bs['SellPointDate']==wtrq].shape[0]

        return self.decay_divide(bs_times, wt_times)

    # 委托方式需要根据标签数据
    # 委托渠道偏好
    '''
    def get_WTQD(self, custid):
    	datas = pd.read_excel(self.file)
    	dirctory = {1: "tel", 2: "card", 4: "key", 8: "account", 16: "long", 32: "internet", 64: "phone", 128: "bank"} #存在问题!!!!!!!!!!!!!!!
    	l = dirctory.keys()
    	datas = datas.loc[:, ["custid", "operway"]]
    	res_k = {}
    	res_k["custid"] = custid
    	df = datas[datas["custid"] == custid]
    	for w in l:
    		try:
    			n = (float(df[df["operway"] == w].shape[0]) / df.shape[0])
    		except:
    			n = (0 / df.shape[0])
    		res_w = n
    		res_k[w] = res_w
    	return res_k
    '''
    # 交易时间偏好 没有decay
    def get_JYSJ(self, custid):
        datas = self.logasset
        lb = {1: "ZPJH", 2: "ZPKH", 3: "ZPPZ", 4: "ZPSQ", 5: "ZPSH", 6: "XPKH", 7: "XPPZ", 8: "XPSQ", 9: "SPJH"}
        tim = {1: "09:30:00", 2: "10:00:00", 3: "11:00:00", 4: "11:30:00", 5: "13:00:00", 6: "13:30:00", 7: "14:30:00",
               8: "15:00:00",
               9: "14:57:00"}
        for t in range(9):  # 提前设定好每个标签类别对应的时间并解析成时间格式
            tim[t + 1] = time.strptime(tim[t + 1], '%H:%M:%S')

        res_al = []
        dataB = datas[datas["stkeffect"] > 0.00]
        dataS = datas[datas["stkeffect"] < 0.00]
        alW = [dataB, dataS]
        for _, data in enumerate(alW):

            if _ == 0:
                way = "B"
            else:
                way = "S"
            res_k = {}
            res_k["custid"] = int(custid)
            df = data
            aln = df.shape[0]
            if aln == 0:
                for j in range(9):
                    res_k[str(j + 1) + way] = -1
            else:
                for i in range(aln):  # 遍历该用户的每条交易记录
                    t = str(df['ordertime'].iloc[i])[:-2]
                    if len(t) < 5:
                        continue
                    trade_time = time.strptime(t, '%H%M%S')  # 解析该记录的用户交易时间
                    if (trade_time < tim[1]) | (trade_time >= tim[8]):
                        if str(1) + way not in res_k.keys():
                            res_k[str(1) + way] = 1
                        else:
                            res_k[str(1) + way] += 1
                    elif (trade_time >= tim[1]) & (trade_time < tim[2]):
                        if str(2) + way not in res_k.keys():
                            res_k[str(2) + way] = 1
                        else:
                            res_k[str(2) + way] += 1

                    elif (trade_time >= tim[2]) & (trade_time < tim[3]):
                        if str(3) + way not in res_k.keys():
                            res_k[str(3) + way] = 1
                        else:
                            res_k[str(3) + way] += 1

                    elif (trade_time >= tim[3]) & (trade_time < tim[4]):
                        if str(4) + way not in res_k.keys():
                            res_k[str(4) + way] = 1
                        else:
                            res_k[str(4) + way] += 1

                    elif (trade_time >= tim[4]) & (trade_time < tim[5]):
                        if str(5) + way not in res_k.keys():
                            res_k[str(5) + way] = 1
                        else:
                            res_k[str(5) + way] += 1

                    elif (trade_time >= tim[5]) & (trade_time < tim[6]):
                        if str(6) + way not in res_k.keys():
                            res_k[str(6) + way] = 1
                        else:
                            res_k[str(6) + way] += 1

                    elif (trade_time >= tim[6]) & (trade_time < tim[7]):
                        if str(7) + way not in res_k.keys():
                            res_k[str(7) + way] = 1
                        else:
                            res_k[str(7) + way] += 1

                    elif (trade_time >= tim[9]) & (trade_time < tim[8]) & (df['market'].iloc[i] == 0):  # 此处关于交易市场存在问题！！！！！
                        if str(9) + way not in res_k.keys():
                            res_k[str(9) + way] = 1
                        else:
                            res_k[str(9) + way] += 1

                    else:
                        if str(8) + way not in res_k.keys():
                            res_k[str(8) + way] = 1
                        else:
                            res_k[str(8) + way] += 1
                for m in range(9):  # 计算用户的每个交易时间偏好的频率
                    if str(m + 1) + way not in res_k.keys():
                        res_k[str(m + 1) + way] = 0.000
                    else:
                        res_k[str(m + 1) + way] = float(res_k[str(m + 1) + way]) / float(aln)
            res = pd.DataFrame(columns=list(res_k.keys()))
            res = res.append(res_k, ignore_index=True)
            resal = res_al.append(res)
        res = pd.merge(res_al[0], res_al[1], how="left", on="custid")
        return res.iloc[0]


    '''
    # 品种偏好
    def get_PZPH(self, custid):
    	datas = pd.read_excel(self.file)
    	N = ["custid", "bsflag", "market", "stktype", "matchamt","stkeffect"]
    	datas = datas[datas["matchamt"] != 0]
    	datas = datas[datas["stkeffect"]<0.00]
    	datas = datas.loc[:, N]
    	sort_label = pd.read_excel("pzph/pzbz.xlsx")                             #此处存在问题
    	datas = pd.merge(datas, sort_label, how="left", on=["market", "stktype"])      #此处存在问题
    	each_kh = {}
    	each_kh["custid"] = int(custid)
    	df = datas[datas['custid'] == int(custid)]
    	dfn = df.shape[0]
    	all_money = float(df.loc[:, "matchamt"].sum())
    	if (dfn == 0) or (all_money == 0.00):
    		for i in range(17):
    			each_kh[i] = 0.00
    	else:
    		for j in range(17):  # 遍历每个品种在该用户买入的记录个数和频率
    			each_kh[j] = float(df[df["FLLB"] == j].loc[:, "matchamt"].sum()) / all_money       #此处存在问题
    	return each_kh  # 合并所有用户的结果

    '''
    # 过度自信
    def get_GDZX_l(self, custid):
        gdzx_dic = {}
        user = self.get_logdata(custid)
        syl_dict = {}
        mr_list = []
        mc_list = []

        for _, line in user.iterrows():
            list1 = []
            cjsl = str(line['matchqty'])
            zqfss = int(line['stkeffect'])
            zqdm = check(line['stkcode'])
            wtrq = str(line['orderdate'])
            cjrq = str(line['cleardate'])
            cjjg = float((line['matchprice']))
            if cjsl == "0":
                continue
            elif zqfss > 0:  # 委托类别为买入
                cjsl = float(cjsl)
                if zqdm in syl_dict.keys():  # 如果已存在该证券，则证券均值改变，股数改变
                    syl_dict[zqdm][0] = (syl_dict[zqdm][0] * syl_dict[zqdm][1] + cjsl * cjjg) / (
                    syl_dict[zqdm][1] + cjsl)
                    syl_dict[zqdm][1] = syl_dict[zqdm][1] + cjsl
                else:  # 如果没有该证券，则新建一个字典
                    jy_list = []
                    jy_list.append(cjjg)
                    jy_list.append(cjsl)
                    syl_dict[zqdm] = jy_list
                list1.append(zqdm)
                list1.append(cjrq)
                mr_list.append(list1)

            elif zqfss < 0:  # 委托类别为卖出
                cjsl = float(cjsl)
                if zqdm in syl_dict.keys():  # 已经有买入交易
                    jc = cjjg - syl_dict[zqdm][0]
                    syl_dict[zqdm][1] = max(syl_dict[zqdm][1] - cjsl, 0)
                    syl = jc / max(syl_dict[zqdm][0], 0.01) # 单笔交易的盈利率
                    list1.append(zqdm)
                    list1.append(wtrq)
                    list1.append(cjrq)
                    list1.append(syl)
                    mc_list.append(list1)

        mr_df = pd.DataFrame(mr_list, columns=('stkcode', 'orderdate'))
        mc_df = pd.DataFrame(mc_list, columns=('stkcode', 'orderdate', 'cleardate', 'syl'))
        mc_times = 0
        gdzx_times = 0
        for _, mc_line in mc_df.iterrows():
            mc_times = mc_times + 1
            mr_zqdm_set = set()
            mc_syl = float(mc_line['syl'])
            mc_cjrq = str(mc_line['cleardate'])
            if mc_cjrq == '0':
                continue
            today_date = time.strptime(mc_cjrq, "%Y%m%d")
            today_date = datetime.datetime(today_date[0], today_date[1], today_date[2])
            ntime = today_date - datetime.timedelta(days=-2)
            mr_string = ntime.strftime("%Y%m%d")
            df = mr_df.loc[(mr_df['orderdate'] <= mr_string) & (mr_df['orderdate'] >= mc_cjrq)]
            if df.empty:
                continue
            else:
                for _, mr_line in df.iterrows():
                    mr_zqdm = mr_line['stkcode']
                    mr_zqdm_set.add(mr_zqdm)
                mr_zqdm_list = list(mr_zqdm_set)
                mc_mr_zqdm_df = mc_df[(mc_df['stkcode'].isin(mr_zqdm_list)) & (mc_df['orderdate'] > mr_string)]  # 不严谨，有点问题
                weights = []
                for _, mc in mc_mr_zqdm_df.iterrows():
                    time_mc =  time.strptime(mc['orderdate'], "%Y%m%d")
                    time_mc = datetime.datetime(time_mc[0], time_mc[1], time_mc[2])
                    weights.append(1/((time_mc - today_date).days**2))
                weight = np.array(weights)
                weight = weight / np.sum(weight)
                mc_mr_syl = np.dot(mc_mr_zqdm_df['syl'], weight)
                if mc_mr_syl < mc_syl: gdzx_times = gdzx_times + 1

        return self.decay_divide(gdzx_times, mc_times)

    def get_PHWT_l(self):
        min_time_interval = datetime.datetime.strptime("00:00:00", "%H:%M:%S") - datetime.datetime.strptime("00:00:00",
                                                                                                            "%H:%M:%S")
        max_time_interval = datetime.datetime.strptime("00:00:03", "%H:%M:%S") - datetime.datetime.strptime("00:00:00",
                                                                                                            "%H:%M:%S")
        phwt_dict = {}

        for custid in self.custids:
            sql = ("select distinct custid, matchtime, ordertime from tcl_logasset where custid=%s" % str(custid))
            ms = self.ms.ExecQuery(sql)
            user = pd.DataFrame(ms, columns=['KHH', 'CJSJ', 'WTSJ'])
            phwt_fre_list = []
            wt_times = 0
            sjwt_times = 0
            xjwt_times = 0
            for _, line in user.iterrows():
                wt_time = str(line['WTSJ'])[0:-2]
                # print(wt_time)
                # sb_time = line.split("\t")[30]
                # print(sb_time)
                cj_time = str(line['CJSJ'])[0:-2]
                # print(cj_time)
                while len(wt_time) < 6:
                    wt_time = '0' + wt_time
                while len(cj_time) < 6:
                    cj_time = '0' + cj_time
                w2 = datetime.datetime.strptime(wt_time, "%H%M%S")
                wt_time = time.strptime(wt_time, "%H%M%S")
                time_flag_1 = self.Checktime("09:30:00", "11:30:00", wt_time)
                time_flag_2 = self.Checktime("13:00:00", "15:00:00", wt_time)
                if time_flag_1 == 0 and time_flag_2 == 0:
                    wt_times = wt_times + 1
                    xjwt_times = xjwt_times + 1
                elif cj_time == "000000":
                    continue
                else:
                    cj_time = datetime.datetime.strptime(cj_time, "%H%M%S")

                    time_interval = cj_time - w2
                    if time_interval < min_time_interval:
                        continue
                    elif time_interval < max_time_interval:
                        wt_times = wt_times + 1
                        sjwt_times = sjwt_times + 1
                    else:
                        wt_times = wt_times + 1
                        xjwt_times = xjwt_times + 1
            sjwt_fre = self.decay_divide(sjwt_times, wt_times)
            xjwt_fre = self.decay_divide(xjwt_times, wt_times)
            phwt_dict[custid] = [sjwt_fre, xjwt_fre]

        sjwt_fre_mean, sjwt_fre_var = self.calculate_the_means_and_var(phwt_dict, 0)
        xjwt_fre_mean, xjwt_fre_var = self.calculate_the_means_and_var(phwt_dict, 1)
        for d in phwt_dict.keys():
            if phwt_dict[d][0] == -1:
                continue
            else:
                phwt_dict[d][0] = (phwt_dict[d][0] - sjwt_fre_mean) / sjwt_fre_var
                phwt_dict[d][1] = (phwt_dict[d][1] - xjwt_fre_mean) / xjwt_fre_var
                if phwt_dict[d][0] > phwt_dict[d][1]:
                    phwt_dict[d][0] = 1
                    phwt_dict[d][1] = 0
                else:
                    phwt_dict[d][0] = 0
                    phwt_dict[d][1] = 1

        return phwt_dict

    # 新股情况
    def get_XGQK(self, custid):
        df = self.get_logdata(custid)
        '''datas = pd.read_excel(self.file)
        B = ["OP", "OT"]
        N = ["custid", "bsflag"]  # 提取需要的数据
        datas = datas[datas["stkeffect"] < 0.00]
        datas = datas.loc[:, N]'''
        df = df[df['stkeffect'] > 0]
        each_kh = {}
        each_kh["custid"] = int(custid)
        dfn = df.shape[0]
        if dfn == 0:
            each_kh["lucky"] = -1
        else:
            suc = float(df[df["bsflag"] == "0T"].shape[0])  # 计算用户在两种业务科目的记录的个数
            all_times = float(df[df["bsflag"] == "0P"].shape[0])
            lucky = self.decay_divide(suc, all_times)
            each_kh["lucky"] = lucky
        return each_kh['lucky']
    # 动量策略
    '''def get_DLCL(self, custid, path='datas/industry.csv'):
        user = self.get_logdata(custid)
        d_kind = pd.read_csv(path)
        d_kind["code"] = d_kind["code"].astype(int)
        d_kind = d_kind.sort_values('code')
        user['stkcode'] = user['stkcode'].astype(int)
        datas = pd.merge(user, d_kind, how="left", left_on="stkcode", right_on="code",  )
        datas = pd.DataFrame(datas)
        datas = datas.dropna(how='any', axis=0)
        if len(datas) == 0:
            return -1
        res_l = []
        dataB = datas[datas["stkeffect"] > 0.00]
        dataS = datas[datas["stkeffect"] < 0.00]
        alW = [dataB, dataS]
        for i, data in enumerate(alW):
            number = data.shape[0]
            kh_res = {}
            kh_res["custid"] = int(custid)
            if i == 0:
                way = "B"
            else:
                way = "S"
            if number == 0:
                kh_res["D" + way] = -1
                kh_res["W" + way] = -1
                kh_res["M" + way] = -1
            else:
                data_DK = data.loc[:, ["orderdate", "c_name"]].drop_duplicates()
                data_D = data_DK["orderdate"].drop_duplicates()
                D_n = data_D.shape[0]  # 全部的日期

                way_name = way
                for n in range(D_n):
                    end_date = datetime.datetime.strptime(str(data_D.iloc[n]), '%Y%m%d')
                    end_date_S = end_date.strftime('%Y-%m-%d')
                    delta_D = datetime.timedelta(days=2)
                    delta_W = datetime.timedelta(days=14)
                    start_date_D = (end_date - delta_D).strftime('%Y-%m-%d')
                    start_date_W = (end_date - delta_W).strftime('%Y-%m-%d')

                    M = end_date.month
                    Y = end_date.year
                    end_date_M = str(Y) + "-" + str(M) + "-01"
                    if M > 2:
                        start_date_M = str(Y) + "-" + str(M - 2) + "-01"
                    elif M == 2:
                        start_date_M = str(Y - 1) + "-" + str(12) + "-01"
                    else:
                        start_date_M = str(Y - 1) + "-" + str(11) + "-01"

                    Each_D_K = data_DK[data_DK["orderdate"] == data_D.iloc[n]]  # 找出当日所有申报的股票列表

                    if Each_D_K.shape[0] != 0:
                        Each_D_K_l = list(set(np.array(Each_D_K["c_name"]).tolist()))
                    else:
                        continue
                    for k in Each_D_K_l:
                        if str(k) == 'nan':
                            Each_D_K_l.remove(k)
                    buy_D = []
                    buy_W = []
                    buy_M = []
                    for dkl in Each_D_K_l:  # 对其中的每一个股票
                        f_kind = open("kind/" + str(dkl) + ".txt")
                        l_kind = f_kind.read().split(" ")
                        l_kind = l_kind[:-1]
                        all_r_original = pd.DataFrame()
                        lk_n = len(l_kind)
                        # 该股票池的所有股票 计算其DMW
                        for lk in l_kind:
                            lk_dic = {}
                            lk_dic["code"] = int(lk)
                            codename = int(lk)

                            filename = str(codename) + '.csv'
                            try:
                                #d_D = ts.get_k_data(str(codename), start=start_date_D, end=end_date_S, ktype="D")
                                d_D = pd.read_csv('datas/D/' + filename)
                                d_D = d_D[d_D['date'] <= end_date_S].tail(2)

                                #d_W = ts.get_k_data(str(codename), start=start_date_W, end=end_date_S, ktype="W")
                                d_W = pd.read_csv('datas/W/' + filename)
                                d_W = d_W[d_W['date'] <= end_date_S].tail(2)

                                #d_M = ts.get_k_data(str(codename), start=start_date_M, end=end_date_M, ktype="M")
                                d_M = pd.read_csv('datas/M/' + filename)
                                d_M = d_M[d_M['date'] <= end_date_M].tail(2)
                                lk_dic["R_D"] = (d_D["close"].iloc[1] - d_D["close"].iloc[0]) / d_D["close"].iloc[0]
                                lk_dic["R_W"] = (d_W["close"].iloc[1] - d_W["close"].iloc[0]) / d_W["close"].iloc[0]
                                lk_dic["R_M"] = (d_M["close"].iloc[1] - d_M["close"].iloc[0]) / d_M["close"].iloc[0]
                                all_r_original = all_r_original.append(lk_dic, ignore_index=True)
                            except:
                                continue

                        if all_r_original.shape[0] == 0:
                            continue
                        else:
                            l = int(len(all_r_original) * 0.25)
                            B_D_L = np.array(
                                all_r_original.sort_values(by="R_D", ascending=False)['code'].iloc[0:l]).tolist()
                            buy_D += B_D_L
                            B_W_L = np.array(
                                all_r_original.sort_values(by="R_W", ascending=False)['code'].iloc[0:l]).tolist()
                            buy_W += B_W_L
                            B_M_L = np.array(
                                all_r_original.sort_values(by="R_M", ascending=False)['code'].iloc[0:l]).tolist()
                            buy_M += B_M_L
                    Buy_D = set(buy_D)
                    Buy_W = set(buy_W)
                    Buy_M = set(buy_M)
                    Each_kh = data[data["orderdate"] == data_D.iloc[n]]
                    bdn = Each_kh[Each_kh['stkcode'].isin(Buy_D)].shape[0]
                    bwn = Each_kh[Each_kh["stkcode"].isin(Buy_W)].shape[0]
                    bmn = Each_kh[Each_kh["stkcode"].isin(Buy_M)].shape[0]

                    if "D" + way not in kh_res.keys():
                        kh_res["D" + way] = bdn
                        kh_res["W" + way] = bwn
                        kh_res["M" + way] = bmn
                    else:
                        kh_res["D" + way] += bdn
                        kh_res["W" + way] += bwn
                        kh_res["M" + way] += bmn

                kh_res["D" + way] = float(kh_res["D" + way]) / float(number)
                kh_res["W" + way] = float(kh_res["W" + way]) / float(number)
                kh_res["M" + way] = float(kh_res["M" + way]) / float(number)
            res = pd.DataFrame(columns=list(kh_res.keys()))
            res = res.append(kh_res, ignore_index=True)
            res_l.append(res)
        res = pd.merge(res_l[0], res_l[1], how="left", on="custid")
        return res'''

    # 分时图偏好问题：无法得到确定时间的5分钟k,该函数只能得到tushare上能爬到的5分钟k数据，所以结果会有问题
    
    def get_FSTPH(self, custid):
        datas = self.logasset
        res_l = []
        dataB = datas[datas["stkeffect"] > 0.00]
        dataS = datas[datas["stkeffect"] < 0.00]
        alW = [dataB, dataS]
        for i, data in enumerate(alW):
            if i == 0:
                way = "B"
            else:
                way = "S"

            kh_res = {}
            kh_res["custid"] = int(custid)
            kh_df = data
            khn = kh_df.shape[0]
            if khn == 0:
                kh_res["up" + way] = 0.00
                kh_res["down" + way] = 0.00
                kh_res["high" + way] = 0.00
                kh_res["low" + way] = 0.00
            else:
                up = 0
                down = 0
                high = 0
                low = 0
                for kn in range(khn):  # 遍历每个用户的每条记录
                    date_s = kh_df['orderdate'].iloc[kn]
                    time_s = str(kh_df['ordertime'].iloc[kn])
                    while len(time_s)<8:
                        time_s = '0'+time_s
                    m = int(time_s[-5])
                    if m>=5:
                        m = '5'
                    else:
                        m = '0'
                    d1 = datetime.datetime.strptime(str(date_s) + " " + str(time_s)[:-5]+m, '%Y%m%d %H%M')
                    d1 = d1 + datetime.timedelta(days=84)
                    #d2 = d1 - datetime.timedelta(minutes=5)
                    d3 = d1 + datetime.timedelta(minutes=5)
                    last_time = d3.strftime('%Y-%m-%d %H:%M')
                    now_time=d1.strftime('%Y-%m-%d %H:%M')
                    #pro_time = d2.strftime('%Y-%m-%d %H:%M')
                    stkcode = check(kh_df['stkcode'].iloc[kn])
                    try:
                        df = pd.read_csv('datas/k/'+stkcode+'.csv')
                        sp = df[df["date"]==now_time].shape[0]
                    except:
                        continue

                    '''except:
                        df = ts.get_k_data(str(kh_df["stkcode"].iloc[kn]), ktype="5")  # 对于该记录进行分类，判断该记录的类别
                        print(str(kh_df['stkcode'].iloc[kn]))
                        df.to_csv('datas/k/'+kh_df['stkcode'].iloc[kn]+'.csv')'''
                    if sp > 0:
                        d = df[df["date"]<last_time]["open"].tail(3)
                        if float(d.iloc[1]) > float(d.iloc[0]):
                            if float(d.iloc[1]) > float(d.iloc[2]):
                                high += 1
                            else:
                                up += 1
                        else:
                            if float(d.iloc[1]) < float(d.iloc[0]):
                                if  (float(d.iloc[0])-float(d.iloc[1]))/float(d.iloc[0])>0.08:
                                    low += 1
                                else:
                                    down += 1
                    else:
                        continue
                kh_res["up" + way] = self.decay_divide(up, khn)  # 计算对应标签的频率
                kh_res["down" + way] = self.decay_divide(down, khn)
                kh_res["high" + way] = self.decay_divide(high, khn)
                kh_res["low" + way] = self.decay_divide(low, khn)
            res_E = pd.DataFrame(columns=list(kh_res.keys()))
            res_E = res_E.append(kh_res, ignore_index=True)
            resl = res_l.append(res_E)
        res = pd.merge(res_l[0], res_l[1], how="left", on="custid")
        return res.iloc[0]

    # 计算全体新股偏好
    def get_XG_dict(self, sta=False):
        res_E = pickle.load(open('dict/xg_dict', 'rb'))
        # res_E = pd.DataFrame(columns=['custid', 'XGPH'])
        custids = self.custids
        khhs = set(res_E['custid'])
        khh = custids-khhs
        print(len(khh))
        for i, kh in enumerate(khh):
            kh_res = {}
            kh_res["custid"] = kh
            kh_res["XGPH"] = self.get_XGPH(kh)
            res_E = res_E.append(kh_res, ignore_index=True)
        # print res_E.head()
        pickle.dump(res_E, open('dict/xg_dict', 'wb'))

        if not sta:
            return res_E
        else:
            res_B = res_E
            res_B["XGPH"] = standardize(res_B['XGPH'])
            key_n = int(res_B.shape[0] * 0.2)
            res = res_B.sort_values(by="XGPH", ascending=False)
            key = res.iloc[key_n, 0]
            self.X_sta = res_E[res_E["custid"] == key].loc[:, "XGPH"].iloc[0]
            return res

    # 新股偏好
    def get_XGPH(self, custid):
        '''sql = ("select distinct custid, bsflag, market, orderdate, stkcode, stkeffect "
               "from tcl_logasset where custid=%s AND matchqty > 0 order by orderdate asc" % str(custid))
        ms = self.ms.ExecQuery(sql)
        datas = pd.DataFrame(ms, columns=["custid", "bsflag", "market", "orderdate", "stkcode", "stkeffect"])
        kh_res = {}
        kh_res["custid"] = int(custid)
        df = datas
        if df.shape[0] == 0:
            kh_pre = -1
        else:
            kh_pre = float(float(df[df["bsflag"] == "0P"].shape[0]) / float(df.shape[0]))
        kh_res["XGPH"] = kh_pre
        sta = max(self.get_XG_sta(), 0.01)
        if kh_res["XGPH"] != -1:
            if kh_res["XGPH"] > sta:
                kh_res["XGPH"] = 1
            else:
                kh_res["XGPH"] = 0
        else:
            kh_res["XGPH"] = -1'''
        datas = self.get_logdata(custid)
        df = datas[datas["stkeffect"] > 0.00]
        call_num = df[df["bsflag"] == "0P"].shape[0]
        call_all = df.shape[0]

        return self.decay_divide(call_num, call_all)

    '''
    # B股偏好
    def get_BGPHcustid(self, custid):
    	datas = pd.read_excel(self.file)
    	N = ["custid", "bsflag", "matchamt", "stktype", "moneytype"]
    	datas = datas[datas["stkeffect"]<0.00]
    	datas = datas.loc[:, N]
    	datas = datas[datas["matchamt"] != 0]
    	each_kh = {}
    	each_kh["custid"] = int(custid)
    	df = datas[datas['custid'] == int(custid)]
    	dfn = df.shape[0]
    	all_money = df.loc[:, "matchamt"].sum()
    	if (dfn == 0) & (all_money == 0.00):
    		each_kh["FB"] = -1
    		each_kh["B"] = -1
    	else:
    		dc = df[df["stktype"] == "B0"]                                            #存在问题！！！！！
    		each_kh["B"] = float(float(dc.loc[:, "matchamt"].sum()) / float(all_money))
    		each_kh["FB"] = 1 - each_kh["B"]
    	return each_kh
    '''
    '''
    # B股情况
    def get_BGQK(self, custid):
    	datas = pd.read_excel(self.file)
    	N = ["custid", "bsflag", "matchamt", "stktype", "moneytype"]
    	datas = datas[datas["stkeffect"]<0.00]
    	datas = datas.loc[:, N]
    	datas = datas[datas["matchamt"] != 0]
    	each_kh = {}
    	each_kh["custid"] = int(custid)
    	df = datas[datas['custid'] == int(custid)]
    	all_money = float(df.loc[:, "matchamt"].sum())
    	df = df[df["stktype"] == "B0"]                                         #存在问题！！！！！
    	dfn = df.shape[0]
    	all_B_money = float(df.loc[:, "matchamt"].sum())
    	if (dfn == 0) & (all_B_money == 0.00):
    		each_kh["FB"] = 1.00
    		each_kh["HB"] = 0.00
    		each_kh["DB"] = 0.00
    	else:
    		each_kh["FB"] = float(float(df[df["stktype"] != "B0"].loc[:, "matchamt"].sum()) / float(all_money))  #存在问题！！！！！
    		each_kh["HB"] = float(float(df[df["moneytype"] == "HKD"].loc[:, "matchamt"].sum()) / float(all_B_money))        #存在问题！！！！！
    		each_kh["DB"] = float(1 - each_kh["HB"]) 
    	return each_kh

    '''
    # ST股上分位数
    def get_ST_dict(self, sta=False):
        res_E = pickle.load(open('dict/st_dict', 'rb'))
        # res_E = pd.DataFrame(columns=['custid', 'ST'])
        custids = self.custids
        khhs = set(res_E['custid'])
        khh = custids - khhs
        for i, kh in enumerate(khh):
            kh_res = {}
            kh_res["custid"] = kh
            kh_res["ST"] = self.get_STPH(kh)
            res_E = res_E.append(kh_res, ignore_index=True)
        # print res_E.head()
        pickle.dump(res_E, open('dict/st_dict', 'wb'))
        if not sta:
            return res_E
        else:
            res_B = res_E
            res_B["ST"] = standardize(res_B['ST'])
            key_n = int(res_B.shape[0] * 0.2)
            res = res_B.sort_values(by="ST", ascending=False)
            key = res.iloc[key_n, 0]
            self.S_sta = res_E[res_E["custid"] == key].loc[:, "XGPH"].iloc[0]
            self.S_flag = 1
            return res

    # ST股偏好
    def get_STPH(self, custid):
        df = self.get_logdata(custid)
        all_money = float(df["matchamt"].sum())
        dfn = df.shape[0]
        if dfn == 0:
            return -1
        else:
            stn = 0.00
            for i in range(dfn):
                if u'ST' in df['stkname'].iloc[i]:
                    stn += float(df['matchamt'].iloc[i])
            return self.decay_divide(stn, all_money)

    # 次新股上分位数
    def get_CX_sta(self, basic_stk):
        sql = ("select distinct custid, bsflag, matchamt, orderdate, stkcode, stkeffect "
               "from tcl_logasset where matchqty > 0 order by orderdate asc")
        ms = self.ms.ExecQuery(sql)
        datas = pd.DataFrame(ms, columns=["custid", "bsflag", "matchamt", "orderdate", "stkcode", "stkeffect"])

        if self.C_flag == 0:
            khh = self.khhs
            res_E = pd.DataFrame(columns=["custid", "CX"])
            for i, kh in enumerate(khh):
                each_kh = {}
                each_kh["custid"] = int(kh)
                df = datas[datas['custid'] == kh]
                all_money = float(df["matchamt"].sum())
                dfn = df.shape[0]
                if (dfn != 0) & (all_money != 0.00):
                    CX_money = 0.00
                    for n in range(dfn):

                        code = df['stkcode'].iloc[n]
                        buy_year = int(str(df['orderdate'].iloc[n])[:4])
                        bt = basic_stk[basic_stk["code"] == int(code)]
                        if len(bt) == 0:
                            continue
                        mkt_year = int(str(bt['timeToMarket'].iloc[0])[:4])
                        if buy_year - mkt_year > 1:
                            CX_money += float(df[df["stkcode"] == code].loc[:, "matchamt"].iloc[0])
                        else:
                            continue
                    each_kh["CX"] = float(CX_money / all_money)
                else:
                    each_kh["CX"] = 0.00
                res_E = res_E.append(each_kh, ignore_index=True)

            res_B = res_E
            mean = res_B["CX"].mean()
            std = res_B["CX"].std()
            res_B["CX"] = np.asarray(((np.asarray(res_B["CX"]) - mean) / std), dtype=np.float)
            key_n = int(res_B.shape[0] * 0.2)
            res = res_B.sort_values(by="CX", ascending=False)
            key = res.iloc[key_n, 0]
            self.C_sta = res_E[res_E["custid"] == int(key)].loc[:, "CX"].iloc[0]
            self.C_flag = 1
        return self.C_sta

    # 次新股偏好
    def get_CXPH(self, custid):
        df = self.get_logdata(custid)
        df = df[df['stkeffect'] > 0]
        basic_stk = pd.read_csv("datas/basic.csv")
        basic_stk = basic_stk.loc[:, ["code", "timeToMarket"]]
        stocks_f = open(u"datas/次新股.txt")
        stocks = stocks_f.read().split(" ")[:-1]
        for j in range(len(stocks)):
            stocks[j] = int(stocks[j])
        each_kh = {}
        each_kh["custid"] = int(custid)
        all_money = max(float(df.loc[:, "matchamt"].sum()), 0.01)
        dfn = df.shape[0]
        if (dfn == 0) or (all_money == 0.00):
            return -1
        else:
            CX_money = 0.0
            for n in range(dfn):

                code = int(df['stkcode'].iloc[n])
                buy_year = int(str(df['orderdate'].iloc[n])[:4])
                bt = basic_stk[basic_stk["code"] == code]
                if len(bt) == 0:
                    continue
                mkt_year = int(str(bt['timeToMarket'].iloc[0])[:4])
                if buy_year - mkt_year == 1:
                    CX_money += float(df['matchamt'].iloc[n])
                else:
                    continue
            return self.decay_divide(CX_money, all_money)
            '''each_kh["CX"] = float(CX_money / all_money)
        sta = max(self.get_CX_sta(basic_stk), 0.01)
        if each_kh["CX"] != -1:
            if each_kh["CX"] > sta:
                each_kh["CX"] = 1
            else:
                each_kh["CX"] = 0
        else:
            each_kh["CX"] = -1
        return each_kh'''

    # 可转债上分位数
    def get_KZZ_sta(self):
        sql = ("select distinct custid, bsflag, matchamt, orderdate, stkname, stkeffect "
               "from tcl_logasset where matchqty > 0 order by orderdate asc")
        ms = self.ms.ExecQuery(sql)
        datas = pd.DataFrame(ms, columns=["custid", "bsflag", "matchamt", "orderdate", "stkname", "stkeffect"])

        s = u'转债'
        if self.K_flag == 0:
            khh = self.khhs
            res_E = pd.DataFrame(columns=['custid', 'KZZ'])
            for j, kh in enumerate(khh):
                each_kh = {}
                df = datas[datas['custid'] == kh]
                all_money = float(df.loc[:, "matchamt"].sum())
                dfn = df.shape[0]
                if (dfn != 0) & (all_money != 0.00):
                    stn = 0.00
                    each_kh["custid"] = int(kh)
                    for i in range(dfn):
                        if s in df['stkname'].iloc[i]:   #如果在判断中出现报错，则将s=u'转债‘改为s="转债“
                            stn += df['matchamt'].iloc[i]

                    each_kh["KZZ"] = float(stn / all_money)
                    res_E = res_E.append(each_kh, ignore_index=True)

            res_B = res_E[res_E["KZZ"] != -1]
            mean = res_B["KZZ"].mean()
            std = max(res_B["KZZ"].std(), 0.01)
            res_B["KZZ"] = np.asarray(((np.asarray(res_B["KZZ"]) - mean) / std), dtype=np.float)
            key_n = int(res_B.shape[0] * 0.2)
            res = res_B.sort_values(by="KZZ", ascending=False)
            key = res.iloc[key_n, 0]
            self.K_sta = res_E[res_E["custid"] == int(key)].loc[:, "KZZ"].iloc[0]
            self.K_flag = 1
        return self.K_sta

    # 可转债偏好
    def get_KZPH(self, custid):
        df = self.get_logdata(custid)
        df = df[df['stkeffect'] > 0]
        s = u'转债'
        each_kh = {}
        each_kh["custid"] = int(custid)
        all_money = float(df.loc[:, "matchamt"].sum())
        dfn = df.shape[0]
        if (dfn == 0) or (all_money == 0.00):
            return -1
        else:
            stn = 0.00
            for i in range(dfn):
                if s in df['stkname'].iloc[i]:
                    stn += df['matchamt'].iloc[i]
            each_kh["KZZ"] = float(stn / all_money)
            return self.decay_divide(stn, all_money)

        # 中小板上分位数

    def get_ZXB_sta(self):
        sql = ("select distinct custid, bsflag, matchamt, stkcode, stkeffect "
                "from tcl_logasset where matchqty > 0 order by orderdate asc")
        ms = self.ms.ExecQuery(sql)
        datas = pd.DataFrame(ms, columns=["custid", "bsflag", "matchamt", "stkcode","stkeffect"])

        if self.Z_flag == 0:
            khh = self.khhs
            res_E = pd.DataFrame(columns=['custid', 'ZXB'])
            for j, kh in enumerate(khh):
                each_kh = {}
                df = datas[datas['custid'] == kh]
                all_money = float(df.loc[:, "matchamt"].sum())
                dfn = df.shape[0]
                if (dfn != 0) & (all_money != 0.00):
                    stn = 0.00
                    each_kh["custid"] = int(kh)
                    for i in range(dfn):
                        if 1999 < df.iloc[i, 3] < 3000:
                            stn += df['matchamt'].iloc[i]

                    each_kh["ZXB"] = float(stn / all_money)
                    res_E = res_E.append(each_kh, ignore_index=True)

            res_B = res_E[res_E["ZXB"] != -1]
            mean = res_B["ZXB"].mean()
            std = max(res_B["ZXB"].std(), 0.01)
            res_B["ZXB"] = np.asarray(((np.asarray(res_B["ZXB"]) - mean) / std), dtype=np.float)
            key_n = int(res_B.shape[0] * 0.2)
            res = res_B.sort_values(by="ZXB", ascending=False)
            key = res.iloc[key_n, 0]
            self.Z_sta = res_E[res_E["custid"] == int(key)].loc[:, "ZXB"].iloc[0]
            self.Z_flag = 1
        return self.Z_sta

    # 中小板偏好
    def get_ZXBPH(self, custid):
        df = self.get_logdata(custid)
        df = df[df['stkeffect'] > 0]
        df = df.dropna(how='any', axis=0)
        each_kh = {}
        each_kh["custid"] = int(custid)
        all_money = float(df.loc[:, "matchamt"].sum())
        dfn = df.shape[0]
        if (dfn == 0) or (all_money == 0.00):
            return -1
        else:
            stn = 0.00
            for i in range(dfn):
                if 1999 < df['stkcode'].iloc[i] < 3000:
                    stn += float(df['matchamt'].iloc[i])
            return self.decay_divide(stn, all_money)

    # 永远满仓上分位数
    def get_YYMC_sta(self):
        sql = ("select distinct custid,fundbal,marketvalue,busi_date "
                "from tcl_fundasset order by busi_date asc")
        ms = self.ms.ExecQuery(sql)
        datas = pd.DataFrame(ms, columns=["custid", "fundbal", "marketvalue", "busi_date"])

        if self.Y_flag == 0:
            khh = self.khhs
            res_E = pd.DataFrame(columns=['custid', 'YYMC'])
            for j, kh in enumerate(khh):
                each_kh = {}
                df = datas[datas['custid'] == kh]
                dfn = df.shape[0]
                if dfn != 0:
                    days = 0
                    bal = 0.00
                    for i in range(dfn):
                        d1 = datetime.datetime.strptime(str(df.iloc[i, 3])[:-2], '%Y%m%d')
                        if i < dfn - 1:
                            d2 = datetime.datetime.strptime(str(df.iloc[i + 1, 3])[:-2], '%Y%m%d')
                        else:
                            d2 = datetime.datetime.strptime("20171001", '%Y%m%d')
                            # d2=datetime.datetime.now()
                        val_day = (d2 - d1).days
                        days += val_day
                        bal += (df.iloc[i, 2] / (df.iloc[i, 2] + df.iloc[i, 1])) * float(val_day)
                    each_kh["YYMC"] = float(bal / float(days))
                    res_E = res_E.append(each_kh, ignore_index=True)

            res_B = res_E[res_E["YYMC"] != -1]
            mean = res_B["YYMC"].mean()
            std = max(res_B["YYMC"].std(), 0.01)
            res_B["YYMC"] = np.asarray(((np.asarray(res_B["YYMC"]) - mean) / std), dtype=np.float)
            key_n = int(res_B.shape[0] * 0.2)
            res = res_B.sort_values(by="YYMC", ascending=False)
            key = res.iloc[key_n, 0]
            self.Z_sta = res_E[res_E["custid"] == int(key)].loc[:, "YYMC"].iloc[0]
            self.Z_flag = 1
        return self.Z_sta

    # 永远满仓偏好
    def get_YYMCPH(self, custid):
        df = self.get_funddata(custid)
        df = df.dropna(how='any', axis=0)

        each_kh = {}
        each_kh["custid"] = int(custid)
        dfn = df.shape[0]
        if dfn == 0:
            return -1
        else:
            days = 0
            bal = 0
            for i in range(dfn):
                d1 = datetime.datetime.strptime(str(df['busi_date'].iloc[i]), '%Y%m%d')
                if i < dfn - 1:
                    d2 = datetime.datetime.strptime(str(df['busi_date'].iloc[i + 1]), '%Y%m%d')
                else:
                    d2 = datetime.datetime.strptime(self.ed, '%Y%m%d')
                    # d2 = datetime.datetime.now()

                val_day = (d2 - d1).days
                days += val_day
                fund = max(0.0001, float(df['marketvalue'].iloc[i] + df['fundbal'].iloc[i]))
                bal += float(float(df['marketvalue'].iloc[i]) / fund) * float(val_day)
            return self.decay_divide(bal, days)
            '''each_kh["YYMC"] = float(bal / float(days))

        sta = max(self.get_YYMC_sta(), 0.01)
        if each_kh["YYMC"] != -1:
 gao           if each_kh["YYMC"] > sta:
                each_kh["YYMC"] = 1
            else:
                each_kh["YYMC"] = 0
        else:
            each_kh["YYMC"] = -1
        return each_kh'''

    # 注意力驱动偏好
    def get_ZJLQD(self, custid):
        df = self.get_logdata(custid)
        df = df.dropna(how='any', axis=0)
        BS_list = pd.read_csv("datas/BS_list.csv")
        res_al = []
        dataB = df[df["stkeffect"] > 0.00]
        dataS = df[df["stkeffect"] < 0.00]
        alW = [dataB, dataS]
        each_kh = {}
        each_kh["custid"] = int(custid)
        for i, data in enumerate(alW):
            if i == 0:
                way = "B"
            else:
                way = "S"
            dfn = data.shape[0]
            each_kh["ZYLQD" + way] = 0.00
            if dfn == 0:
                each_kh["ZYLQD" + way] = -1
            else:
                for i in range(dfn):  # 遍历该用户的每条交易记录
                    each_data = data.iloc[i, :]
                    record = BS_list[(BS_list["newsPublishDate"] == each_data["orderdate"])
                                     & (BS_list["ticker"] == int(each_data["stkcode"]))
                                     & (BS_list["way"] == way)]
                    each_kh["ZYLQD" + way] += float(record.shape[0])

                each_kh["ZYLQD" + way] = self.decay_divide(float(each_kh["ZYLQD" + way]) , float(dfn))

        return each_kh['ZYLQDB'], each_kh['ZYLQDS']
    
    
    def tor_pref(self, custid):
        return invest_pref(self.logasset, custid, 'daily', 'tor', 3)

    def fund_pref(self, custid):
        return invest_pref(self.logasset, custid, 'quarterly', 'fund', 2)

    def otstd_pref(self, custid):
        return invest_pref(self.logasset, custid, 'static', 'outstanding', 3)

    def op_pref(self, custid):
        return invest_pref(self.logasset, custid, 'daily', 'op', 3)

    def pe_pref(self, custid):
        return invest_pref(self.logasset, custid, 'static', 'pe', 3)

    def pb_pref(self, custid):
        return invest_pref(self.logasset, custid, 'static', 'pb', 3)

    def npr_pref(self, custid):
        return invest_pref(self.logasset, custid, 'static', 'npr', 2)

    def top_pref(self, custid):
        return invest_pref(self.logasset, custid, 'daily', 'top', 2)

    def macd_pref(self, custid):
        return operate_pref(self.logasset, custid, 'MACD')

    def kdj_pref(self, custid):
        return operate_pref(self.logasset, custid, 'KDJ')

    def industry_pref(self, custid):
        return invest_pref(self.logasset, custid, 'static', 'industry', 49)

    def concept_pref(self, custid):
        return invest_pref(self.logasset, custid, 'static', 'concept', 99)

if __name__ == '__main__':
    database = 'Yichuang'
    log_num = 1
    file = pd.read_excel('selected_behavior_value_label.xlsx')
    users = Users(database=database, endtime='20170915')
    print(file.columns)
    ad_khhs = set(file.loc[:,0])
    print(ad_khhs)
    custids = users.custids
    log = fund = stk = pd.DataFrame()
    l = len(ad_khhs)
    st = time.time()
    i = 0
    for i, cus in enumerate(ad_khhs):
        # custid = '115001237'
        custid = str(cus)
        print(custid)
        logdata = users.get_logdata(custid)

        funddata = users.get_funddata(custid)

        stkdata = users.get_stkdata(custid)
        #print(len(logdata), len(funddata))
        print(i, l)
        log = log.append(logdata, ignore_index=True)
        fund = fund.append(funddata, ignore_index=True)
        stk = stk.append(stkdata, ignore_index=True)
    log.to_csv('test/log.csv')
    fund.to_csv('test/fund.csv')
    stk.to_csv('test/stk.csv')